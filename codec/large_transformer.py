"""Training execution for a large, checkpoint-compatible quality Transformer.

The statistical model and state-dict names are those of ``DirectQualityTransformer``.
Only training attention execution and activation storage change. In particular,
``context_length`` remains a per-layer sliding window, including on reads longer
than the window; it is never treated as a data crop or enlarged to the read length.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata as package_metadata
import inspect
import re
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import DirectQualityModelConfig, DirectQualityTransformer


def _flash2_version() -> str:
    try:
        version = package_metadata.version("flash-attn")
    except package_metadata.PackageNotFoundError as exc:
        raise RuntimeError("flash2 requires an explicit flash-attn 2.3+ installation") from exc
    parsed = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if parsed is None or int(parsed[1]) != 2 or int(parsed[2]) < 3:
        raise RuntimeError(f"flash2 requires flash-attn >=2.3,<3 with local windows; found {version!r}")
    return version


@lru_cache(maxsize=1)
def _load_flash2():
    version = _flash2_version()
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError) as exc:
        raise RuntimeError("flash2 could not load its compatible CUDA extension") from exc
    try:
        parameters = inspect.signature(flash_attn_func).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("flash2 cannot verify its local-window function API") from exc
    if "window_size" not in parameters or "causal" not in parameters:
        raise RuntimeError("flash2 requires flash_attn_func with explicit window_size and causal arguments")
    return flash_attn_func, version


def _flash2_attention(query: Tensor, key: Tensor, value: Tensor, *,
                      context_length: int, dropout_p: float) -> Tensor:
    if not query.is_cuda or query.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError("FlashAttention2 training requires CUDA BF16/FP16 autocast")
    attention, _ = _load_flash2()
    # Native FA2 expects [B,T,H,D]. The local left window is inclusive, so
    # context_length C corresponds to window_size=(C-1,0), not (C,0).
    attended = attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        dropout_p=dropout_p, softmax_scale=None, causal=True,
        window_size=(context_length - 1, 0),
    )
    return attended.transpose(1, 2)


def _sdpa(query: Tensor, key: Tensor, value: Tensor, *, dropout_p: float,
          is_causal: bool, backend: str) -> Tensor:
    if backend == "flash":
        if not query.is_cuda or query.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("FlashAttention training requires CUDA BF16/FP16 autocast")
        # Unsupported CUDA architectures/shapes/dropout configurations must fail,
        # rather than silently changing the requested attention implementation.
        options = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    elif backend == "math":
        options = dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    else:
        raise ValueError(f"unsupported training attention backend {backend!r}")
    with torch.backends.cuda.sdp_kernel(**options):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout_p,
            is_causal=is_causal,
        )


def _local_causal_attention(query: Tensor, key: Tensor, value: Tensor, *,
                            context_length: int, tail_query_chunk: int,
                            dropout_p: float, backend: str) -> Tensor:
    """Apply the exact ``0 <= query-key < context_length`` attention mask.

    The first C queries use one square causal attention call. Each later query
    attends its own C-key window, batched as independent one-query problems.
    The one-query calls are *noncausal*: all gathered keys precede that query.
    Using is_causal=True there would incorrectly keep only the first key under
    PyTorch 2.0's upper-left convention for nonsquare causal attention.

    Overlapping K/V windows are materialized only for ``tail_query_chunk``
    queries at a time. This bounds temporary memory without changing the mask.
    Dropout samples independent attention edges just as dense attention does;
    its RNG ordering differs, so train-mode bit identity is not claimed.
    """
    if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape:
        raise ValueError("local attention requires equal [batch, heads, time, dim] Q/K/V")
    if context_length <= 0 or tail_query_chunk <= 0:
        raise ValueError("attention context and query chunk must be positive")
    batch, heads, length, head_dim = query.shape
    if length == 0:
        return query
    first = min(length, context_length)
    pieces = [_sdpa(query[:, :, :first], key[:, :, :first], value[:, :, :first],
                    dropout_p=dropout_p, is_causal=True, backend=backend)]
    if length > context_length:
        # unfold yields [B,H,window_start,D,C], with overlapping storage.
        key_windows = key.unfold(2, context_length, 1)
        value_windows = value.unfold(2, context_length, 1)
        for begin in range(context_length, length, tail_query_chunk):
            end = min(length, begin + tail_query_chunk)
            count = end - begin
            start = begin - context_length + 1
            q = query[:, :, begin:end].transpose(1, 2).reshape(
                batch * count, heads, 1, head_dim)
            k = key_windows[:, :, start:start + count].permute(0, 2, 1, 4, 3).reshape(
                batch * count, heads, context_length, head_dim)
            v = value_windows[:, :, start:start + count].permute(0, 2, 1, 4, 3).reshape(
                batch * count, heads, context_length, head_dim)
            attended = _sdpa(q, k, v, dropout_p=dropout_p,
                             is_causal=False, backend=backend)
            pieces.append(attended.reshape(batch, count, heads, head_dim).transpose(1, 2))
    return torch.cat(pieces, dim=2)


class _TrainingEncoderLayer(nn.TransformerEncoderLayer):
    """Reuse all original parameters and execute only training attention here."""

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor], *args, **kwargs) -> Tensor:
        if not self.training:
            return super()._sa_block(x, attn_mask, key_padding_mask, *args, **kwargs)
        if key_padding_mask is not None:
            raise ValueError("quality attention expects right padding and no explicit key-padding mask")
        if attn_mask is None or attn_mask.shape != (x.shape[1], x.shape[1]):
            raise ValueError("quality attention requires the model's sliding causal mask")
        attention = self.self_attn
        if not attention.batch_first or not attention._qkv_same_embed_dim:
            raise ValueError("unsupported quality attention projection layout")
        batch, length, dimension = x.shape
        head_dim = dimension // attention.num_heads
        qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        qkv = qkv.reshape(batch, length, 3, attention.num_heads, head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        if self.quality_attention_backend == "flash2":
            attended = _flash2_attention(query, key, value,
                context_length=self.quality_context_length, dropout_p=attention.dropout)
        else:
            attended = _local_causal_attention(
                query, key, value, context_length=self.quality_context_length,
                tail_query_chunk=self.quality_tail_query_chunk,
                dropout_p=attention.dropout, backend=self.quality_attention_backend,
            )
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        return self.dropout1(F.linear(attended, attention.out_proj.weight,
                                      attention.out_proj.bias))


class _TrainingEncoder(nn.TransformerEncoder):
    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None) -> Tensor:
        if not self.training:
            return super().forward(src, mask=mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   is_causal=is_causal)
        if src_key_padding_mask is not None:
            raise ValueError("quality attention uses decoder-synchronous right padding")
        output = src
        for layer in self.layers:
            if self.quality_activation_checkpointing and torch.is_grad_enabled():
                # Bind layer now: backward recomputation happens after this loop.
                def forward_layer(hidden, current_layer=layer):
                    return current_layer(hidden, src_mask=mask)
                output = checkpoint(forward_layer, output, use_reentrant=True,
                                    preserve_rng_state=True)
            else:
                output = layer(output, src_mask=mask)
        return self.norm(output) if self.norm is not None else output


def _initialize_layer_independently(layer: nn.TransformerEncoderLayer) -> None:
    """Repeat the standard PyTorch constructors' parameter distributions.

    TransformerEncoder deep-copies one initialized layer. Reinitializing each
    copy makes the draws independent, without residual scaling or new designs.
    MultiheadAttention resets QKV and biases, but not its out-projection weight,
    hence the separate Linear reset before its own reset.
    """
    layer.self_attn.out_proj.reset_parameters()
    layer.self_attn._reset_parameters()
    layer.linear1.reset_parameters()
    layer.linear2.reset_parameters()
    layer.norm1.reset_parameters()
    layer.norm2.reset_parameters()


class LargeQualityTransformer(DirectQualityTransformer):
    """Plain quality Transformer with explicit flash/math training execution.

    Right-padding proof: for every active query q, its allowed keys satisfy
    k <= q < read_length, so no padded key can affect an active output. Hidden
    values computed at later padded queries are never used by active queries
    in any subsequent causal layer. The original active loss/output masks and
    base CNN remain unchanged.

    In eval mode the original full sliding mask and FP32 math path are used.
    Saved state dicts load strictly into ordinary DirectQualityTransformer;
    training execution choices belong in run metadata, not architecture schema.
    """

    def __init__(self, config: Optional[DirectQualityModelConfig] = None, *,
                 attention_backend: str = "flash", attention_query_chunk_size: int = 16,
                 independent_layer_initialization: bool = True):
        config = config or DirectQualityModelConfig()
        if attention_backend not in ("flash", "flash2", "math"):
            raise ValueError("attention_backend must be flash, flash2 or math")
        if type(attention_query_chunk_size) is not int or attention_query_chunk_size <= 0:
            raise ValueError("attention_query_chunk_size must be a positive integer")
        if type(independent_layer_initialization) is not bool:
            raise ValueError("independent_layer_initialization must be bool")
        if (config.ffn_type != "gelu" or config.base_encoder_type != "local_cnn"
                or config.quality_prefix_features or config.relative_position_bias
                or config.base_cross_attention or config.block_quality_features
                or getattr(config, "short_quality_features", False)
                or getattr(config, "quality_history_cnn", False)
                or getattr(config, "public_read_quality_state", False)):
            raise ValueError("large-model execution currently supports the original plain quality architecture")
        if attention_backend == "flash":
            head_dim = config.d_model // config.num_heads
            if head_dim % 8 or head_dim > 128:
                raise ValueError("PyTorch 2.0 flash requires a head dimension divisible by 8 and at most 128")
        elif attention_backend == "flash2":
            head_dim = config.d_model // config.num_heads
            if head_dim % 8 or head_dim > 256:
                raise ValueError("flash2 requires a head dimension divisible by 8 and at most 256")
        super().__init__(config)
        self.attention_backend = attention_backend
        self.attention_query_chunk_size = attention_query_chunk_size
        self.independent_layer_initialization = independent_layer_initialization
        # These subclasses add execution only: no nested modules or renamed keys.
        self.transformer.__class__ = _TrainingEncoder
        self.transformer.quality_activation_checkpointing = False
        for layer in self.transformer.layers:
            if independent_layer_initialization:
                _initialize_layer_independently(layer)
            layer.__class__ = _TrainingEncoderLayer
            layer.quality_attention_backend = attention_backend
            layer.quality_context_length = config.context_length
            layer.quality_tail_query_chunk = attention_query_chunk_size

    def set_activation_checkpointing(self, enabled: bool = True) -> None:
        if type(enabled) is not bool:
            raise ValueError("activation checkpointing must be bool")
        self.transformer.quality_activation_checkpointing = enabled

    def forward(self, bases, qualities, lengths, active_mask,
                read_states=None, block_features=None):
        return self.forward_full(bases, qualities, lengths, active_mask,
                                 read_states, block_features)

    def training_attention_metadata(self) -> dict:
        values = {
            "backend": self.attention_backend,
            "implementation": "pytorch_scaled_dot_product_attention",
            "torch_version": torch.__version__,
            "flash_fallback_enabled": False,
            "window": "0 <= query-key < context_length",
            "context_length": self.config.context_length,
            "long_read_strategy": "causal_prefix_then_batched_one_query_sliding_windows",
            "tail_query_chunk_size": self.attention_query_chunk_size,
            "eval_attention": "original_float32_math_with_full_sliding_mask",
            "layer_initialization": ("independent_pytorch_default"
                                     if self.independent_layer_initialization
                                     else "pytorch_encoder_cloned_layer"),
        }
        if self.attention_backend == "flash2":
            values.update(
                implementation="flash_attn.flash_attn_func",
                flash_attn_version=_flash2_version(),
                window_size=[self.config.context_length - 1, 0],
                causal=True, long_read_strategy="native_local_window",
                tail_query_chunk_size=None,
            )
        return values

    def _hidden_impl(self, bases, qualities, lengths, active_mask,
                     read_states=None, block_features=None):
        if self.training:
            return super()._hidden_impl(bases, qualities, lengths, active_mask,
                                        read_states, block_features)
        if self.output_head.weight.dtype != torch.float32:
            raise RuntimeError("quality evaluation requires FP32 model parameters")
        with torch.autocast(device_type=bases.device.type, enabled=False), \
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                              enable_mem_efficient=False):
            return super()._hidden_impl(bases, qualities, lengths, active_mask,
                                        read_states, block_features)

    def _forward_impl(self, bases, qualities, lengths, active_mask,
                      read_states=None, block_features=None):
        if self.training:
            return super()._forward_impl(bases, qualities, lengths, active_mask,
                                         read_states, block_features)
        # Include the output projection: guarding only _hidden_impl would let
        # an outer autocast convert the final linear head back to BF16.
        with torch.autocast(device_type=bases.device.type, enabled=False), \
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                              enable_mem_efficient=False):
            return super()._forward_impl(bases, qualities, lengths, active_mask,
                                         read_states, block_features)
