"""Mask, gradient, initialization and checkpoint contracts for large training."""

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from codec.fastq_stream import make_fastq_batch
from codec.large_transformer import (LargeQualityTransformer, _load_flash2,
                                     _local_causal_attention)
from codec.model import (DirectQualityModelConfig, DirectQualityTransformer,
                         fastq_batch_to_tensors, masked_cross_entropy)


def config(**changes):
    values = dict(d_model=32, num_heads=4, num_layers=2, feedforward_dim=64,
                  context_length=8, dropout=0.0)
    values.update(changes)
    return DirectQualityModelConfig(**values)


def tensors(lengths=(19, 7)):
    generator = np.random.default_rng(123)
    batch = make_fastq_batch(
        [generator.integers(0, 5, length, dtype=np.uint8) for length in lengths],
        [generator.integers(0, 42, length, dtype=np.uint8) for length in lengths],
        list(range(len(lengths))), source_name="synthetic.fastq",
    )
    return fastq_batch_to_tensors(batch, torch.device("cpu"))


class LargeTransformerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_large_parameter_count_without_allocating_model_storage(self):
        cfg = DirectQualityModelConfig(d_model=1536, num_heads=24, num_layers=27,
                                       feedforward_dim=3072)
        with torch.device("meta"):
            model = LargeQualityTransformer(cfg)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 510353050)
        self.assertEqual({parameter.device.type for parameter in model.parameters()}, {"meta"})
        self.assertEqual(model.input_projection[0].in_features, 75)
        self.assertEqual(model.output_head.out_features, 42)
        self.assertEqual(type(model.output_head), torch.nn.Linear)

    def test_independent_standard_layer_initialization_and_state_keys(self):
        model = LargeQualityTransformer(config(), attention_backend="math")
        ordinary = DirectQualityTransformer(config())
        self.assertEqual(set(model.state_dict()), set(ordinary.state_dict()))
        ordinary.load_state_dict(model.state_dict(), strict=True)
        first, second = model.transformer.layers
        self.assertFalse(torch.equal(first.self_attn.in_proj_weight,
                                     second.self_attn.in_proj_weight))
        self.assertNotEqual(first.self_attn.in_proj_weight.data_ptr(),
                            second.self_attn.in_proj_weight.data_ptr())
        torch.testing.assert_close(first.norm1.weight, torch.ones_like(first.norm1.weight))
        torch.testing.assert_close(second.self_attn.in_proj_bias,
                                   torch.zeros_like(second.self_attn.in_proj_bias))
        self.assertEqual(model.training_attention_metadata()["layer_initialization"],
                         "independent_pytorch_default")

    def test_local_attention_outputs_and_gradients_match_dense_sliding_mask(self):
        for length, context in ((7, 8), (8, 8), (19, 8), (386, 256)):
            with self.subTest(length=length, context=context):
                qkv = [torch.randn(2, 2, length, 8, requires_grad=True) for _ in range(3)]
                reference_qkv = [value.detach().clone().requires_grad_() for value in qkv]
                position = torch.arange(length)
                distance = position[:, None] - position[None, :]
                allowed = (distance >= 0) & (distance < context)
                actual = _local_causal_attention(*qkv, context_length=context,
                                                 tail_query_chunk=5, dropout_p=0.0,
                                                 backend="math")
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                                    enable_mem_efficient=False):
                    expected = F.scaled_dot_product_attention(
                        *reference_qkv, attn_mask=allowed, dropout_p=0.0)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=5e-7)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                for actual_input, reference_input in zip(qkv, reference_qkv):
                    torch.testing.assert_close(actual_input.grad, reference_input.grad,
                                               rtol=2e-5, atol=3e-6)

    def test_model_train_outputs_and_all_parameter_gradients_match_original(self):
        cfg = config(context_length=8)
        model = LargeQualityTransformer(cfg, attention_backend="math",
                                         attention_query_chunk_size=3)
        original = DirectQualityTransformer(cfg)
        original.load_state_dict(model.state_dict(), strict=True)
        batch = tensors()
        actual, expected = model(**batch), original.forward_full(**batch)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        masked_cross_entropy(actual, batch["qualities"], batch["active_mask"]).backward()
        masked_cross_entropy(expected, batch["qualities"], batch["active_mask"]).backward()
        for name, parameter in model.named_parameters():
            reference = dict(original.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, reference.grad,
                                       rtol=3e-4, atol=3e-7, msg=name)

    def test_eval_is_original_fp32_path_for_mixed_lengths_and_long_reads(self):
        cfg = config(context_length=256)
        model = LargeQualityTransformer(cfg, attention_backend="flash").eval()
        original = DirectQualityTransformer(cfg).eval()
        original.load_state_dict(model.state_dict(), strict=True)
        batch = tensors((386, 150))
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = model(**batch)
        with torch.no_grad():
            expected = original.forward_full(**batch)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.all(actual[~batch["active_mask"]] == 0))

    def test_future_quality_values_do_not_change_earlier_active_logits(self):
        model = LargeQualityTransformer(config(), attention_backend="math")
        batch = tensors()
        changed = dict(batch)
        changed["qualities"] = batch["qualities"].clone()
        changed["qualities"][0, 10:] = (changed["qualities"][0, 10:] + 7) % 42
        first = model(**batch)
        second = model(**changed)
        torch.testing.assert_close(first[0, :11], second[0, :11], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0, 11:], second[0, 11:]))

    def test_activation_checkpoint_preserves_dropout_recomputation(self):
        cfg = config(dropout=0.15)
        direct = LargeQualityTransformer(cfg, attention_backend="math")
        recomputed = copy.deepcopy(direct)
        recomputed.set_activation_checkpointing(True)
        batch = tensors()
        outputs = []
        for model in (direct, recomputed):
            torch.manual_seed(82)
            logits = model(**batch)
            outputs.append(logits.detach())
            masked_cross_entropy(logits, batch["qualities"], batch["active_mask"]).backward()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for name, parameter in direct.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(recomputed.named_parameters())[name].grad,
                                       rtol=0, atol=0, msg=name)

    def test_flash_training_rejects_cpu_instead_of_falling_back(self):
        model = LargeQualityTransformer(config(), attention_backend="flash")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA BF16/FP16"):
            model(**tensors())

    def test_flash2_is_lazy_and_rejects_cpu_training_without_fallback(self):
        with patch("codec.large_transformer._load_flash2") as extension:
            model = LargeQualityTransformer(config(), attention_backend="flash2")
            with self.assertRaisesRegex(RuntimeError, "requires CUDA BF16/FP16"):
                model(**tensors())
            model.eval()
            with torch.no_grad():
                self.assertEqual(model(**tensors()).dtype, torch.float32)
            extension.assert_not_called()

    def test_flash2_metadata_records_exact_native_window_and_validates_version(self):
        model = LargeQualityTransformer(config(), attention_backend="flash2")
        with patch("codec.large_transformer.package_metadata.version", return_value="2.5.5"):
            metadata = model.training_attention_metadata()
        self.assertEqual(metadata["flash_attn_version"], "2.5.5")
        self.assertEqual(metadata["implementation"], "flash_attn.flash_attn_func")
        self.assertEqual(metadata["window_size"], [7, 0])
        self.assertTrue(metadata["causal"])
        self.assertFalse(metadata["flash_fallback_enabled"])
        self.assertEqual(metadata["long_read_strategy"], "native_local_window")
        for version in ("1.0.9", "2.2.4", "3.0.0", "unknown"):
            with self.subTest(version=version), \
                    patch("codec.large_transformer.package_metadata.version", return_value=version):
                with self.assertRaisesRegex(RuntimeError, "requires flash-attn"):
                    model.training_attention_metadata()

    def test_flash2_loader_refuses_functions_without_local_window_api(self):
        self.addCleanup(_load_flash2.cache_clear)
        _load_flash2.cache_clear()

        def legacy_attention(query, key, value, dropout_p=0.0, causal=False):
            raise AssertionError("unsupported stub must never execute")

        def local_attention(query, key, value, dropout_p=0.0, softmax_scale=None,
                            causal=False, window_size=(-1, -1)):
            raise AssertionError("loader does not execute the attention function")

        with patch("codec.large_transformer.package_metadata.version", return_value="2.5.5"):
            with patch.dict(sys.modules, flash_attn=SimpleNamespace(flash_attn_func=legacy_attention)):
                with self.assertRaisesRegex(RuntimeError, "explicit window_size"):
                    _load_flash2()
            with patch.dict(sys.modules, flash_attn=SimpleNamespace(flash_attn_func=local_attention)):
                function, version = _load_flash2()
                self.assertIs(function, local_attention)
                self.assertEqual(version, "2.5.5")


if __name__ == "__main__":
    unittest.main()
