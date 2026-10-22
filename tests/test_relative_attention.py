"""Explicit distance bias remains active during training and codec inference."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from codec.relative_attention import RelativePositionTransformerEncoderLayer
from codec.suffix_lora_adapter import install_suffix_lora


class RelativeAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(836)

    def layer(self, **overrides):
        options = dict(d_model=8, nhead=2, dim_feedforward=16, dropout=0,
                       activation="gelu", batch_first=True, norm_first=True,
                       context_length=4)
        options.update(overrides)
        return RelativePositionTransformerEncoderLayer(**options)

    def test_zero_bias_matches_baseline_and_parameter_names(self):
        for norm_first in (True, False):
            layer = self.layer(norm_first=norm_first).eval()
            baseline = nn.TransformerEncoderLayer(
                8, 2, dim_feedforward=16, dropout=0, activation="gelu",
                batch_first=True, norm_first=norm_first,
            ).eval()
            shared = {k: v for k, v in layer.state_dict().items()
                      if k != "relative_position_bias"}
            baseline.load_state_dict(shared)
            src = torch.randn(3, 7, 8)
            indices = torch.arange(7)
            distance = indices[:, None] - indices[None, :]
            mask = (distance < 0) | (distance >= 4)
            torch.testing.assert_close(layer(src), baseline(src, src_mask=mask),
                                       rtol=1e-6, atol=1e-6)
            self.assertEqual(layer.relative_position_bias.shape, (2, 4))

    def test_nonzero_bias_changes_output_and_receives_gradients(self):
        layer = self.layer()
        src = torch.randn(2, 7, 8, requires_grad=True)
        before = layer(src)
        with torch.no_grad():
            layer.relative_position_bias[0].copy_(torch.tensor([0., 2., -1., 1.]))
            layer.relative_position_bias[1].copy_(torch.tensor([1., -2., 2., 0.]))
        after = layer(src)
        self.assertGreater((after - before).abs().max().item(), 1e-3)
        after.square().mean().backward()
        gradient = layer.relative_position_bias.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 1e-5)
        self.assertTrue(torch.isfinite(src.grad).all())

    def test_eval_no_grad_and_inference_keep_the_same_bias_path(self):
        layer = self.layer().eval()
        with torch.no_grad():
            layer.relative_position_bias.normal_()
        src = torch.randn(2, 7, 8)
        expected = layer(src).detach()
        with mock.patch("torch._transformer_encoder_layer_fwd", side_effect=AssertionError("fused encoder")), \
                mock.patch("torch._native_multi_head_attention", side_effect=AssertionError("native MHA")):
            with torch.no_grad():
                torch.testing.assert_close(layer(src), expected, rtol=0, atol=0)
            with torch.inference_mode():
                torch.testing.assert_close(layer(src), expected, rtol=0, atol=0)

    def test_future_window_and_extra_masks_are_never_reopened(self):
        layer = self.layer(context_length=3).eval()
        with torch.no_grad():
            layer.relative_position_bias.fill_(1e3)
        src = torch.randn(2, 8, 8)
        before = layer(src)
        changed = src.clone()
        changed[:, 6:] += torch.randn(2, 2, 8) * 100
        torch.testing.assert_close(layer(changed)[:, :6], before[:, :6], rtol=0, atol=0)
        changed = src.clone()
        changed[:, :3] += torch.randn(2, 3, 8) * 100
        torch.testing.assert_close(layer(changed)[:, 5:], before[:, 5:], rtol=0, atol=0)
        # A caller can narrow attention, including one batch/head, but cannot
        # override this layer's causal window by supplying positive infinity.
        external = torch.zeros(4, 8, 8)
        external[0, 5, 4] = float("-inf")
        external[:, 0, 7] = float("inf")
        original = external.clone()
        combined = layer._relative_mask(src, external)
        self.assertTrue(torch.isneginf(combined[:, 0, 7]).all())
        self.assertTrue(torch.isneginf(combined[0, 5, 4]))
        torch.testing.assert_close(external, original, rtol=0, atol=0)

    def test_boolean_float_masks_and_padding_are_equivalent(self):
        layer = self.layer().eval()
        with torch.no_grad():
            layer.relative_position_bias.normal_()
        src = torch.randn(2, 7, 8)
        positions = torch.arange(7)
        mask = positions[:, None] < positions[None, :]
        # Never mask every key of an active query. Right padding is ignored
        # only when scoring the active prefix of each read.
        padding = torch.tensor([[False] * 7, [False] * 4 + [True] * 3])
        float_mask = torch.zeros(7, 7).masked_fill(mask, float("-inf"))
        float_padding = torch.zeros(2, 7).masked_fill(padding, float("-inf"))
        bool_result = layer(src, src_mask=mask, src_key_padding_mask=padding)
        float_result = layer(src, src_mask=float_mask, src_key_padding_mask=float_padding)
        torch.testing.assert_close(bool_result, float_result, rtol=0, atol=0)
        torch.testing.assert_close(
            bool_result[0], layer(src[:1], src_mask=mask)[0], rtol=1e-6, atol=1e-6,
        )
        torch.testing.assert_close(
            bool_result[1, :4], layer(src[1:2, :4])[0], rtol=1e-6, atol=1e-6,
        )
        per_head = mask.unsqueeze(0).expand(4, -1, -1).clone()
        per_head[0, 3, 2] = True
        per_head_original = per_head.clone()
        three_dimensional = layer(src, src_mask=per_head)
        torch.testing.assert_close(three_dimensional[1], layer(src[1:2])[0],
                                   rtol=1e-6, atol=1e-6)
        self.assertGreater((three_dimensional[0, 3] - layer(src[:1])[0, 3]).abs().max().item(), 0)
        self.assertTrue(torch.equal(per_head, per_head_original))

    def test_unbatched_and_sequence_first_inputs(self):
        layer = self.layer().eval()
        with torch.no_grad():
            layer.relative_position_bias.normal_()
        sequence_first = self.layer(batch_first=False).eval()
        sequence_first.load_state_dict(layer.state_dict())
        src = torch.randn(2, 7, 8)
        expected = layer(src)
        torch.testing.assert_close(
            sequence_first(src.transpose(0, 1)).transpose(0, 1), expected, rtol=0, atol=0,
        )
        torch.testing.assert_close(layer(src[0]), expected[0], rtol=1e-6, atol=1e-6)

    def test_suffix_lora_parameters_change_predictions_and_receive_gradients(self):
        class Model(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.config = SimpleNamespace(num_layers=1, d_model=8, feedforward_dim=16)
                self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

        model = Model(self.layer()).eval().requires_grad_(False)
        layout = install_suffix_lora(model, 1, 2, 88)
        src = torch.randn(2, 7, 8)
        before = model.transformer(src).detach()
        for item in layout:
            module_name, name = item["name"].rsplit(".", 1)
            factors = model.get_submodule(module_name).parametrizations[name][0]
            with torch.no_grad():
                factors.b.normal_(std=.1)
        changed = model.transformer(src)
        self.assertGreater((changed - before).abs().max().item(), 1e-4)
        changed.square().mean().backward()
        for item in layout:
            module_name, name = item["name"].rsplit(".", 1)
            factors = model.get_submodule(module_name).parametrizations[name][0]
            self.assertIsNotNone(factors.b.grad)
            self.assertGreater(factors.b.grad.abs().sum().item(), 0)
        with torch.no_grad():
            torch.testing.assert_close(model.transformer(src), changed.detach(), rtol=0, atol=0)

    def test_rejects_bad_context_and_masks(self):
        for context in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "context_length"):
                self.layer(context_length=context)
        layer = self.layer()
        src = torch.randn(2, 7, 8)
        for mask in (torch.zeros(7, 7, dtype=torch.long), torch.zeros(2, 7, 7)):
            with self.assertRaises(ValueError):
                layer(src, src_mask=mask)


if __name__ == "__main__":
    unittest.main()
