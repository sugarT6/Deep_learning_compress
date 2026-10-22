"""Base-memory isolation, decoder causality, and padding boundary checks."""

from types import SimpleNamespace
import unittest

import torch

from codec.base_memory import BaseMemoryDecoderLayer, BaseMemoryEncoder


def config():
    return SimpleNamespace(
        base_memory_dim=16,
        base_memory_heads=4,
        base_memory_layers=2,
        base_embed_dim=8,
        dropout=0.0,
    )


def position_features(batch, length):
    cycle = torch.arange(length, dtype=torch.float32)[None, :, None]
    return cycle.expand(batch, -1, 3) / 32.0


def causal_mask(length):
    return torch.ones(length, length, dtype=torch.bool).triu(1)


class BaseMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(731)

    def test_base_memory_has_no_quality_input_and_reads_future_bases(self):
        model = BaseMemoryEncoder(config()).eval()
        bases = torch.zeros(1, 12, dtype=torch.long)
        active = torch.ones_like(bases, dtype=torch.bool)
        before = model(bases, active, position_features(1, 12))
        changed = bases.clone()
        changed[:, -1] = 3
        after = model(changed, active, position_features(1, 12))
        self.assertGreater(float((before[:, 0] - after[:, 0]).abs().max()), 1e-5)
        with self.assertRaises(TypeError):
            model(bases, active, position_features(1, 12), qualities=bases)

    def test_memory_padding_length_and_content_do_not_change_active_positions(self):
        model = BaseMemoryEncoder(config()).eval()
        bases = torch.tensor([[0, 1, 2, 3, 4]])
        before = model(bases, torch.ones_like(bases, dtype=torch.bool),
                       position_features(1, 5))
        padded = torch.cat([bases, torch.full((1, 7), 6, dtype=torch.long)], dim=1)
        active = torch.arange(12)[None, :] < 5
        positions = position_features(1, 12)
        positions[:, 5:] = float("nan")
        after = model(padded, active, positions)
        torch.testing.assert_close(before, after[:, :5], atol=1e-6, rtol=1e-5)
        self.assertEqual(int(torch.count_nonzero(after[:, 5:])), 0)
        padded[:, 5:] = 12345
        torch.testing.assert_close(model(padded, active, positions), after,
                                   atol=0.0, rtol=0.0)

    def test_empty_reads_batches_and_widths_are_finite(self):
        model = BaseMemoryEncoder(config()).eval()
        for shape in ((2, 7), (0, 7), (2, 0), (0, 0)):
            with self.subTest(shape=shape):
                bases = torch.full(shape, 6, dtype=torch.long)
                active = torch.zeros(shape, dtype=torch.bool)
                result = model(bases, active, torch.zeros(*shape, 3))
                self.assertEqual(tuple(result.shape), (*shape, 16))
                self.assertTrue(bool(torch.isfinite(result).all()))
                self.assertEqual(int(torch.count_nonzero(result)), 0)

    def test_empty_read_does_not_change_neighboring_read(self):
        model = BaseMemoryEncoder(config()).eval()
        bases = torch.tensor([[0, 1, 2, 3, 4], [6, 6, 6, 6, 6]])
        active = bases != 6
        together = model(bases, active, position_features(2, 5))
        alone = model(bases[:1], active[:1], position_features(1, 5))
        torch.testing.assert_close(together[:1], alone, atol=1e-6, rtol=1e-5)
        self.assertEqual(int(torch.count_nonzero(together[1])), 0)


class BaseMemoryDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(889)
        self.layer = BaseMemoryDecoderLayer(
            d_model=24, nhead=4, dim_feedforward=48,
            dropout=0.0, memory_dim=16,
        ).eval()

    def test_quality_self_attention_is_causal_with_fixed_base_memory(self):
        hidden = torch.randn(2, 11, 24)
        memory = torch.randn(2, 11, 16)
        before = self.layer(hidden, src_mask=causal_mask(11), memory=memory)
        changed = hidden.clone()
        changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 5
        after = self.layer(changed, src_mask=causal_mask(11), memory=memory)
        torch.testing.assert_close(before[:, :6], after[:, :6], atol=0.0, rtol=0.0)
        self.assertGreater(float((before[:, 6:] - after[:, 6:]).abs().max()), 0.1)

    def test_cross_attention_runs_with_and_without_grad_and_receives_gradients(self):
        hidden = torch.randn(2, 7, 24)
        memory = torch.randn(2, 7, 16, requires_grad=True)
        expected = self.layer(hidden, src_mask=causal_mask(7), memory=memory)
        with torch.no_grad():
            inferred = self.layer(hidden, src_mask=causal_mask(7), memory=memory)
            other = self.layer(hidden, src_mask=causal_mask(7), memory=memory + 2)
        torch.testing.assert_close(expected, inferred, atol=1e-6, rtol=1e-5)
        self.assertGreater(float((other - inferred).abs().max()), 0.01)
        expected.square().mean().backward()
        self.assertGreater(float(memory.grad.abs().sum()), 0.0)
        for name in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
            self.assertGreater(float(getattr(self.layer.cross_attn, name).grad.abs().sum()), 0.0)

    def test_memory_padding_content_and_width_do_not_affect_output(self):
        hidden = torch.randn(2, 5, 24)
        memory = torch.randn(2, 5, 16)
        before = self.layer(hidden, src_mask=causal_mask(5), memory=memory)
        padded = torch.cat([memory, torch.full((2, 8, 16), float("nan"))], dim=1)
        padding = torch.arange(13)[None, :].expand(2, -1) >= 5
        after = self.layer(hidden, src_mask=causal_mask(5), memory=padded,
                           memory_key_padding_mask=padding)
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)

    def test_empty_memory_and_queries_are_safe(self):
        for batch, length, mem_length in ((2, 7, 7), (0, 7, 7), (2, 0, 7), (2, 7, 0)):
            with self.subTest(shape=(batch, length, mem_length)):
                hidden = torch.zeros(batch, length, 24)
                memory = torch.zeros(batch, mem_length, 16)
                padding = torch.ones(batch, length, dtype=torch.bool)
                result = self.layer(
                    hidden, src_mask=causal_mask(length),
                    src_key_padding_mask=padding, memory=memory,
                    memory_key_padding_mask=torch.ones(batch, mem_length, dtype=torch.bool),
                )
                self.assertTrue(bool(torch.isfinite(result).all()))
                self.assertEqual(int(torch.count_nonzero(result)), 0)

    def test_self_attention_names_and_float_padding_mask_remain_compatible(self):
        keys = set(self.layer.state_dict())
        for key in ("self_attn.in_proj_weight", "self_attn.out_proj.weight",
                    "linear1.weight", "linear2.weight", "norm1.weight", "norm2.weight"):
            self.assertIn(key, keys)
        hidden = torch.randn(1, 5, 24)
        memory = torch.randn(1, 5, 16)
        boolean = torch.tensor([[False, False, False, True, True]])
        additive = torch.zeros(1, 5).masked_fill(boolean, float("-inf"))
        attn_additive = torch.zeros(5, 5).masked_fill(causal_mask(5), float("-inf"))
        expected = self.layer(hidden, src_mask=causal_mask(5), memory=memory,
                              src_key_padding_mask=boolean,
                              memory_key_padding_mask=boolean)
        actual = self.layer(hidden, src_mask=attn_additive, memory=memory,
                            src_key_padding_mask=additive,
                            memory_key_padding_mask=additive)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_short_and_empty_read_padding_with_sliding_window_has_finite_gradients(self):
        hidden = torch.randn(2, 12, 24, requires_grad=True)
        memory = torch.randn(2, 12, 16, requires_grad=True)
        active = torch.arange(12)[None, :] < torch.tensor([2, 0])[:, None]
        query = torch.arange(12)[:, None]
        key = torch.arange(12)[None, :]
        window = (key > query) | (query - key >= 4)
        output = self.layer(hidden, src_mask=window,
                            src_key_padding_mask=~active, memory=memory,
                            memory_key_padding_mask=~active)
        output.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(hidden.grad).all()))
        self.assertTrue(bool(torch.isfinite(memory.grad).all()))
        self.assertEqual(int(torch.count_nonzero(output[~active])), 0)


if __name__ == "__main__":
    unittest.main()
