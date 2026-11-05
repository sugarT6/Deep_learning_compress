"""Whole-block prefix isolation and frozen-backbone head adaptation checks."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from codec.block_features import decode_block_features, encode_block_features
from codec.fastq_stream import encode_base_ids, iter_fastq_records, make_fastq_batch
from codec.head_adapter import (
    HeadAdaptationConfig,
    adapt_output_head,
    apply_head_adapter,
    serialize_head,
)
from codec.model import DirectQualityModelConfig, DirectQualityTransformer
from codec.read_state import clear_conditioning


BASE_SHA = "b" * 64


def block_model(mode="features", *, layers=1):
    torch.manual_seed(414)
    model = DirectQualityTransformer(DirectQualityModelConfig(
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, d_model=8,
        num_heads=2, num_layers=layers, feedforward_dim=16,
        context_length=8, dropout=0.0, block_quality_features=True,
        block_reads=4, block_feature_mode=mode, block_conditioning_dim=5,
    ))
    # Every test exercises a trained, non-identity conditioner.
    with torch.no_grad():
        model.block_conditioning.encoder[-1].weight.uniform_(-0.2, 0.2)
        model.block_conditioning.encoder[-1].bias.uniform_(-0.2, 0.2)
    return model


def adaptation_config(**changes):
    values = dict(
        max_reads=64, max_symbols=10000, max_read_length=8,
        steps=2, symbols_per_step=32, learning_rate=0.05,
        anchor_strength=0.0, max_seconds=0.0, quantization_version=2,
    )
    values.update(changes)
    return HeadAdaptationConfig(**values)


def write_source(root, count, *, lengths=None, constant_quality=None):
    raw = []
    for index in range(count):
        length = 4 if lengths is None else lengths[index]
        quality = (index * 3 + 7) % 42 if constant_quality is None else constant_quality
        ending = b"\r\n" if index % 2 else b"\n"
        raw.append(
            b"@read" + str(index).encode() + ending + b"A" * length + ending
            + b"+description" + ending + bytes([33 + quality]) * length + ending
        )
    path = root / "input.fastq"
    path.write_bytes(b"".join(raw))
    return path, raw


def batch_for(records):
    return make_fastq_batch(
        [encode_base_ids(record.sequence) for record in records],
        [np.frombuffer(record.quality, dtype=np.uint8) - 33 for record in records],
        [record.read_index for record in records], source_name="input.fastq",
    )


def adapt(model, source, config, **changes):
    arguments = dict(device=torch.device("cpu"), batch_reads=4, total=65536,
                     base_sha256=BASE_SHA, config=config)
    arguments.update(changes)
    return adapt_output_head(model, source, **arguments)


class BlockHeadAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_wire_conditions(self, calls, all_records, expected_reads):
        expected = {}
        for start in range(0, len(all_records), 4):
            rows = all_records[start:start + 4]
            feature = decode_block_features(encode_block_features(batch_for(rows)))
            for row in rows:
                expected[row.read_index] = feature
        # Distinct first-Q values identify these test reads without exposing
        # source indices to the model's forward interface.
        lookup = {record.quality[0] - 33: record.read_index for record in all_records}
        seen = []
        for call in calls:
            kwargs = call.kwargs
            condition = kwargs["block_features"]
            self.assertEqual(condition.dtype, torch.float32)
            self.assertFalse(condition.requires_grad)
            for quality, feature in zip(kwargs["qualities"], condition):
                index = lookup[int(quality[0])]
                seen.append(index)
                np.testing.assert_array_equal(feature.cpu().numpy(), expected[index])
        self.assertEqual(seen, list(range(expected_reads)))

    def test_read_cap_finishes_only_original_blocks_and_recomputes_report_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            source, raw = write_source(Path(directory), 14)
            records = list(iter_fastq_records(source))
            model = block_model()
            with mock.patch.object(model, "forward_features", wraps=model.forward_features) as forward:
                _, report = adapt(model, source, adaptation_config(max_reads=11))
            self.assertEqual(report["sample_stop"], "max_reads")
            self.assertEqual(report["prefix_reads"], 8)
            self.assertEqual(report["prefix_symbols"], 32)
            self.assertEqual(report["prefix_blocks"], 2)
            self.assertEqual(report["train_read_range"], [0, 4])
            self.assertEqual(report["validation_read_range"], [4, 8])
            self.assertEqual(report["prefix_sha256"], hashlib.sha256(b"".join(raw[:8])).hexdigest())
            self.assert_wire_conditions(forward.call_args_list, records, 8)

    def test_symbol_and_length_caps_do_not_create_partial_block_descriptions(self):
        for cap in ("symbols", "length"):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                lengths = [4] * 14
                changes = {"max_symbols": 38}
                if cap == "length":
                    lengths[9] = 5
                    changes = {"max_read_length": 4}
                source, raw = write_source(Path(directory), 14, lengths=lengths)
                records = list(iter_fastq_records(source))
                model = block_model()
                with mock.patch.object(model, "forward_features", wraps=model.forward_features) as forward:
                    _, report = adapt(model, source, adaptation_config(**changes))
                self.assertEqual(report["sample_stop"], "length_or_symbol_cap")
                self.assertEqual((report["prefix_reads"], report["prefix_symbols"]), (8, 32))
                self.assertEqual(report["train_read_range"], [0, 4])
                self.assertEqual(report["validation_read_range"], [4, 8])
                self.assertEqual(report["prefix_sha256"], hashlib.sha256(b"".join(raw[:8])).hexdigest())
                self.assert_wire_conditions(forward.call_args_list, records, 8)

    def test_genuine_eof_partial_block_is_kept_whole_in_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, raw = write_source(Path(directory), 10, lengths=[1, 2, 3, 4] * 2 + [2, 3])
            records = list(iter_fastq_records(source))
            model = block_model()
            with mock.patch.object(model, "forward_features", wraps=model.forward_features) as forward:
                _, report = adapt(model, source, adaptation_config())
            self.assertEqual(report["sample_stop"], "eof")
            self.assertEqual(report["prefix_reads"], 10)
            self.assertEqual(report["prefix_blocks"], 3)
            self.assertEqual(report["train_read_range"], [0, 8])
            self.assertEqual(report["validation_read_range"], [8, 10])
            self.assertEqual(report["prefix_sha256"], hashlib.sha256(b"".join(raw)).hexdigest())
            self.assertEqual(report["train_symbols"], 20)
            self.assertEqual(report["validation_symbols"], 5)
            self.assert_wire_conditions(forward.call_args_list, records, 10)

    def test_single_complete_prefix_block_is_rejected_without_model_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = write_source(Path(directory), 14)
            model = block_model().train()
            before = copy.deepcopy(model.state_dict())
            with mock.patch.object(model, "forward_features", wraps=model.forward_features) as forward:
                adapter, report = adapt(model, source, adaptation_config(max_reads=7))
            self.assertIsNone(adapter)
            self.assertEqual(report["reason"], "insufficient_prefix_blocks")
            self.assertEqual(report["prefix_reads"], 4)
            self.assertEqual(forward.call_count, 0)
            self.assertTrue(model.training)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_accepted_cross_head_changes_no_backbone_or_conditioner_parameters(self):
        for mode in ("features", "constant"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source, _ = write_source(Path(directory), 14, constant_quality=30)
                model = block_model(mode, layers=4).train()
                model.base_encoder.requires_grad_(False)
                with torch.no_grad():
                    model.output_head.weight.zero_()
                    model.output_head.bias.zero_()
                before = copy.deepcopy(model.state_dict())
                requires_grad = {name: value.requires_grad for name, value in model.named_parameters()}
                conditioner = model.block_conditioning
                with mock.patch.object(model, "forward_cross_layer_features",
                                       wraps=model.forward_cross_layer_features) as forward:
                    adapter, report = adapt(model, source, adaptation_config(
                        head_type="residual", residual_dim=4, cross_layer=True,
                        cross_dim=2, steps=12,
                    ))
                self.assertTrue(report["accepted"], report)
                self.assertIsNotNone(adapter)
                self.assertIs(model.block_conditioning, conditioner)
                self.assertTrue(model.training)
                self.assertFalse(torch.equal(before["output_head.bias"], model.output_head.bias))
                for name, value in model.named_parameters():
                    if not name.startswith("output_head."):
                        self.assertTrue(torch.equal(before[name], value), name)
                        self.assertEqual(value.requires_grad, requires_grad[name], name)
                        self.assertIsNone(value.grad, name)
                self.assertGreater(forward.call_count, 0)
                for call in forward.call_args_list:
                    feature = call.kwargs["block_features"]
                    if mode == "constant":
                        self.assertEqual(torch.count_nonzero(feature).item(), 0)
                    else:
                        self.assertGreater(torch.count_nonzero(feature).item(), 0)

    def test_clear_read_state_and_apply_head_preserve_block_conditioning(self):
        model = block_model()
        original = copy.deepcopy(model.block_conditioning.state_dict())
        conditioner = model.block_conditioning
        model.read_state_conditioning = torch.nn.Identity()
        clear_conditioning(model)
        self.assertFalse(hasattr(model, "read_state_conditioning"))
        self.assertIs(model.block_conditioning, conditioner)
        head = copy.deepcopy(model.output_head)
        with torch.no_grad():
            head.bias.add_(0.1)
        apply_head_adapter(model, serialize_head(head, BASE_SHA), BASE_SHA)
        self.assertIs(model.block_conditioning, conditioner)
        for name, value in original.items():
            self.assertTrue(torch.equal(value, model.block_conditioning.state_dict()[name]), name)

    def test_unsupported_backbone_lora_and_wrong_batch_reject_before_source_access(self):
        model = block_model(layers=4)
        missing = Path("missing-block-head-source.fastq")
        with self.assertRaisesRegex(ValueError, "head fitting only"):
            adapt(model, missing, adaptation_config(backbone_layers=1))
        with self.assertRaisesRegex(ValueError, "head fitting only"):
            adapt(model, missing, adaptation_config(
                head_type="residual", cross_layer=True, lora=True,
            ))
        with self.assertRaisesRegex(ValueError, "equal to block_reads"):
            adapt(model, missing, adaptation_config(), batch_reads=2)


if __name__ == "__main__":
    unittest.main()
