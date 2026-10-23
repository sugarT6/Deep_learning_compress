"""Check diagnostics against actual encoder inference, not a second prediction pass."""

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from codec.checkpoint import save_training_checkpoint
from codec.container import sha256_file
from codec.decode import decode_fastq
from codec.encode import _quantize_verified_batch, encode_fastq, main
from codec.fastq_stream import iter_fastq_batches
from codec.head_adapter import HeadAdaptationConfig, serialize_head
from codec.model import (
    DirectQualityModelConfig,
    DirectQualityTransformer,
    fastq_batch_to_tensors,
)
from codec.probability_quantization import TOTAL
from codec.prediction_report import PredictionReport
from codec.range_encoder import RangeEncoder


class PredictionReportIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _checkpoint(self, root, *, uniform=False, name="model.pt"):
        torch.manual_seed(904)
        model = DirectQualityTransformer(DirectQualityModelConfig(
            d_model=8, num_heads=2, num_layers=1, feedforward_dim=16,
            prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
            base_conv_channels=3, base_context_dim=4, context_length=16,
            dropout=0.0,
        ))
        if uniform:
            with torch.no_grad():
                model.output_head.weight.zero_()
                model.output_head.bias.zero_()
        path = root / name
        save_training_checkpoint(
            path, model=model, optimizer=None, epoch=1, global_step=1,
            data_split={}, sampler_config={}, sampler_statistics={},
            best_validation_bits_per_quality=1.0, validation_metrics={},
        )
        model.eval()
        return path, model

    def _source(self, root, *, count=180, fixed_quality=None):
        records = []
        for read in range(count):
            length = read % 14  # Empty reads and padding must not enter accuracy.
            bases = bytes(b"ACGTN"[(read + cycle) % 5] for cycle in range(length))
            qualities = bytes(
                33 + (fixed_quality if fixed_quality is not None else (read * 3 + cycle * 7) % 42)
                for cycle in range(length)
            )
            records.append(b"@r" + str(read).encode() + b"\n" + bases
                           + b"\n+\n" + qualities + b"\n")
        source = root / "input.fq"
        source.write_bytes(b"".join(records))
        return source

    def _expected(self, model, source, batch_reads):
        symbols, probabilities = [], []
        with torch.inference_mode():
            for batch in iter_fastq_batches(source, batch_reads=batch_reads):
                logits = model.forward_full(**fastq_batch_to_tensors(batch, torch.device("cpu")))
                cycles, rows = np.nonzero(batch.active_mask.T)
                active_logits = logits.cpu().numpy()[rows, cycles].astype(np.float64)
                active_logits -= active_logits.max(axis=1, keepdims=True)
                weights = np.exp(active_logits)
                probabilities.extend(weights / weights.sum(axis=1, keepdims=True))
                symbols.extend(batch.qualities[rows, cycles].tolist())
        if not symbols:
            return np.empty(0, dtype=np.int64), np.empty((0, 42), dtype=np.float64)
        return np.asarray(symbols), np.asarray(probabilities)

    def _assert_report(self, output, statistics, symbols, probabilities):
        report = output.with_suffix(".predictions.txt")
        self.assertEqual(statistics.prediction_report_path, str(report.resolve()))
        lines = report.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1 + min(1000, len(symbols)))
        header = re.fullmatch(r"prediction_accuracy=(\S+) correct=(\d+) total=(\d+)", lines[0])
        self.assertIsNotNone(header, lines[0])
        correct = int((probabilities.argmax(axis=1) == symbols).sum()) if len(symbols) else 0
        self.assertEqual((int(header[2]), int(header[3])), (correct, len(symbols)))
        self.assertEqual(statistics.prediction_correct_symbols, correct)
        if len(symbols):
            self.assertTrue(header[1].endswith("%"))
            self.assertAlmostEqual(float(header[1][:-1]), correct / len(symbols) * 100, places=4)
            self.assertAlmostEqual(statistics.prediction_top1_accuracy, correct / len(symbols))
        else:
            self.assertEqual(header[1], "N/A")
            self.assertIsNone(statistics.prediction_top1_accuracy)
        for index, line in enumerate(lines[1:]):
            pairs = line.split()
            expected_top = np.argsort(-probabilities[index], kind="stable")[:5]
            correct_prediction = expected_top[0] == symbols[index]
            self.assertEqual(pairs[-1] == "yes", correct_prediction)
            if correct_prediction:
                pairs.pop()
            self.assertEqual(len(pairs), 10)
            self.assertEqual([int(value) for value in pairs[::2]], expected_top.tolist())
            np.testing.assert_allclose(
                [float(value) for value in pairs[1::2]],
                probabilities[index, expected_top], rtol=1e-7, atol=1e-10,
            )
        return lines

    def _encode(self, source, output, checkpoint, **options):
        return encode_fastq(
            source, output, checkpoint, device=torch.device("cpu"),
            batch_reads=31, quantization_version=2, coder_backend="python",
            progress=False, **options,
        )

    def test_all_symbols_accuracy_and_first_thousand_follow_cycle_major_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, model = self._checkpoint(root)
            source = self._source(root)
            symbols, probabilities = self._expected(model, source, batch_reads=31)
            self.assertGreater(len(symbols), 1000)
            output = root / "compressed.fqdc"
            with mock.patch.object(
                DirectQualityTransformer, "forward_full", autospec=True,
                side_effect=DirectQualityTransformer.forward_full,
            ) as forward:
                statistics = self._encode(source, output, checkpoint)
            self.assertEqual(forward.call_count, statistics.batch_count)
            self.assertEqual(statistics.quality_symbols, len(symbols))
            self._assert_report(output, statistics, symbols, probabilities)

    def test_loaded_adapter_predictions_are_reported_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, model = self._checkpoint(root)
            source = self._source(root, count=25, fixed_quality=37)
            adapted_head = copy.deepcopy(model.output_head)
            with torch.no_grad():
                adapted_head.weight.zero_()
                adapted_head.bias.copy_(-torch.arange(42, dtype=torch.float32) / 10)
                adapted_head.bias[37] = 3.0
            adapter_path = root / "adapter.json"
            digest = sha256_file(checkpoint)
            adapter_path.write_text(json.dumps({
                "format": "direct-quality-head-adaptation-artifact", "version": 1,
                "base_checkpoint_sha256": digest,
                "adapter": serialize_head(adapted_head, digest), "report": {},
            }))
            original_symbols, original_probabilities = self._expected(model, source, 31)
            self.assertFalse(np.all(original_probabilities.argmax(axis=1) == original_symbols))
            model.output_head = adapted_head
            symbols, probabilities = self._expected(model, source, 31)
            output = root / "adapted.fqdc"
            statistics = self._encode(source, output, checkpoint, head_adapter_path=adapter_path)
            lines = self._assert_report(output, statistics, symbols, probabilities)
            self.assertEqual(statistics.prediction_top1_accuracy, 1.0)
            self.assertTrue(all(line.endswith(" yes") for line in lines[1:]))
            restored = root / "restored.fq"
            decode_fastq(output, restored, checkpoint, device=torch.device("cpu"),
                         batch_reads=31, progress=False)
            self.assertEqual(restored.read_bytes(), source.read_bytes())

    def test_observing_predictions_preserves_both_quantizers_and_range_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, model = self._checkpoint(root)
            source = self._source(root, count=25)
            batch = next(iter_fastq_batches(source, batch_reads=31))
            for version in (1, 2):
                with self.subTest(quantization_version=version):
                    plain = _quantize_verified_batch(
                        model, batch, torch.device("cpu"), total=TOTAL,
                        online_prior=None, verify_cdf=True, quantization_version=version,
                    )
                    report = PredictionReport()
                    observed = _quantize_verified_batch(
                        model, batch, torch.device("cpu"), total=TOTAL,
                        online_prior=None, verify_cdf=True, quantization_version=version,
                        prediction_report=report,
                    )
                    np.testing.assert_array_equal(plain[0], observed[0])
                    if version == 1:
                        np.testing.assert_array_equal(plain[1], observed[1])
                    else:
                        for plain_interval, observed_interval in zip(plain[1], observed[1]):
                            np.testing.assert_array_equal(plain_interval, observed_interval)
                    self.assertEqual(plain[2:], observed[2:])
                    streams = []
                    for symbols, coded, _, _ in (plain, observed):
                        encoder = RangeEncoder(backend="python")
                        if version == 1:
                            encoder.encode_prevalidated_batch(symbols, coded, total=TOTAL)
                        else:
                            encoder.encode_prevalidated_intervals(*coded)
                        streams.append(encoder.finish())
                    self.assertEqual(streams[0], streams[1])
                    self.assertEqual(report.total_symbols, int(batch.active_mask.sum()))

    def test_ties_choose_smaller_q_in_top_one_and_top_five(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, model = self._checkpoint(root, uniform=True)
            source = self._source(root, count=15)
            symbols, probabilities = self._expected(model, source, 31)
            output = root / "uniform.fqdc"
            statistics = self._encode(source, output, checkpoint)
            lines = self._assert_report(output, statistics, symbols, probabilities)
            for line in lines[1:]:
                self.assertEqual([int(value) for value in line.split()[:10:2]], [0, 1, 2, 3, 4])

    def test_no_quality_symbols_produce_header_only_without_nan(self):
        for contents in (b"", b"@empty\n\n+\n\n" * 3):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint, _ = self._checkpoint(root)
                source = root / "empty.fq"
                source.write_bytes(contents)
                output = root / "empty.fqdc"
                statistics = self._encode(source, output, checkpoint)
                self._assert_report(output, statistics, np.empty(0), np.empty((0, 42)))

    def test_report_path_collisions_fail_before_loading_checkpoint(self):
        cases = ("existing", "input", "output", "checkpoint", "loaded_adapter", "exported_adapter", "dangling_symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self._source(root, count=3)
                checkpoint, _ = self._checkpoint(root)
                output = root / "compressed.fqdc"
                report = output.with_suffix(".predictions.txt")
                options = {}
                if case == "existing":
                    report.write_text("preserve existing report\n")
                elif case == "input":
                    report.symlink_to(source)
                elif case == "output":
                    report.symlink_to(output)
                elif case == "checkpoint":
                    checkpoint.rename(report)
                    checkpoint = report
                elif case == "loaded_adapter":
                    options["head_adapter_path"] = report
                elif case == "dangling_symlink":
                    report.symlink_to(root / "unrelated_target.txt")
                else:
                    options.update(save_head_adapter_path=report,
                                   head_adaptation_config=HeadAdaptationConfig())
                initial_files = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
                with mock.patch("codec.encode.load_training_checkpoint") as load:
                    with self.assertRaises((ValueError, OSError)):
                        self._encode(source, output, checkpoint, **options)
                    load.assert_not_called()
                self.assertFalse(output.exists())
                if case == "dangling_symlink":
                    self.assertTrue(report.is_symlink())
                    self.assertFalse((root / "unrelated_target.txt").exists())
                for path, contents in initial_files.items():
                    self.assertEqual(path.read_bytes(), contents)

    def test_cli_default_requires_no_prediction_flag(self):
        with mock.patch("sys.argv", ["codec.encode", "input.fq", "output.fqdc", "base.pt", "--device", "cpu"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            self.assertEqual(main(), 0)
        self.assertEqual(encode.call_args.args[:3], (Path("input.fq"), Path("output.fqdc"), Path("base.pt")))


if __name__ == "__main__":
    unittest.main()
