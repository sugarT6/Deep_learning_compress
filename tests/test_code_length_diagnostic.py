"""Check fresh sampling, exact symbol code lengths and read-only saved-model scoring."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from codec.diagnose_code_lengths import (
    CodeLengthConfig, _load_inputs, run_code_length_diagnostic,
    sample_fresh_records, score_symbols, symbol_code_lengths, symbol_metadata,
)
from codec.fastq_stream import iter_fastq_records
from test_adaptation_ablation import prepared_diagnostic


def saved_fixture(root):
    from codec.checkpoint import load_training_checkpoint
    from codec.container import sha256_file
    from codec.model import CrossLayerOutputHead
    from codec.suffix_lora_adapter import install_suffix_lora, serialize_suffix_lora

    source, checkpoint, diagnostic, _ = prepared_diagnostic(root)
    manifest = json.loads((diagnostic / "manifest.json").read_text())
    base = load_training_checkpoint(checkpoint).model.eval()
    candidate = copy.deepcopy(base)
    candidate.output_head = CrossLayerOutputHead(8, 16, 4)
    saved = torch.load(diagnostic / "full.best.pt", map_location="cpu")
    head_state = {name[len("output_head."):]: value for name, value in saved["model_state_dict"].items()
                  if name.startswith("output_head.")}
    candidate.output_head.load_state_dict(head_state)
    layout = install_suffix_lora(candidate, 4, 2, 17)
    with torch.no_grad():
        for item in layout:
            module_name, name = item["name"].rsplit(".", 1)
            candidate.get_submodule(module_name).parametrizations[name][0].b.fill_(.01)
    digest = sha256_file(checkpoint)
    artifact = dict(format="direct-quality-head-adaptation-artifact", version=1,
                    source=str(source.resolve()), base_checkpoint_sha256=digest,
                    adapter=serialize_suffix_lora(candidate, 4, 2, digest, base_model=base),
                    report=dict(backbone_accepted=True, backbone_sampling=manifest["sample"],
                                warmup_report=manifest["warmup"],
                                backbone_train_reads=manifest["split"]["train_reads"],
                                backbone_validation_reads=manifest["split"]["validation_reads"],
                                backbone_lora_rank=2, backbone_selected_step=4,
                                config=dict(backbone_layers=4, quantization_version=2)))
    adapter_path = root / "suffix.adapter.json"
    adapter_path.write_text(json.dumps(artifact))
    return source, checkpoint, diagnostic, adapter_path, manifest


class CodeLengthDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_symbol_losses_use_true_row_totals_and_match_cdf_reference(self):
        from codec.encode_fastpath import selected_quantized_bits
        from codec.probability_quantization import logits_symbols_bits, logits_to_cdfs

        rng = np.random.default_rng(23)
        logits = rng.normal(size=(71, 42)) * 3
        logits[0] = 0
        logits[1, 0] = 10000
        labels = rng.integers(0, 42, len(logits))
        labels[1] = 41
        floating, quantized = symbol_code_lengths(logits, labels)
        cdfs = logits_to_cdfs(logits, version=2)
        freq = cdfs[np.arange(len(labels)), labels + 1] - cdfs[np.arange(len(labels)), labels]
        np.testing.assert_allclose(quantized, -np.log2(freq / cdfs[:, -1]), rtol=0, atol=0)
        self.assertAlmostEqual(float(quantized.sum()), selected_quantized_bits(labels, cdfs, cdfs[:, -1]), places=10)
        self.assertAlmostEqual(float(floating.sum()), logits_symbols_bits(logits, labels), places=9)
        self.assertTrue(np.any(cdfs[:, -1] != 65536))
        self.assertTrue(np.isfinite(floating).all())
        for bad_logits, bad_labels in ((np.full((1, 42), np.nan), [0]),
                                      (np.zeros((1, 42)), [42]),
                                      (np.zeros((1, 42)), [0.0])):
            with self.assertRaises(ValueError):
                symbol_code_lengths(bad_logits, bad_labels)

    def test_transition_groups_do_not_overflow_and_reads_keep_boundaries(self):
        from codec.fastq_stream import RawFastqRecord

        def record(index, qualities):
            return RawFastqRecord(b"@ignored", b"A" * len(qualities), b"+",
                                  bytes(q + 33 for q in qualities), (b"\n",) * 4, index)

        rows = [record(3, [30, 30, 29, 20, 22, 30]), record(8, [5])]
        metadata = symbol_metadata(rows, total_reads=10, blocks=2)
        np.testing.assert_array_equal(metadata["transition"], [0, 1, 2, 3, 4, 5, 0])
        np.testing.assert_array_equal(metadata["quality_delta"], [0, 0, -1, -9, 2, 8, 0])
        np.testing.assert_array_equal(metadata["cycle"], [1, 2, 3, 4, 5, 6, 1])
        np.testing.assert_array_equal(metadata["file_block"], [1] * 6 + [2])

    def test_fresh_sample_excludes_both_old_pools_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, diagnostic, _, manifest = saved_fixture(Path(directory))
            with np.load(diagnostic / "split_indices.npz") as arrays:
                excluded = np.concatenate((arrays["train"], arrays["validation"]))
            config = CodeLengthConfig(reads=16, blocks=4, batch_reads=4, bootstrap_samples=20)
            kwargs = dict(total_reads=96, prefix_reads=8, excluded_indices=excluded,
                          max_read_length=2048, config=config, progress=False)
            rows, report = sample_fresh_records(source, **kwargs)
            again, second = sample_fresh_records(source, **kwargs)
            indices = [r.read_index for r in rows]
            self.assertEqual(indices, [r.read_index for r in again])
            self.assertEqual(report, second)
            self.assertEqual(report["block_sample_reads"], [4] * 4)
            self.assertEqual(report["excluded_reservoir"], len(excluded))
            self.assertFalse(set(indices) & set(excluded))
            self.assertTrue(all(i >= 8 for i in indices))
            self.assertEqual(report["total_symbols"], manifest["sample"]["total_file_symbols"])
            with self.assertRaisesRegex(ValueError, "too few"):
                sample_fresh_records(source, **{**kwargs, "config": replace(config, reads=96)})
            with self.assertRaisesRegex(ValueError, "max-symbols"):
                sample_fresh_records(source, **{**kwargs, "config": replace(config, max_symbols=1)})

    def test_end_to_end_outputs_conserve_bits_and_do_not_train_or_change_inputs(self):
        import codec.diagnose_code_lengths as diagnostic_module
        from codec.adaptation_diagnostic import score_records

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint, diagnostic, adapter, _ = saved_fixture(root)
            protected = [source, checkpoint, adapter] + [p for p in diagnostic.iterdir() if p.is_file()]
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
            config = CodeLengthConfig(reads=16, blocks=4, batch_reads=4, bootstrap_samples=20)
            output = root / "new_report"
            with mock.patch.object(diagnostic_module, "iter_fastq_records", wraps=iter_fastq_records) as scan, \
                    mock.patch("torch.optim.Adam", side_effect=AssertionError("no training")), \
                    mock.patch("torch.optim.AdamW", side_effect=AssertionError("no training")):
                report = run_code_length_diagnostic(diagnostic, adapter, output_dir=output,
                                                    config=config, progress=False)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(set(report["models"]), {"base", "lora", "full"})
            arrays = np.load(output / "per_symbol.npz")
            self.addCleanup(arrays.close)
            chosen = set(int(i) for i in arrays["sample_read_indices"])
            records = [r for r in iter_fastq_records(source) if r.read_index in chosen]
            _, _, _, models, _ = _load_inputs(diagnostic, adapter)
            for name, model in models.items():
                actual = report["models"][name]
                expected = score_records(model, records, device="cpu", batch_reads=4,
                                         source=source, quantized=True)
                self.assertAlmostEqual(actual["float_bpq"], expected["float_bpq"], places=6)
                self.assertAlmostEqual(actual["quantized_bpq"], expected["quantized_bpq"], places=12)
                self.assertAlmostEqual(float(arrays[f"{name}_quantized_bits"].sum()), actual["quantized_bits"], places=10)
            import csv
            with (output / "grouped_code_lengths.csv").open() as f:
                grouped = list(csv.DictReader(f))
            for dimension in ("cycle", "quality", "transition", "file_block"):
                for name in models:
                    selected = [r for r in grouped if r["dimension"] == dimension and r["model"] == name]
                    self.assertEqual(sum(int(r["symbols"]) for r in selected), len(arrays["quality"]))
                    self.assertAlmostEqual(sum(float(r["quantized_bits"]) for r in selected),
                                           report["models"][name]["quantized_bits"], places=9)
                    expected_gain = report["models"]["lora"]["quantized_bits"] - report["models"][name]["quantized_bits"]
                    self.assertAlmostEqual(sum(float(r["gain_bits_vs_lora"]) for r in selected), expected_gain, places=9)
            self.assertEqual(report["comparisons"]["lora"]["gain_bpq_95pct_read_bootstrap"], [0.0, 0.0])
            self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
            self.assertFalse(any(p.suffix in (".pt", ".fqdc") for p in output.iterdir()))
            with self.assertRaises(FileExistsError):
                run_code_length_diagnostic(diagnostic, adapter, output_dir=output, config=config, progress=False)
            # Headers and plus lines cannot affect the numerical diagnostic.
            renamed = [replace(r, header=b"@different:instrument:lane:x:y", plus=b"+different") for r in records]
            for model in models.values():
                before = score_symbols(model, records, source=source, device="cpu", batch_reads=4, total=65536, progress=False)
                after = score_symbols(model, renamed, source=source, device="cpu", batch_reads=4, total=65536, progress=False)
                for a, b in zip(before, after):
                    np.testing.assert_array_equal(a, b)

    def test_mismatched_exclusion_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, diagnostic, adapter, _ = saved_fixture(root)
            original = json.loads(adapter.read_text())
            for damage in ("source", "base_sha", "reservoir", "prefix", "rank", "version"):
                with self.subTest(damage=damage):
                    changed = copy.deepcopy(original)
                    if damage == "source":
                        changed["source"] = str(root / "other.fastq")
                    elif damage == "base_sha":
                        changed["base_checkpoint_sha256"] = "0" * 64
                    elif damage == "reservoir":
                        changed["report"]["backbone_sampling"]["read_indices_sha256"] = "0" * 64
                    elif damage == "prefix":
                        changed["report"]["warmup_report"]["prefix_reads"] += 1
                    elif damage == "rank":
                        changed["report"]["backbone_lora_rank"] += 1
                    else:
                        changed["adapter"]["version"] = 9
                    adapter.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        _load_inputs(diagnostic, adapter)
            adapter.write_text(json.dumps(original))
            _load_inputs(diagnostic, adapter)

    def test_invalid_sampling_config_is_rejected(self):
        for values in (dict(reads=0), dict(blocks=0), dict(reads=2, blocks=3),
                       dict(seed=-1), dict(batch_reads=257), dict(bootstrap_samples=0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CodeLengthConfig(**values)


if __name__ == "__main__":
    unittest.main()
