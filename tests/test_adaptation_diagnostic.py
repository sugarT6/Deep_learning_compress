"""CPU checks for the paired per-file adaptation capacity diagnostic."""

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from codec.checkpoint import save_training_checkpoint
from codec.fastq_stream import iter_fastq_records
from codec.model import CrossLayerOutputHead, DirectQualityModelConfig, DirectQualityTransformer


def write_source(root, *, name="reads.fastq", header_prefix="read", count=96):
    path = root / name
    rows = []
    for index in range(count):
        length = 5 + index % 4
        sequence = (b"ACGTACGT"[:length] if index % 2 else b"TGCATGCA"[:length])
        quality = bytes([33 + 10 + index % 3]) * length
        rows.append(b"@" + header_prefix.encode() + str(index).encode() + b"\n"
                    + sequence + b"\n+\n" + quality + b"\n")
    path.write_bytes(b"".join(rows))
    return path


def tiny_model(root, *, warm_head=False):
    torch.manual_seed(217)
    model = DirectQualityTransformer(DirectQualityModelConfig(
        d_model=8, num_heads=2, num_layers=4, feedforward_dim=16,
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, dropout=0.1,
    ))
    checkpoint = root / "base.pt"
    save_training_checkpoint(
        checkpoint, model=model, optimizer=None, epoch=1, global_step=1,
        data_split={}, sampler_config={}, sampler_statistics={},
        best_validation_bits_per_quality=3.0, validation_metrics={},
    )
    if warm_head:
        head = CrossLayerOutputHead(8, 16, 4)
        with torch.no_grad():
            head.weight.copy_(model.output_head.weight)
            head.bias.copy_(model.output_head.bias)
        model.output_head = head
    return model.eval(), checkpoint


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AdaptationDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fit_fixture(self, root):
        from codec.adaptation_diagnostic import score_records

        source = write_source(root)
        model, checkpoint = tiny_model(root, warm_head=True)
        records = list(iter_fastq_records(source))
        train, validation = records[8:32], records[32:40]
        probe = train[:8]
        common = dict(device=torch.device("cpu"), batch_reads=4, source=source,
                      quantized=True)
        initial = {
            "train": score_records(model, train, **common),
            "validation": score_records(model, validation, **common),
            "train_probe": score_records(model, probe, **common),
        }
        return model, checkpoint, source, train, validation, probe, initial

    def fit(self, root, fixture, mode, **overrides):
        from codec.adaptation_diagnostic import fit_branch

        model, checkpoint, source, train, validation, probe, initial = fixture
        output = root / mode
        options = dict(
            steps=4, eval_every=2, batch_reads=4, eval_batch_reads=4,
            learning_rate=.003, head_learning_rate=.03,
            anchor_strength=.01, seed=23, lora_layers=2, lora_rank=2,
            patience=0, progress=False,
        )
        options.update(overrides)
        result = fit_branch(
            model, train, validation, probe,
            mode=mode, source=source, device=torch.device("cpu"),
            output_dir=output, base_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            initial_metrics=initial, **options,
        )
        saved = torch.load(output / f"{mode}.best.pt", map_location="cpu")
        history = [json.loads(line) for line in
                   (output / f"{mode}.history.jsonl").read_text().splitlines()]
        return result, saved, history

    def test_scores_count_only_real_symbols_and_report_quantized_cost(self):
        from codec.adaptation_diagnostic import score_records

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root, count=11)
            records = list(iter_fastq_records(source))
            model, _ = tiny_model(root)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            expected_symbols = sum(len(record.quality) for record in records)
            result = score_records(model, records, device=torch.device("cpu"),
                                   batch_reads=3, source=source, quantized=True)
            self.assertEqual(result["reads"], len(records))
            self.assertEqual(result["quality_symbols"], expected_symbols)
            self.assertAlmostEqual(result["float_bpq"], math.log2(42), places=5)
            self.assertAlmostEqual(result["float_bits"], expected_symbols * math.log2(42), places=3)
            self.assertAlmostEqual(result["quantized_bits"] / expected_symbols,
                                   result["quantized_bpq"], places=12)
            self.assertLess(abs(result["float_bpq"] - result["quantized_bpq"]), .001)
            larger_batches = score_records(model, records, device=torch.device("cpu"),
                                           batch_reads=8, source=source, quantized=True)
            self.assertAlmostEqual(result["float_bpq"], larger_batches["float_bpq"], places=5)
            self.assertAlmostEqual(result["quantized_bits"], larger_batches["quantized_bits"], places=10)

    def test_full_updates_early_features_while_lora_freezes_them_and_draws_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fit_fixture(root)
            warm = fixture[0]
            before = {name: value.detach().clone() for name, value in warm.state_dict().items()}
            warm_digest = state_digest(warm)
            lora_result, lora, lora_history = self.fit(root, fixture, "lora")
            full_result, full, full_history = self.fit(root, fixture, "full")
            self.assertEqual(state_digest(warm), warm_digest, "paired initial model must remain untouched")
            self.assertEqual([row["step"] for row in lora_history], [0, 2, 4])
            self.assertEqual(lora_history[0]["validation"], full_history[0]["validation"])
            self.assertEqual(lora_history[0]["train_probe"], full_history[0]["train_probe"])
            for lora_row, full_row in zip(lora_history[1:], full_history[1:]):
                self.assertEqual(lora_row["batch_read_indices_sha256"],
                                 full_row["batch_read_indices_sha256"])
            lora_state, full_state = lora["model_state_dict"], full["model_state_dict"]
            frozen_prefixes = ("prev_q_embedding.", "qmer_embeddings.", "base_encoder.",
                               "input_projection.", "transformer.layers.0.", "transformer.layers.1.")
            for name, value in before.items():
                if name.startswith(frozen_prefixes):
                    torch.testing.assert_close(lora_state[name], value, rtol=0, atol=0)
            for prefix in ("prev_q_embedding.", "base_encoder.", "input_projection.",
                           "transformer.layers.0."):
                self.assertTrue(any(not torch.equal(full_state[name], value)
                                    for name, value in before.items() if name.startswith(prefix)), prefix)
            self.assertTrue(any(not torch.equal(lora_state[name], value)
                                for name, value in before.items()
                                if name.startswith("transformer.layers.3.")))
            self.assertFalse(any("parametrizations" in name for name in lora_state))
            from codec.adaptation_diagnostic import score_records
            for result, payload, history in ((lora_result, lora, lora_history),
                                             (full_result, full, full_history)):
                best = min(history[1:], key=lambda row: row["validation"]["float_bpq"])
                self.assertEqual(result["selected_step"], best["step"])
                self.assertEqual(payload["selected_step"], best["step"])
                restored = copy.deepcopy(warm)
                restored.load_state_dict(payload["model_state_dict"], strict=True)
                actual = score_records(restored, fixture[4], device=torch.device("cpu"),
                                       batch_reads=4, source=fixture[2], quantized=True)
                self.assertEqual(actual, result["selected_metrics"]["validation"])

    def test_best_checkpoint_restores_earlier_holdout_choice_not_last_update(self):
        import codec.adaptation_diagnostic as diagnostic

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fit_fixture(root)
            validation = fixture[4]
            original_score = diagnostic.score_records
            forced = iter((3.0, 4.0))
            snapshots = []

            def score(model, records, **kwargs):
                result = original_score(model, records, **kwargs)
                if records is validation and not kwargs.get("quantized", False):
                    value = next(forced)
                    snapshots.append({name: tensor.detach().cpu().clone()
                                      for name, tensor in model.state_dict().items()})
                    result["float_bpq"] = value
                    result["float_bits"] = value * result["quality_symbols"]
                return result

            with mock.patch.object(diagnostic, "score_records", side_effect=score):
                _, selected, history = self.fit(root, fixture, "full")
            self.assertEqual(selected["selected_step"], 2)
            self.assertEqual([row["validation"]["float_bpq"] for row in history[1:]], [3.0, 4.0])
            self.assertEqual(len(snapshots), 2)
            for name, value in selected["model_state_dict"].items():
                torch.testing.assert_close(value, snapshots[0][name], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(snapshots[0][name], snapshots[1][name])
                                for name in snapshots[0]))

    def test_header_changes_do_not_affect_scores_and_quality_history_is_causal(self):
        from codec.adaptation_diagnostic import score_records
        from codec.backbone_adapter import _tensors

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root, count=6)
            records = list(iter_fastq_records(source))
            renamed = [replace(record, header=b"@new:instrument:metadata", plus=b"+also_changed")
                       for record in records]
            model, _ = tiny_model(root, warm_head=True)
            kwargs = dict(device=torch.device("cpu"), batch_reads=3, source=source, quantized=True)
            first = score_records(model, records, **kwargs)
            second = score_records(model, renamed, **kwargs)
            self.assertEqual(first, second)
            self.assertFalse(hasattr(model, "read_state_conditioning"))
            tensors = _tensors(records, torch.device("cpu"), source)
            changed = {**tensors, "qualities": tensors["qualities"].clone()}
            future_mask = tensors["active_mask"].clone()
            future_mask[:, :3] = False
            changed["qualities"][future_mask] = 41
            with torch.no_grad():
                original = model.forward_full(**tensors)
                perturbed = model.forward_full(**changed)
            torch.testing.assert_close(original[:, :4], perturbed[:, :4], rtol=0, atol=0)

    def test_exported_lora_artifact_matches_saved_model_and_codec_roundtrip(self):
        from codec.checkpoint import load_training_checkpoint
        from codec.container import sha256_file
        from codec.decode import decode_fastq
        from codec.encode import encode_fastq
        from codec.head_adapter import apply_head_adapter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fit_fixture(root)
            _, checkpoint, source, _, _, _, _ = fixture
            _, selected, _ = self.fit(root, fixture, "lora")
            artifact_path = root / "lora" / "lora.adapter.json"
            artifact = json.loads(artifact_path.read_text())
            restored = load_training_checkpoint(checkpoint, device="cpu").model.eval()
            apply_head_adapter(restored, artifact["adapter"], sha256_file(checkpoint))
            self.assertEqual(restored.state_dict().keys(), selected["model_state_dict"].keys())
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, selected["model_state_dict"][name], rtol=0, atol=0)
            encoded, decoded = root / "tiny.fqdc", root / "restored.fastq"
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
            stats = encode_fastq(source, encoded, checkpoint, head_adapter_path=artifact_path,
                                 quantization_version=2, coder_backend="python", **kwargs)
            decode_fastq(encoded, decoded, checkpoint, **kwargs)
            self.assertEqual(decoded.read_bytes(), source.read_bytes())
            self.assertEqual(stats.read_count, 96)
            self.assertEqual(stats.read_state_stream_bytes, 0)

    def diagnostic_config(self, **overrides):
        from codec.diagnose_adaptation import DiagnosticConfig

        values = dict(
            steps=4, eval_every=2, batch_reads=4, eval_batch_reads=4,
            lora_lr=.003, full_lr=.003, head_lr=.03, seed=23,
            sample_reads=32, sample_symbols=256, train_probe_reads=8,
            warmup_reads=8, warmup_symbols=64, warmup_steps=2,
            warmup_symbols_per_step=16, residual_dim=16, cross_dim=4,
            lora_layers=2, lora_rank=2, patience=0,
        )
        values.update(overrides)
        return DiagnosticConfig(**values)

    def test_preparation_uses_one_disjoint_split_and_shared_warm_start(self):
        from codec.diagnose_adaptation import run_diagnostic

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root)
            _, checkpoint = tiny_model(root)
            original_source, original_checkpoint = source.read_bytes(), checkpoint.read_bytes()
            output = root / "paired"
            result = run_diagnostic(source, checkpoint, output_dir=output,
                                    device=torch.device("cpu"),
                                    config=self.diagnostic_config(), progress=False)
            self.assertEqual(set(result["modes"]), {"lora", "full"})
            manifest = json.loads((output / "manifest.json").read_text())
            with np.load(output / "split_indices.npz") as split:
                train, validation, probe = split["train"], split["validation"], split["train_probe"]
                self.assertEqual((len(train), len(validation), len(probe)), (24, 8, 8))
                self.assertFalse(set(train) & set(validation))
                self.assertTrue(set(probe) <= set(train))
                self.assertTrue(all(index >= manifest["warmup"]["prefix_reads"]
                                    for index in list(train) + list(validation)))
                for key, values in (("train", train), ("validation", validation), ("train_probe", probe)):
                    digest = hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()
                    self.assertEqual(digest, manifest["split"][key + "_indices_sha256"])
            histories = {
                mode: [json.loads(line) for line in
                       (output / f"{mode}.history.jsonl").read_text().splitlines()]
                for mode in ("lora", "full")
            }
            for mode in histories:
                self.assertEqual(histories[mode][0]["validation"], result["common_initial"]["validation"])
                self.assertEqual(histories[mode][0]["train_probe"], result["common_initial"]["train_probe"])
            self.assertEqual(
                [row["batch_read_indices_sha256"] for row in histories["lora"]],
                [row["batch_read_indices_sha256"] for row in histories["full"]],
            )
            full_only = root / "full_only"
            standalone = run_diagnostic(source, checkpoint, output_dir=full_only, mode="full",
                                        device=torch.device("cpu"),
                                        config=self.diagnostic_config(), progress=False)
            standalone_manifest = json.loads((full_only / "manifest.json").read_text())
            self.assertEqual(manifest["warm_model_sha256"], standalone_manifest["warm_model_sha256"])
            self.assertEqual(manifest["split"], standalone_manifest["split"])
            self.assertEqual(result["common_initial"], standalone["common_initial"])
            standalone_history = [json.loads(line) for line in
                                  (full_only / "full.history.jsonl").read_text().splitlines()]
            for paired_row, standalone_row in zip(histories["full"], standalone_history):
                self.assertEqual(paired_row["batch_read_indices_sha256"],
                                 standalone_row["batch_read_indices_sha256"])
                self.assertEqual(paired_row["validation"], standalone_row["validation"])
            self.assertEqual(source.read_bytes(), original_source)
            self.assertEqual(checkpoint.read_bytes(), original_checkpoint)
            self.assertFalse(any(path.suffix == ".fqdc" for path in output.rglob("*")))
            with self.assertRaisesRegex(FileExistsError, "empty"):
                run_diagnostic(source, checkpoint, output_dir=output,
                               config=self.diagnostic_config(), progress=False)

    def test_cached_and_scanned_diagnostics_match_without_second_reservoir_scan(self):
        from codec.adaptation_sample import prepare_sample, load_sample, content_digest
        from codec.diagnose_adaptation import run_diagnostic
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root)
            _, checkpoint = tiny_model(root)
            config = self.diagnostic_config(lora_layers=4)
            cache = root / "cache"
            prepared = prepare_sample(source, cache, config)
            records, manifest = load_sample(source, cache, config)
            self.assertEqual(content_digest(records), prepared["sample_content_sha256"])
            self.assertEqual(manifest, prepared)
            scanned = run_diagnostic(source, checkpoint, output_dir=root / "scan", mode="lora",
                                     config=config, progress=False)
            with mock.patch("codec.diagnose_adaptation.sample_records", side_effect=AssertionError("rescan")):
                cached = run_diagnostic(source, checkpoint, output_dir=root / "cached", mode="lora",
                                        config=config, progress=False, sample_cache=cache)
            self.assertEqual(scanned["common_initial"], cached["common_initial"])
            a, b = scanned["modes"]["lora"], cached["modes"]["lora"]
            for key in ("selected_metrics", "selected_step", "batch_read_indices_sha256",
                        "canonical_container_adapter_bytes"):
                self.assertEqual(a[key], b[key])
            self.assertGreater(b["canonical_container_adapter_bytes"], 0)
            scan_manifest = json.loads((root / "scan/manifest.json").read_text())
            cache_manifest = json.loads((root / "cached/manifest.json").read_text())
            self.assertEqual(scan_manifest["split"], cache_manifest["split"])
            self.assertEqual(scan_manifest["warm_model_sha256"], cache_manifest["warm_model_sha256"])
            self.assertIsNotNone(cache_manifest["base_validation_metrics"]["quantized_bpq"])
            self.assertFalse(any(root.rglob("*.fqdc")))

    def test_sample_cache_rejects_wrong_config_source_and_corruption_before_model_load(self):
        from codec.adaptation_sample import prepare_sample, load_sample
        from codec.diagnose_adaptation import run_diagnostic
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root)
            cache, config = root / "cache", self.diagnostic_config()
            prepare_sample(source, cache, config)
            with mock.patch("codec.diagnose_adaptation.load_training_checkpoint", side_effect=AssertionError("load")):
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    run_diagnostic(source, root / "missing.pt", output_dir=root / "bad",
                                   config=replace(config, seed=999), sample_cache=cache, progress=False)
            with mock.patch("codec.adaptation_sample.sample_records", side_effect=AssertionError("rescan")):
                with self.assertRaises(FileExistsError):
                    prepare_sample(source, cache, config)
            arrays = cache / "records.npz"
            arrays.write_bytes(arrays.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_sample(source, cache, config)
            source.write_bytes(source.read_bytes() + b"@extra\nA\n+\n!\n")
            with self.assertRaisesRegex(ValueError, "identity"):
                load_sample(source, cache, config)

    def test_sample_cache_rejects_overlapping_or_inconsistent_imports(self):
        from codec.adaptation_sample import prepare_sample, load_sample, save_sample
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = write_source(root), self.diagnostic_config()
            prepare_sample(source, root / "cache", config)
            records, manifest = load_sample(source, root / "cache", config)
            with self.assertRaisesRegex(ValueError, "indices"):
                save_sample(source, root / "overlap", config,
                            [replace(records[0], read_index=0)] + records[1:], manifest["sample"])
            with self.assertRaisesRegex(ValueError, "indices"):
                save_sample(source, root / "duplicate", config,
                            [records[1]] + records[1:], manifest["sample"])
            with self.assertRaisesRegex(ValueError, "counts"):
                save_sample(source, root / "counts", config, records,
                            dict(manifest["sample"], sampled_symbols=1))

    def test_invalid_budgets_and_modes_are_rejected(self):
        from codec.diagnose_adaptation import run_diagnostic

        for invalid in (dict(steps=0), dict(eval_every=0), dict(sample_reads=3),
                        dict(warmup_reads=3), dict(batch_reads=257), dict(full_lr=float("nan"))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.diagnostic_config(**invalid)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root)
            _, checkpoint = tiny_model(root)
            with self.assertRaisesRegex(ValueError, "mode"):
                run_diagnostic(source, checkpoint, output_dir=root / "bad", mode="unrecognized",
                               config=self.diagnostic_config(), progress=False)

    def test_branch_rejects_overlapping_or_inconsistent_read_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fit_fixture(root)
            train, validation, probe = fixture[3:6]
            variants = (
                ("overlap", train, validation[:-1] + [train[0]], probe),
                ("duplicate", train[:-1] + [train[0]], validation, probe),
                ("foreign_probe", train, validation, [validation[0]]),
                ("changed_probe", train, validation,
                 [replace(train[0], quality=b"J" * len(train[0].quality))]),
                ("empty_training", [replace(train[0], sequence=b"", quality=b"")] + train[1:],
                 validation, probe),
            )
            for name, bad_train, bad_validation, bad_probe in variants:
                changed = (*fixture[:3], bad_train, bad_validation, bad_probe, fixture[6])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.fit(root / name, changed, "full")
