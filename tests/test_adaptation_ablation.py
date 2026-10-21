"""Read-only ablations must preserve the fitted state and its exact read split."""

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

from codec.diagnose_adaptation import DiagnosticConfig, run_diagnostic
from test_adaptation_diagnostic import tiny_model, write_source


def prepared_diagnostic(root):
    source = write_source(root)
    _, checkpoint = tiny_model(root)
    config = DiagnosticConfig(
        steps=4, eval_every=2, batch_reads=4, eval_batch_reads=4,
        lora_lr=.003, full_lr=.003, head_lr=.03, seed=23,
        sample_reads=32, sample_symbols=256, train_probe_reads=8,
        warmup_reads=8, warmup_symbols=64, warmup_steps=2,
        warmup_symbols_per_step=16, residual_dim=16, cross_dim=4,
        lora_layers=2, lora_rank=2, patience=0,
    )
    output = root / "diagnostic"
    result = run_diagnostic(source, checkpoint, output_dir=output, mode="full",
                            device=torch.device("cpu"), config=config, progress=False)
    return source, checkpoint, output, result


class AdaptationAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fitted_models(self, checkpoint, diagnostic):
        from codec.checkpoint import load_training_checkpoint
        from codec.model import CrossLayerOutputHead

        base = load_training_checkpoint(checkpoint, device="cpu").model.eval()
        full = copy.deepcopy(base)
        saved = torch.load(diagnostic / "full.best.pt", map_location="cpu")
        self.assertEqual(saved["head_spec"]["type"], "cross_layer")
        full.output_head = CrossLayerOutputHead(
            full.config.d_model, saved["head_spec"]["residual_dim"],
            saved["head_spec"]["cross_dim"],
        )
        full.load_state_dict(saved["model_state_dict"], strict=True)
        return base, full.eval()

    def test_candidates_restore_only_requested_groups_and_are_independent(self):
        from codec.ablate_adaptation import build_candidate, parameter_groups

        with tempfile.TemporaryDirectory() as directory:
            _, checkpoint, diagnostic, _ = prepared_diagnostic(Path(directory))
            base, full = self.fitted_models(checkpoint, diagnostic)
            base_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
            full_state = {name: tensor.detach().clone() for name, tensor in full.state_dict().items()}
            groups = parameter_groups(full, early_layers=2)
            early, inputs = set(groups["early_layers"]), set(groups["input_features"])
            self.assertTrue(early)
            self.assertTrue(inputs)
            self.assertFalse(early & inputs)
            self.assertEqual(early, {name for name in full_state
                                     if name.startswith(("transformer.layers.0.", "transformer.layers.1."))})
            self.assertEqual(inputs, {name for name in full_state
                                      if name.startswith(("prev_q_embedding.", "qmer_embeddings.",
                                                          "base_encoder.", "input_projection."))})
            self.assertTrue(any(not torch.equal(full_state[name], base_state[name]) for name in early))
            self.assertTrue(any(not torch.equal(full_state[name], base_state[name]) for name in inputs))
            candidates = {}
            for name, revert_early, revert_inputs in (
                ("full", False, False), ("revert_early_layers", True, False),
                ("revert_input_features", False, True), ("revert_both", True, True),
            ):
                candidate = build_candidate(full, base, revert_early_layers=revert_early,
                                            revert_input_features=revert_inputs, early_layers=2)
                candidates[name] = candidate
                reverted = (early if revert_early else set()) | (inputs if revert_inputs else set())
                self.assertFalse(candidate.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in candidate.parameters()))
                self.assertEqual(candidate.state_dict().keys(), full_state.keys())
                for tensor_name, value in candidate.state_dict().items():
                    expected = base_state[tensor_name] if tensor_name in reverted else full_state[tensor_name]
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            # Changing one branch cannot leak through aliased tensors to the
            # fitted source model or another candidate.
            head_name = "output_head.weight"
            with torch.no_grad():
                candidates["revert_early_layers"].output_head.weight.add_(1)
            torch.testing.assert_close(full.state_dict()[head_name], full_state[head_name], rtol=0, atol=0)
            torch.testing.assert_close(candidates["revert_input_features"].state_dict()[head_name],
                                       full_state[head_name], rtol=0, atol=0)
            for tensor_name, value in base.state_dict().items():
                torch.testing.assert_close(value, base_state[tensor_name], rtol=0, atol=0)

    def test_full_scores_reproduce_and_exact_saved_pools_are_scanned_once(self):
        import codec.ablate_adaptation as ablation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint, diagnostic, original = prepared_diagnostic(root)
            protected = [source, checkpoint] + [p for p in diagnostic.iterdir() if p.is_file()]
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
            with np.load(diagnostic / "split_indices.npz") as arrays:
                expected_probe = tuple(int(index) for index in arrays["train_probe"])
                expected_validation = tuple(int(index) for index in arrays["validation"])
            original_score = ablation.score_records
            seen = []

            def score(model, records, **kwargs):
                seen.append(tuple(record.read_index for record in records))
                return original_score(model, records, **kwargs)

            output = root / "ablation"
            with mock.patch.object(ablation, "score_records", side_effect=score), \
                    mock.patch.object(ablation, "iter_fastq_records", wraps=ablation.iter_fastq_records) as scan:
                report = ablation.run_ablation(diagnostic, output_dir=output,
                                               device=torch.device("cpu"), eval_batch_reads=4,
                                               progress=False)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(set(report["cases"]),
                             {"full", "revert_early_layers", "revert_input_features", "revert_both"})
            self.assertEqual(seen.count(expected_probe), 4)
            self.assertEqual(seen.count(expected_validation), 4)
            selected = original["modes"]["full"]["selected_metrics"]
            for pool in ("train_probe", "validation"):
                actual = report["cases"]["full"][pool]
                self.assertEqual(actual["reads"], selected[pool]["reads"])
                self.assertEqual(actual["quality_symbols"], selected[pool]["quality_symbols"])
                self.assertAlmostEqual(actual["float_bpq"], selected[pool]["float_bpq"], places=6)
                self.assertAlmostEqual(actual["quantized_bpq"], selected[pool]["quantized_bpq"], places=8)
            for name, case in report["cases"].items():
                self.assertAlmostEqual(case["validation_delta_from_full"],
                                       case["validation"]["float_bpq"]
                                       - report["cases"]["full"]["validation"]["float_bpq"], places=12)
                self.assertAlmostEqual(case["validation_quantized_delta_from_full"],
                                       case["validation"]["quantized_bpq"]
                                       - report["cases"]["full"]["validation"]["quantized_bpq"], places=12)
                self.assertFalse(any(key.startswith(("output_head.", "transformer.norm."))
                                     for key in case["reverted_parameters"]))
            self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
            self.assertFalse(any(path.suffix in (".pt", ".fqdc") for path in output.rglob("*")))
            with self.assertRaisesRegex(FileExistsError, "empty"):
                ablation.run_ablation(diagnostic, output_dir=output,
                                      device=torch.device("cpu"), progress=False)

    def test_headers_are_not_used_as_model_features(self):
        import codec.ablate_adaptation as ablation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, diagnostic, _ = prepared_diagnostic(root)
            first = ablation.run_ablation(diagnostic, output_dir=root / "normal",
                                         device=torch.device("cpu"), eval_batch_reads=4, progress=False)
            original_score = ablation.score_records

            def rename_headers(model, records, **kwargs):
                changed = [replace(record, header=b"@ignored:tile:x:y", plus=b"+ignored_header")
                           for record in records]
                return original_score(model, changed, **kwargs)

            with mock.patch.object(ablation, "score_records", side_effect=rename_headers):
                second = ablation.run_ablation(diagnostic, output_dir=root / "renamed",
                                              device=torch.device("cpu"), eval_batch_reads=4, progress=False)
            self.assertEqual(first["cases"], second["cases"])

    def test_changed_checkpoint_source_and_split_provenance_are_rejected(self):
        from codec.ablate_adaptation import run_ablation

        for damage in ("base_sha", "source_identity", "sample_content", "indices_sha", "duplicate_indices",
                       "full_config", "full_mode"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, _, diagnostic, _ = prepared_diagnostic(root)
                manifest_path = diagnostic / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if damage == "base_sha":
                    manifest["base_checkpoint_sha256"] = "0" * 64
                elif damage == "source_identity":
                    source.write_bytes(source.read_bytes() + b"\n")
                elif damage == "sample_content":
                    manifest["split"]["sample_content_sha256"] = "0" * 64
                elif damage == "indices_sha":
                    manifest["split"]["train_indices_sha256"] = "0" * 64
                elif damage == "duplicate_indices":
                    arrays_path = diagnostic / "split_indices.npz"
                    with np.load(arrays_path) as loaded:
                        arrays = {key: loaded[key].copy() for key in loaded.files}
                    arrays["train"][1] = arrays["train"][0]
                    np.savez_compressed(arrays_path, **arrays)
                elif damage in ("full_config", "full_mode"):
                    path = diagnostic / "full.best.pt"
                    saved = torch.load(path, map_location="cpu")
                    if damage == "full_config":
                        saved["model_config"]["d_model"] = 16
                    else:
                        saved["mode"] = "lora"
                    torch.save(saved, path)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    run_ablation(diagnostic, output_dir=root / "invalid",
                                 device=torch.device("cpu"), eval_batch_reads=4, progress=False)
