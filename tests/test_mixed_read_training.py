"""Real-cache FP32 Transformer continuation, objective and logging checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from codec.checkpoint import load_training_checkpoint
from codec.datasets import DirectQualityDataset
from codec.mixed_read_training import preflight, run_training
from codec.model import DirectQualityModelConfig
from codec.training_cache import create_training_cache
from codec.training_control import TrainingControlConfig


class MixedReadTransformerTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.datasets = (DirectQualityDataset("TRAIN_A", "Illumina", True),
                         DirectQualityDataset("TRAIN_B", "MGI/BGI", True))
        for source, dataset in enumerate(self.datasets):
            source_path = self.root / f"{dataset.accession}.fastq"
            rows = []
            for index in range(12):
                length = 1 + 4 * source + index % 3
                bases = (b"ACGTN" * 2)[:length]
                qualities = bytes(33 + (source + index + cycle) % 42
                                  for cycle in range(length))
                rows.append(b"@read" + str(index).encode() + b"\n" + bases
                            + b"\n+\n" + qualities + b"\n")
            source_path.write_bytes(b"".join(rows))
            create_training_cache(source_path, dataset.cache_path(self.root))
        self.config = DirectQualityModelConfig(d_model=8, num_heads=2,
            num_layers=2, feedforward_dim=16, dropout=.2, context_length=4)
        self.kwargs = dict(model_config=self.config, cache_dir=self.root,
            epochs=2, steps_per_epoch=2, batch_reads=4, train_fraction=.75,
            validation_max_reads_per_file=2, train_datasets=self.datasets,
            unseen_datasets=(), family_weights=None, device="cpu", seed=71,
            progress=False, preload_training_values=True,
            log_loss_curves=True, loss_smoothing_window=2,
            control_config=TrainingControlConfig(lr_scheduler="plateau", lr_patience=0))

    def train(self, name, **changes):
        return run_training(output_dir=self.root / name, **{**self.kwargs, **changes})

    def load(self, name):
        return load_training_checkpoint(self.root / name / "last.pt").payload

    def assert_optimizer_equal(self, left, right):
        self.assertEqual(left["param_groups"], right["param_groups"])
        self.assertEqual(set(left["state"]), set(right["state"]))
        for parameter, state in left["state"].items():
            for name, value in state.items():
                other = right["state"][parameter][name]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, other), (parameter, name))
                else:
                    self.assertEqual(value, other)

    def test_dropout_resume_is_exact_with_optimizer_sampler_and_complete_loss_curves(self):
        self.train("full")
        self.train("first", epochs=1)
        self.train("resume", resume=self.root / "first" / "last.pt")
        full, resumed = self.load("full"), self.load("resume")
        for name in ("sampler_state", "validation_metrics", "global_step"):
            self.assertEqual(full[name], resumed[name], name)
        for name in ("training_quality_tokens", "training_tokens_by_dataset",
                     "training_reads_by_dataset", "control_state"):
            self.assertEqual(full["training_state"][name], resumed["training_state"][name], name)
        for name, tensor in full["model_state_dict"].items():
            self.assertTrue(torch.equal(tensor, resumed["model_state_dict"][name]), name)
        self.assert_optimizer_equal(full["optimizer_state_dict"], resumed["optimizer_state_dict"])
        root = self.root / "resume"
        batches = [json.loads(line) for line in
                   (root / "training_batch_log.jsonl").read_text().splitlines()]
        self.assertEqual([row["global_step"] for row in batches], [1, 2, 3, 4])
        for row in batches:
            self.assertEqual(sum(row["source_read_counts"].values()), 4)
            self.assertEqual(sum(row["source_quality_symbols"].values()), row["quality_symbols"])
            self.assertAlmostEqual(row["loss_nats"], sum(row["source_loss_nats"].values()) / 2,
                                   places=6)
            self.assertAlmostEqual(row["pooled_loss_nats"], sum(
                row["source_loss_nats"][name] * count
                for name, count in row["source_quality_symbols"].items())
                / row["quality_symbols"], places=6)
        epochs = [json.loads(line) for line in
                  (root / "training_log.jsonl").read_text().splitlines()]
        self.assertEqual([row["epoch"] for row in epochs], [1, 2])
        for row in epochs:
            group = [batch for batch in batches if batch["epoch"] == row["epoch"]]
            self.assertAlmostEqual(row["train_objective_loss"],
                                   sum(batch["loss_nats"] for batch in group) / len(group))
            self.assertEqual(sum(row["epoch_reads_by_dataset"].values()), 8)
            self.assertEqual(sum(row["epoch_tokens_by_dataset"].values()),
                             row["epoch_training_quality_tokens"])
        for name in ("loss_batch.png", "loss_epoch.png"):
            self.assertEqual((root / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(list((root / "tensorboard").glob("events.out.tfevents.*")))
        self.assertEqual(self.load("resume")["model_config"], self.config.to_dict())

    def test_tensorboard_and_plotting_do_not_change_dropout_training_rng(self):
        self.train("logged", epochs=1)
        self.train("plain", epochs=1, log_loss_curves=False)
        left, right = self.load("logged"), self.load("plain")
        for name, tensor in left["model_state_dict"].items():
            self.assertTrue(torch.equal(tensor, right["model_state_dict"][name]), name)
        self.assert_optimizer_equal(left["optimizer_state_dict"], right["optimizer_state_dict"])
        self.assertTrue(torch.equal(left["training_state"]["torch_rng"],
                                    right["training_state"]["torch_rng"]))

    def test_microchunks_keep_whole_batch_source_denominators(self):
        config = replace(self.config, dropout=0.)
        gradients = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def capture_gradients(parameters, *args, **kwargs):
            parameters = list(parameters)
            gradients.append([parameter.grad.detach().clone() for parameter in parameters])
            return original_clip(parameters, *args, **kwargs)

        with mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=capture_gradients):
            self.train("whole", model_config=config, epochs=1,
                       steps_per_epoch=1, log_loss_curves=False)
            self.train("chunks", model_config=config, epochs=1,
                       steps_per_epoch=1, microbatch_reads=1, log_loss_curves=False)
        left, right = self.load("whole"), self.load("chunks")
        self.assertEqual(len(gradients), 2)
        # Compare the actual summed objective gradients. Adam can amplify tiny
        # FP32 reduction differences in theoretically zero attention-bias grads.
        for index, (whole, chunks) in enumerate(zip(*gradients)):
            self.assertTrue(torch.allclose(whole, chunks, atol=3e-7, rtol=1e-5), index)
        objectives = [json.loads((self.root / name / "training_log.jsonl").read_text())
                      ["train_objective_loss"] for name in ("whole", "chunks")]
        self.assertAlmostEqual(*objectives, places=6)
        self.assertEqual(left["sampler_state"], right["sampler_state"])
        self.assertEqual(left["training_state"]["training_quality_tokens"],
                         right["training_state"]["training_quality_tokens"])

    def test_preflight_does_not_treat_attention_window_as_read_length_limit(self):
        report = preflight(model_config=self.config, cache_dir=self.root,
            batch_reads=4, train_fraction=.75, steps_per_epoch=2,
            validation_max_reads_per_file=2, train_datasets=self.datasets,
            unseen_datasets=(), family_weights=None)
        self.assertEqual(report["datasets"]["TRAIN_B"]["maximum_read_length"], 7)
        self.assertGreater(7, self.config.context_length)
        self.assertFalse(report["reads_truncated"])
        self.assertEqual(report["data_split"]["datasets"]["TRAIN_A"]["train_range"], [0, 9])
        self.assertEqual(report["data_split"]["datasets"]["TRAIN_A"]["validation_range"], [9, 12])

    def test_refuses_legacy_resume_changed_contract_and_nonempty_output_before_writes(self):
        self.train("first", epochs=1)
        source = self.root / "first" / "last.pt"
        legacy = torch.load(source, map_location="cpu")
        legacy["training_state"] = {"version": 1}
        old = self.root / "legacy.pt"
        torch.save(legacy, old)
        with self.assertRaisesRegex(ValueError, "mixed-read Transformer training state"):
            self.train("legacy_resume", resume=old)
        self.assertFalse((self.root / "legacy_resume").exists())
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.train("wrong_lr", resume=source, learning_rate=.0005)
        self.assertFalse((self.root / "wrong_lr").exists())
        marker = self.root / "nonempty"
        marker.mkdir()
        (marker / "keep").write_text("keep")
        with self.assertRaisesRegex(FileExistsError, "new empty"):
            self.train("nonempty")
        self.assertEqual((marker / "keep").read_text(), "keep")
        best_path = self.root / "first" / "best.pt"
        best = torch.load(best_path, map_location="cpu")
        best["validation_metrics"]["dataset_macro_bits_per_quality"] += 1
        torch.save(best, best_path)
        with self.assertRaisesRegex(ValueError, "matching best.pt"):
            self.train("wrong_best", resume=source)
        self.assertFalse((self.root / "wrong_best").exists())


if __name__ == "__main__":
    unittest.main()
