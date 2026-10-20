import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from codec.datasets import EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS, EXPANDED_UNSEEN_DATASETS
from codec.model import DirectQualityModelConfig
from codec import train_experiments


class PretrainingExperimentTests(unittest.TestCase):
    def test_presets_change_only_the_intended_architecture_fields(self):
        baseline = DirectQualityModelConfig()
        expected = {
            "ffn1024": {"feedforward_dim": 1024},
            "wide384": {"d_model": 384, "num_heads": 6, "feedforward_dim": 768},
            "base_dilated": {"base_encoder_type": "dilated_cnn"},
            "swiglu": {"ffn_type": "swiglu", "feedforward_dim": 344},
            "qprefix": {"quality_prefix_features": True},
        }
        for name, changes in expected.items():
            with self.subTest(experiment=name):
                self.assertEqual(train_experiments.experiment_config(name), replace(baseline, **changes))

    def test_dry_run_never_creates_a_run_or_starts_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            group = Path(temporary) / "uncreated"
            stdout = io.StringIO()
            with mock.patch.object(train_experiments, "run_training") as train:
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(train_experiments.main([
                        "qprefix", "--run-group", str(group), "--dry-run",
                    ]), 0)
            description = json.loads(stdout.getvalue())
            self.assertEqual(description["target_updates"], 200000)
            self.assertEqual(description["feature_schema"]["version"], 2)
            self.assertEqual(description["initialization"], "random")
            self.assertEqual(len(description["training_accessions"]), 20)
            self.assertFalse(group.exists())
            train.assert_not_called()

    def test_repeated_invocations_allocate_empty_independent_runs_with_same_training_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            group = Path(temporary) / "experiments"
            output_dirs = []

            def train(**kwargs):
                output_dir = kwargs["output_dir"]
                self.assertTrue(output_dir.is_dir())
                self.assertEqual(list(output_dir.iterdir()), [])
                output_dirs.append(output_dir)
                return output_dir / "best.pt"

            with mock.patch.object(Path, "is_file", return_value=True):
                with mock.patch.object(train_experiments, "run_training", side_effect=train) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        for _ in range(2):
                            train_experiments.main(["swiglu", "--run-group", str(group), "--device", "cpu"])
            self.assertNotEqual(output_dirs[0], output_dirs[1])
            self.assertEqual({path.parent for path in output_dirs}, {group})
            for call in run.call_args_list:
                kwargs = call.kwargs
                self.assertEqual(kwargs["epochs"], 100)
                self.assertEqual(kwargs["steps_per_epoch"], 2000)
                self.assertEqual(kwargs["batch_reads"], 256)
                self.assertEqual(kwargs["seed"], 20260909)
                self.assertEqual(kwargs["train_fraction"], 0.9)
                self.assertEqual(kwargs["validation_max_reads_per_file"], 5000)
                self.assertEqual(kwargs["train_datasets"], EXPANDED_TRAIN_DATASETS)
                self.assertEqual(kwargs["unseen_datasets"], EXPANDED_UNSEEN_DATASETS)
                self.assertEqual(kwargs["family_weights"], EXPANDED_FAMILY_WEIGHTS)
                self.assertIsNone(kwargs["resume"])
                self.assertEqual(kwargs["control_config"].lr_scheduler, "plateau")
                self.assertEqual(kwargs["model_config"].ffn_type, "swiglu")


if __name__ == "__main__":
    unittest.main()
