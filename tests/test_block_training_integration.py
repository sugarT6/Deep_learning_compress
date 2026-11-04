"""Small HDF5 end-to-end checks for paid block-conditioned training."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import codec.train as train_module
from codec.block_features import decode_block_features, encode_block_features
from codec.block_training import BLOCK_SELECTION_METRIC, transfer_block_initialization
from codec.checkpoint import load_training_checkpoint, save_training_checkpoint
from codec.datasets import (
    DirectQualityDataset, EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS,
    EXPANDED_UNSEEN_DATASETS,
)
from codec.model import DirectQualityTransformer
from codec.train import BalancedTrainingSampler, evaluate_validation, run_training
import test_codec_training as legacy_helpers


class BlockTrainingIntegrationTest(unittest.TestCase):
    _create_dataset_cache = legacy_helpers.DirectQualityTrainingTest._create_dataset_cache

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20261104)

    def fixture(self, root):
        datasets = (DirectQualityDataset("BLOCK_A", "Illumina", True),
                    DirectQualityDataset("BLOCK_B", "MGI/BGI", True))
        for index, dataset in enumerate(datasets):
            self._create_dataset_cache(root, dataset, read_count=30, offset=index * 9)
        return datasets

    def parent_checkpoint(self, root, datasets, *, train_fraction=.65, name="parent.pt", model_config=None):
        model = DirectQualityTransformer(model_config or legacy_helpers.tiny_config())
        with BalancedTrainingSampler(datasets, root, train_fraction=train_fraction,
                                     batch_reads=4, seed=31, steps_per_epoch=3) as sampler:
            split = sampler.data_split_manifest(())
            path = root / name
            save_training_checkpoint(
                path, model=model, optimizer=None, epoch=1, global_step=0,
                data_split=split, sampler_config=sampler.config_dict(),
                sampler_statistics=sampler.statistics_dict(),
                best_validation_bits_per_quality=6.0,
                validation_metrics={"dataset_macro_bits_per_quality": 6.0},
            )
        return path

    def kwargs(self, root, datasets, *, model_config=None, parent=None, **changes):
        values = dict(
            cache_dir=root,
            model_config=model_config or replace(legacy_helpers.tiny_config(),
                                                block_quality_features=True, block_reads=4,
                                                block_conditioning_dim=5),
            steps_per_epoch=3, batch_reads=4, microbatch_reads=2,
            train_fraction=.65, validation_max_reads_per_file=5,
            learning_rate=.001, weight_decay=0.0, grad_clip=1.0,
            device=torch.device("cpu"), seed=31, train_datasets=datasets,
            unseen_datasets=(), progress=False,
        )
        if parent is not None:
            values["init_from"] = parent
        values.update(changes)
        return values

    def assert_nested_exact(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_exact(a, b)
        else:
            self.assertEqual(left, right)

    def test_sampler_ceil_split_draws_complete_file_relative_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = self.fixture(root)
            with BalancedTrainingSampler(datasets, root, train_fraction=.65,
                                         batch_reads=4, feature_block_reads=4,
                                         seed=31, steps_per_epoch=10) as sampler:
                # floor(30*.65)=19, rounded upward to 20; validation starts at 20.
                for dataset in datasets:
                    self.assertEqual(sampler.train_stop_for(dataset.accession), 20)
                manifest = sampler.data_split_manifest(())
                for entry in manifest["datasets"].values():
                    self.assertEqual(entry["train_range"], [0, 20])
                    self.assertEqual(entry["validation_range"], [20, 30])
                seen = {dataset.accession: [] for dataset in datasets}
                for _ in range(10):
                    sample = sampler.sample_batch()
                    ids = sample.batch.read_indices
                    self.assertEqual(len(ids), 4)
                    self.assertEqual(int(ids[0]) % 4, 0)
                    np.testing.assert_array_equal(ids, np.arange(ids[0], ids[0] + 4))
                    self.assertTrue(np.all(ids // 4 == ids[0] // 4))
                    self.assertLess(int(ids[-1]), 20)
                    seen[sample.dataset.accession].extend(ids.tolist())
                for values in seen.values():
                    self.assertEqual(sorted(values), list(range(20)))

    def test_plain_checkpoint_transfer_microbatches_and_paid_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = self.fixture(root)
            parent = self.parent_checkpoint(root, datasets)
            parent_state = load_training_checkpoint(parent).model.state_dict()
            samples, forwards = [], []
            sample_batch = BalancedTrainingSampler.sample_batch
            forward_full = DirectQualityTransformer.forward_full

            def record_sample(sampler):
                sample = sample_batch(sampler)
                samples.append(decode_block_features(encode_block_features(sample.batch)).copy())
                return sample

            def record_forward(model, *args, **kwargs):
                if model.training:
                    forwards.append((kwargs["block_features"].detach().clone(),
                                     all(p.requires_grad for p in model.parameters())))
                return forward_full(model, *args, **kwargs)

            with mock.patch.object(BalancedTrainingSampler, "sample_batch", record_sample), \
                    mock.patch.object(DirectQualityTransformer, "forward_full", record_forward):
                path = run_training(output_dir=root / "trained", epochs=1,
                                    **self.kwargs(root, datasets, parent=parent))
            self.assertEqual(len(samples), 3)
            self.assertEqual(len(forwards), 6)
            for index, (z, all_trainable) in enumerate(forwards):
                self.assertTrue(all_trainable)
                self.assertEqual(z.shape, (2, 16))
                np.testing.assert_array_equal(z.numpy(), np.broadcast_to(samples[index // 2], (2, 16)))
            loaded = load_training_checkpoint(path)
            payload = loaded.payload
            self.assertTrue(loaded.model.config.block_quality_features)
            self.assertEqual(payload["selection_metric"], BLOCK_SELECTION_METRIC)
            self.assertEqual(payload["global_step"], 3)
            self.assertEqual(len(payload["optimizer_state_dict"]["state"]),
                             len(tuple(loaded.model.parameters())))
            self.assertFalse(torch.equal(parent_state["output_head.weight"],
                                         loaded.model.state_dict()["output_head.weight"]))
            self.assertGreater(int(torch.count_nonzero(loaded.model.block_conditioning.encoder[-1].weight)), 0)
            config = json.loads((root / "trained/run_config.json").read_text())
            self.assertEqual(config["validation_read_counts"], {"BLOCK_A": 8, "BLOCK_B": 8})
            self.assertEqual(config["block_training"]["microbatch_reads"], 2)
            metrics = payload["validation_metrics"]
            self.assertEqual(metrics["block_count"], 4)
            self.assertEqual(metrics["block_side_bits"], 4 * 16 * 8)
            for entry in metrics["per_dataset"].values():
                self.assertEqual(entry["reads"], 8)
                self.assertEqual(entry["block_count"], 2)
                self.assertEqual(entry["block_side_bits"], 256)
                self.assertAlmostEqual(entry["quantized_Q_plus_block_side_bits_per_quality"],
                                       (entry["quantized_bits"] + 256) / entry["symbols"], places=12)
            self.assertEqual(payload["best_validation_bits_per_quality"], metrics[BLOCK_SELECTION_METRIC])

    def test_constant_control_has_same_initial_Q_and_parameters_but_no_side_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = self.fixture(root)
            parent = load_training_checkpoint(self.parent_checkpoint(root, datasets)).model
            active_config = self.kwargs(root, datasets)["model_config"]
            features_model = DirectQualityTransformer(active_config).eval()
            transfer_block_initialization(features_model, parent)
            constant = DirectQualityTransformer(replace(active_config, block_feature_mode="constant")).eval()
            constant.load_state_dict(features_model.state_dict(), strict=True)
            self.assertEqual(sum(p.numel() for p in features_model.parameters()),
                             sum(p.numel() for p in constant.parameters()))
            self.assert_nested_exact(features_model.state_dict(), constant.state_dict())
            with BalancedTrainingSampler(datasets, root, train_fraction=.65, batch_reads=4,
                                         feature_block_reads=4, seed=31) as sampler:
                a, active = evaluate_validation(features_model, sampler, max_reads_per_file=5,
                                               device=torch.device("cpu"))
                b, control = evaluate_validation(constant, sampler, max_reads_per_file=5,
                                                device=torch.device("cpu"))
            self.assertEqual(active["quantized_Q_bits"], control["quantized_Q_bits"])
            self.assertEqual(active["block_side_bits"], 512)
            self.assertEqual(control["block_side_bits"], 0)
            self.assertEqual(active["dataset_macro_bits_per_quality"], control["dataset_macro_bits_per_quality"])
            for x, y in zip(a, b):
                self.assertEqual(x.total_bits, y.total_bits)
                self.assertEqual(x.quantized_bits, y.quantized_bits)
                self.assertAlmostEqual(x.quantized_Q_plus_block_side_bits_per_quality
                                       - y.quantized_Q_plus_block_side_bits_per_quality,
                                       256 / x.symbols, places=12)

    def test_resume_matches_uninterrupted_model_optimizer_sampler_and_rng_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = self.fixture(root)
            parent = self.parent_checkpoint(root, datasets,
                                            model_config=replace(legacy_helpers.tiny_config(), dropout=.2))
            config = replace(self.kwargs(root, datasets)["model_config"], dropout=.2)
            kwargs = self.kwargs(root, datasets, model_config=config, parent=parent)
            run_training(output_dir=root / "whole", epochs=2, **kwargs)
            run_training(output_dir=root / "first", epochs=1, **kwargs)
            resume_kwargs = dict(kwargs)
            resume_kwargs.pop("init_from")
            run_training(output_dir=root / "resumed", epochs=2,
                         resume=root / "first/last.pt", **resume_kwargs)
            a = load_training_checkpoint(root / "whole/last.pt").payload
            b = load_training_checkpoint(root / "resumed/last.pt").payload
            for key in ("model_state_dict", "optimizer_state_dict", "sampler_state", "sampler_statistics"):
                self.assert_nested_exact(a[key], b[key])
            for key in ("python_rng", "numpy_rng", "torch_rng", "control_state", "training_quality_tokens"):
                self.assert_nested_exact(a["training_state"][key], b["training_state"][key])
            self.assertEqual(a["validation_metrics"], b["validation_metrics"])
            with self.assertRaisesRegex(ValueError, "run configuration"):
                run_training(output_dir=root / "invalid_resume", epochs=2,
                             resume=root / "first/last.pt",
                             **dict(resume_kwargs, microbatch_reads=4))

    def test_token_weighted_micro_gradients_and_update_match_whole_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = self.fixture(root)
            parent = self.parent_checkpoint(root, datasets)
            kwargs = self.kwargs(root, datasets, parent=parent, steps_per_epoch=1)
            snapshots = []
            clip = torch.nn.utils.clip_grad_norm_

            def record_gradients(parameters, *args, **kwargs):
                parameters = list(parameters)
                snapshots.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
                return clip(parameters, *args, **kwargs)

            with mock.patch.object(torch.nn.utils, "clip_grad_norm_", record_gradients):
                # One-read slices have unequal symbol counts (one or two Q),
                # so averaging micro losses equally would fail this check.
                run_training(output_dir=root / "micro", epochs=1, **dict(kwargs, microbatch_reads=1))
                run_training(output_dir=root / "full", epochs=1, **dict(kwargs, microbatch_reads=4))
            self.assertEqual(len(snapshots), 2)
            for a, b in zip(*snapshots):
                self.assertIsNotNone(a)
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
            a = load_training_checkpoint(root / "micro/last.pt").model.state_dict()
            b = load_training_checkpoint(root / "full/last.pt").model.state_dict()
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=3e-6, msg=key)

    def test_parent_training_data_cannot_enter_new_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = self.fixture(root)
            parent = self.parent_checkpoint(root, datasets, train_fraction=.85)
            with self.assertRaisesRegex(ValueError, "parent training reads would enter new validation"):
                run_training(output_dir=root / "must_reject", epochs=1,
                             **self.kwargs(root, datasets, parent=parent))
            self.assertFalse((root / "must_reject/last.pt").exists())

    def test_expanded20_cli_keeps_registry_and_family_weight_schedule(self):
        argv = ["codec.train", "--training-set", "expanded20", "--block-quality-features",
                "--block-reads", "256", "--batch-reads", "256", "--microbatch-reads", "32",
                "--device", "cpu", "--no-progress"]
        with mock.patch("sys.argv", argv), mock.patch.object(train_module, "run_training",
                                                              return_value=Path("best.pt")) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train_module.main(), 0)
        call = run.call_args.kwargs
        self.assertIs(call["train_datasets"], EXPANDED_TRAIN_DATASETS)
        self.assertEqual(len(call["train_datasets"]), 20)
        self.assertIs(call["unseen_datasets"], EXPANDED_UNSEEN_DATASETS)
        self.assertEqual(call["family_weights"], EXPANDED_FAMILY_WEIGHTS)
        self.assertEqual(sorted(call["family_weights"].values()), [1, 1, 3, 5])
        self.assertTrue(call["model_config"].block_quality_features)
        self.assertEqual(call["microbatch_reads"], 32)


if __name__ == "__main__":
    unittest.main()
