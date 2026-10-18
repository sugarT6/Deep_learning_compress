import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

import torch

import codec.train as train_module
from codec.checkpoint import load_training_checkpoint
from codec.datasets import (
    ADDITIONAL_TRAIN_DATASETS,
    DATASETS,
    EXPANDED_FAMILY_WEIGHTS,
    EXPANDED_TRAIN_DATASETS,
    EXPANDED_UNSEEN_DATASETS,
    NEW_EXTERNAL_HOLDOUT_DATASETS,
    TRAIN_DATASETS,
    UNSEEN_DATASETS,
    UNSEEN_INSTRUMENT_DATASETS,
    DirectQualityDataset,
)
from codec.model import (
    DirectQualityModelConfig, DirectQualityTransformer,
    fastq_batch_to_tensors, masked_cross_entropy,
)
from codec.train import BalancedTrainingSampler, run_training
from codec.training_control import TrainingControlConfig
from codec.training_cache import create_training_cache, default_cache_path


def small_fastq(read_count, offset=0):
    records = []
    for index in range(read_count):
        quality = bytes([33 + ((index + offset) % 42), 33 + ((index + offset + 1) % 42)])
        sequence = b"AC" if index % 2 else b"A"
        quality = quality[:len(sequence)]
        records.append(
            b"@r" + str(index).encode("ascii") + b"\n" + sequence + b"\n+\n" + quality + b"\n"
        )
    return b"".join(records)


def tiny_config():
    return DirectQualityModelConfig(
        prev_q_embed_dim=4,
        qmer_embed_dim=2,
        base_embed_dim=3,
        base_conv_channels=3,
        base_context_dim=4,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        context_length=4,
        dropout=0.0,
    )


class DirectQualityTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _create_dataset_cache(self, root, dataset, read_count=12, offset=0):
        fastq_path = root / f"{dataset.accession}.fq"
        fastq_path.write_bytes(small_fastq(read_count, offset=offset))
        cache_path = default_cache_path(fastq_path, root)
        create_training_cache(fastq_path, cache_path)
        return cache_path

    def test_fixed_registry_is_ten_train_nine_unseen(self):
        self.assertEqual(len(DATASETS), 19)
        self.assertEqual(len(TRAIN_DATASETS), 10)
        self.assertEqual(len(UNSEEN_DATASETS), 9)
        self.assertEqual(
            {dataset.accession for dataset in UNSEEN_INSTRUMENT_DATASETS},
            {"SRR10965088", "SRR29287266"},
        )

    def test_dataset_sampler_is_balanced_and_never_reads_validation(self):
        datasets = (
            DirectQualityDataset("A1", "Illumina", True),
            DirectQualityDataset("A2", "Illumina", True),
            DirectQualityDataset("B1", "MGI/BGI", True),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, dataset in enumerate(datasets):
                self._create_dataset_cache(root, dataset, offset=index)
            with BalancedTrainingSampler(
                datasets,
                root,
                train_fraction=0.75,
                batch_reads=1,
                seed=19,
                steps_per_epoch=3000,
            ) as sampler:
                for _ in range(3000):
                    sample = sampler.sample_batch()
                    self.assertLess(
                        int(sample.batch.read_indices.max()),
                        sampler.train_stop_for(sample.dataset.accession),
                    )
                statistics = sampler.statistics_dict()

        self.assertEqual(statistics["total_batches"], 3000)
        self.assertAlmostEqual(
            statistics["families"]["Illumina"]["proportion"], 2 / 3, delta=0.001
        )
        self.assertAlmostEqual(
            statistics["families"]["MGI/BGI"]["proportion"], 1 / 3, delta=0.001
        )
        a1 = statistics["datasets"]["A1"]["proportion"]
        a2 = statistics["datasets"]["A2"]["proportion"]
        self.assertAlmostEqual(a1, 1 / 3, delta=0.001)
        self.assertAlmostEqual(a2, 1 / 3, delta=0.001)

    def test_two_step_training_prediction_and_checkpoint_smoke(self):
        train_datasets = (
            DirectQualityDataset("TRAIN_A", "Illumina", True),
            DirectQualityDataset("TRAIN_B", "MGI/BGI", True),
        )
        unseen = (DirectQualityDataset("NEVER_USED", "Illumina", False, True),)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, dataset in enumerate(train_datasets):
                self._create_dataset_cache(root, dataset, read_count=400, offset=index)
            output_dir = root / "run"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                best_path = run_training(
                    cache_dir=root,
                    output_dir=output_dir,
                    model_config=tiny_config(),
                    epochs=1,
                    steps_per_epoch=2,
                    batch_reads=256,
                    train_fraction=0.75,
                    validation_max_reads_per_file=2,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    grad_clip=1.0,
                    device=torch.device("cpu"),
                    seed=11,
                    train_datasets=train_datasets,
                    unseen_datasets=unseen,
                    progress=True,
                )

            display = stdout.getvalue() + stderr.getvalue()
            self.assertIn("epoch=1 loss=", display)
            self.assertNotIn("dataset=", display)
            self.assertNotIn("TRAIN_A", display)
            self.assertNotIn("TRAIN_B", display)
            if train_module.tqdm is not None:
                self.assertIn("epoch 1/1", display)

            self.assertTrue(best_path.is_file())
            self.assertTrue((output_dir / "last.pt").is_file())
            self.assertTrue((output_dir / "run_config.json").is_file())
            self.assertTrue((output_dir / "training_log.jsonl").is_file())
            statistics = json.loads(
                (output_dir / "sampling_statistics.json").read_text("utf-8")
            )
            self.assertEqual(statistics["total_batches"], 2)
            self.assertEqual(
                [v["count"] for v in statistics["datasets"].values()], [1, 1]
            )
            self.assertEqual(
                [v["proportion"] for v in statistics["families"].values()], [.5, .5]
            )
            self.assertAlmostEqual(
                sum(
                    entry["proportion"]
                    for entry in statistics["datasets"].values()
                ),
                1.0,
            )

            loaded = load_training_checkpoint(best_path)
            payload = loaded.payload
            self.assertEqual(payload["global_step"], 2)
            self.assertEqual(payload["selection_metric"], "dataset_macro_bits_per_quality")
            self.assertEqual(payload["sampler_state"]["total_samples"], 2)
            self.assertEqual(payload["sampler_statistics"], statistics)
            self.assertFalse(
                payload["data_split"]["unseen_used_for_checkpoint_selection"]
            )
            self.assertEqual(
                payload["data_split"]["unseen_datasets"], ["NEVER_USED"]
            )
            self.assertEqual(
                set(payload["data_split"]["datasets"]),
                {"TRAIN_A", "TRAIN_B"},
            )
            validation = payload["validation_metrics"]
            self.assertGreater(validation["symbol_micro_bits_per_quality"], 0.0)
            self.assertEqual(set(validation["per_dataset"]), {"TRAIN_A", "TRAIN_B"})
            config = json.loads((output_dir / "run_config.json").read_text())
            self.assertEqual(config["validation_read_counts"], {"TRAIN_A":2, "TRAIN_B":2})
            self.assertIn("worst_dataset", validation)
            self.assertIn("platform_family_macro_bits_per_quality", validation)

    def test_exact_ten_file_schedule_and_rotating_remainders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in TRAIN_DATASETS:
                self._create_dataset_cache(root, dataset)
            for steps in (2000, 13, 3):
                with self.subTest(steps=steps), BalancedTrainingSampler(
                    TRAIN_DATASETS, root, train_fraction=0.75,
                    batch_reads=256, seed=7, steps_per_epoch=steps,
                ) as sampler:
                    # The scheduling test does not need repeated HDF5 reads.
                    for reader in sampler._readers.values():
                        reader.read_range = mock.Mock(return_value=None)
                    for epoch in range(10):
                        for _ in range(steps):
                            sampler.sample_batch()
                        counts = [v["count"] for v in sampler.statistics_dict()["datasets"].values()]
                        self.assertLessEqual(max(counts) - min(counts), 1)
                        if steps == 2000:
                            self.assertEqual(counts, [200 * (epoch + 1)] * 10)
                    stats = sampler.statistics_dict()
                    if steps == 2000:
                        for family, proportion in (("MGI/BGI", .3), ("Illumina", .5), ("ABI SOLiD", .1), ("Ion Torrent", .1)):
                            self.assertEqual(stats["families"][family]["proportion"], proportion)

    def test_blocks_tail_epoch_cursor_seed_and_checkpoint_continuation(self):
        dataset = TRAIN_DATASETS[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 640 training reads: 256,256,128. Separately verify the production
            # 1.8M layout without allocating a production cache.
            self._create_dataset_cache(root, dataset, read_count=800)
            kwargs = dict(train_fraction=.8, batch_reads=256, seed=9, steps_per_epoch=2)
            with BalancedTrainingSampler([dataset], root, **kwargs) as first, BalancedTrainingSampler([dataset], root, **kwargs) as second:
                seen = []
                for _ in range(3):
                    a, b = first.sample_batch(), second.sample_batch()
                    np.testing.assert_array_equal(a.batch.read_indices, b.batch.read_indices)
                    seen.extend(a.batch.read_indices.tolist())
                self.assertEqual(sorted(seen), list(range(640)))
                self.assertEqual(first._epoch, 2)
                self.assertEqual(first._block_rounds[dataset.accession], 0)
                # Serialize, resume mid-epoch and cross several reshuffles.
                path = root / "sampler.pt"
                torch.save(first.state_dict(), path)
                second.load_state_dict(torch.load(path))
                for _ in range(15):
                    a, b = first.sample_batch(), second.sample_batch()
                    np.testing.assert_array_equal(a.batch.read_indices, b.batch.read_indices)
                self.assertEqual(first.state_dict(), second.state_dict())
                first._train_stops[dataset.accession] = 1800000
                order = first._block_order(dataset, 0)
                lengths = [min(256, 1800000 - i * 256) for i in order]
                self.assertEqual(len(order), 7032)
                self.assertEqual(lengths.count(256), 7031)
                self.assertEqual(lengths.count(64), 1)
                self.assertEqual(sum(lengths), 1800000)

    def test_tail64_is_read_and_unseen_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = TRAIN_DATASETS[0]
            self._create_dataset_cache(root, dataset, read_count=640)
            with BalancedTrainingSampler([dataset], root, train_fraction=.9,
                                         batch_reads=256, seed=2) as sampler:
                batches = [sampler.sample_batch().batch for _ in range(3)]
                self.assertEqual(sorted(b.read_count for b in batches), [64,256,256])
                self.assertEqual(sorted(np.concatenate([b.read_indices for b in batches]).tolist()), list(range(576)))
                batch = next(b for b in batches if b.read_count == 256)
                tensors = fastq_batch_to_tensors(batch, torch.device("cpu"))
                model = DirectQualityTransformer(tiny_config())
                logits = model.forward_full(**tensors)
                self.assertEqual(tuple(logits.shape), (256, 2, 42))
                self.assertEqual(int(tensors["active_mask"].sum()), 384)
                loss = masked_cross_entropy(logits, tensors["qualities"], tensors["active_mask"])
                altered = logits.clone()
                altered[~tensors["active_mask"]] = 100.
                torch.testing.assert_close(loss, masked_cross_entropy(altered, tensors["qualities"], tensors["active_mask"]))
            with self.assertRaisesRegex(ValueError, "unseen"):
                BalancedTrainingSampler(UNSEEN_DATASETS, root, train_fraction=.9,
                                        batch_reads=256, seed=2)
        self.assertEqual(train_module.build_parser().parse_args([]).batch_reads, 256)

    def test_macro_selection_and_training_resume(self):
        dataset = TRAIN_DATASETS[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_dataset_cache(root, dataset, read_count=640)
            kwargs = dict(cache_dir=root, model_config=replace(tiny_config(), dropout=.2), steps_per_epoch=1,
                          batch_reads=256, train_fraction=.9, validation_max_reads_per_file=2,
                          learning_rate=.001, weight_decay=0., grad_clip=1.,
                          device=torch.device("cpu"), seed=12, train_datasets=[dataset],
                          progress=False)
            def metrics(macro, micro):
                return [], {"dataset_macro_bits_per_quality": macro,
                            "symbol_micro_bits_per_quality": micro}
            with mock.patch.object(train_module, "evaluate_validation", side_effect=[metrics(1., 2.), metrics(1.5, 1.)]):
                run_training(output_dir=root / "whole", epochs=2, **kwargs)
            self.assertEqual(load_training_checkpoint(root / "whole/best.pt").payload["epoch"], 1)
            with mock.patch.object(train_module, "evaluate_validation", return_value=metrics(1.,2.)):
                run_training(output_dir=root / "first", epochs=1, **kwargs)
            with mock.patch.object(train_module, "evaluate_validation", return_value=metrics(1.5,1.)):
                run_training(output_dir=root / "resumed", epochs=2,
                             resume=root / "first/last.pt", **kwargs)
            a = load_training_checkpoint(root / "whole/last.pt").payload
            b = load_training_checkpoint(root / "resumed/last.pt").payload
            self.assertEqual(a["sampler_state"], b["sampler_state"])
            for key in a["model_state_dict"]:
                torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
            self.assertEqual(load_training_checkpoint(root / "resumed/best.pt").payload["epoch"], 1)

    def _control_training_kwargs(self, root):
        dataset = TRAIN_DATASETS[0]
        self._create_dataset_cache(root, dataset, read_count=640)
        return dict(cache_dir=root, model_config=replace(tiny_config(), dropout=.2),
                    steps_per_epoch=1, batch_reads=256, train_fraction=.9,
                    validation_max_reads_per_file=2, learning_rate=.001,
                    weight_decay=0., grad_clip=1., device=torch.device("cpu"),
                    seed=12, train_datasets=[dataset], progress=False)

    @staticmethod
    def _validation(bits):
        return [], {"dataset_macro_bits_per_quality": bits,
                    "symbol_micro_bits_per_quality": bits}

    def test_lr_reductions_resume_with_identical_model_optimizer_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = self._control_training_kwargs(root)
            kwargs["control_config"] = TrainingControlConfig(
                lr_scheduler="plateau", lr_patience=0, min_lr=1e-4)
            scores = [self._validation(v) for v in (1., 1.2, 1.3, .9)]
            with mock.patch.object(train_module, "evaluate_validation", side_effect=scores):
                run_training(output_dir=root / "whole", epochs=4, **kwargs)
            with mock.patch.object(train_module, "evaluate_validation", side_effect=scores[:2]):
                run_training(output_dir=root / "first", epochs=2, **kwargs)
            with mock.patch.object(train_module, "evaluate_validation", side_effect=scores[2:]):
                run_training(output_dir=root / "resumed", epochs=4,
                             resume=root / "first/last.pt", **kwargs)
            a = load_training_checkpoint(root / "whole/last.pt").payload
            b = load_training_checkpoint(root / "resumed/last.pt").payload
            self.assertEqual(a["sampler_state"], b["sampler_state"])
            self.assertEqual(a["training_state"]["control_state"], b["training_state"]["control_state"])
            self.assertEqual(a["optimizer_state_dict"]["param_groups"], b["optimizer_state_dict"]["param_groups"])
            for key in a["model_state_dict"]:
                torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
            for key, state in a["optimizer_state_dict"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b["optimizer_state_dict"]["state"][key][name], rtol=0, atol=0)
            logs = [json.loads(line) for line in (root / "whole/training_log.jsonl").read_text().splitlines()]
            self.assertEqual([x["learning_rate"] for x in logs], [.001, .001, .0005, .00025])
            self.assertFalse(json.loads((root / "whole/training_summary.json").read_text())["stopped_early"])

    def test_early_stop_state_resumes_and_strict_improvements_still_save_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = self._control_training_kwargs(root)
            kwargs["control_config"] = TrainingControlConfig(
                lr_scheduler="plateau", lr_patience=0,
                early_stopping_patience=2, min_delta=.01)
            scores = [self._validation(v) for v in (1., .995, .994)]
            with mock.patch.object(train_module, "evaluate_validation", side_effect=scores):
                run_training(output_dir=root / "whole", epochs=10, **kwargs)
            with mock.patch.object(train_module, "evaluate_validation", side_effect=scores[:2]):
                run_training(output_dir=root / "first", epochs=2, **kwargs)
            with mock.patch.object(train_module, "evaluate_validation", side_effect=scores[2:]):
                run_training(output_dir=root / "resumed", epochs=10,
                             resume=root / "first/last.pt", **kwargs)
            for name in ("whole", "resumed"):
                summary = json.loads((root / name / "training_summary.json").read_text())
                self.assertTrue(summary["stopped_early"])
                self.assertEqual(summary["completed_epoch"], 3)
                self.assertEqual(load_training_checkpoint(root / name / "best.pt").payload["epoch"], 3)

    def test_legacy_checkpoint_can_start_controls_but_new_resume_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = self._control_training_kwargs(root)
            with mock.patch.object(train_module, "evaluate_validation", return_value=self._validation(1.)):
                run_training(output_dir=root / "legacy", epochs=1, **kwargs)
            for name in ("best.pt", "last.pt"):
                path = root / "legacy" / name
                payload = torch.load(path)
                payload["training_state"]["version"] = 1
                payload["training_state"].pop("control_state")
                payload["training_state"].pop("training_quality_tokens")
                payload["training_state"].pop("training_tokens_by_dataset")
                payload["training_state"]["run_config"].pop("training_control")
                torch.save(payload, path)
            controls = TrainingControlConfig(lr_scheduler="plateau", lr_patience=0)
            with mock.patch.object(train_module, "evaluate_validation", return_value=self._validation(1.1)):
                run_training(output_dir=root / "upgraded", epochs=2,
                             resume=root / "legacy/last.pt", control_config=controls, **kwargs)
            payload = load_training_checkpoint(root / "upgraded/last.pt").payload
            self.assertEqual(payload["global_step"], 2)
            self.assertEqual(payload["training_state"]["version"], 2)
            self.assertIsNone(payload["training_state"]["training_quality_tokens"])
            self.assertIsNone(payload["training_state"]["training_tokens_by_dataset"])
            self.assertEqual(payload["optimizer_state_dict"]["param_groups"][0]["lr"], .0005)
            with self.assertRaisesRegex(ValueError, "controls do not match"):
                run_training(output_dir=root / "mismatch", epochs=3,
                             resume=root / "upgraded/last.pt",
                             control_config=replace(controls, lr_patience=1), **kwargs)

    def test_invalid_training_controls_are_rejected(self):
        for fields in ({"lr_scheduler": "unknown"}, {"lr_factor": 1.},
                       {"lr_patience": -1}, {"min_lr": float("nan")},
                       {"min_delta": -.1}, {"early_stopping_patience": True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                TrainingControlConfig(**fields)

    def test_expanded_registry_preserves_old_sources_and_protects_all_holdouts(self):
        self.assertEqual(EXPANDED_TRAIN_DATASETS[:10], TRAIN_DATASETS)
        expected_new = {
            "ERR15766980", "ERR15801889", "SRR13142212", "SRR33170689",
            "ERR12916630", "ERR15158516", "ERR11454184", "ERR3989434",
            "DRR917081", "SRR29727531",
        }
        self.assertEqual({d.accession for d in ADDITIONAL_TRAIN_DATASETS}, expected_new)
        train_ids = {d.accession for d in EXPANDED_TRAIN_DATASETS}
        self.assertEqual(len(train_ids), 20)
        self.assertEqual(len(EXPANDED_UNSEEN_DATASETS), 14)
        self.assertEqual(len(NEW_EXTERNAL_HOLDOUT_DATASETS), 5)
        self.assertFalse(train_ids & {d.accession for d in EXPANDED_UNSEEN_DATASETS})
        self.assertFalse(any(d.unseen_instrument for d in EXPANDED_UNSEEN_DATASETS))
        # Historical flags and default CLI behavior remain reproducible.
        self.assertEqual(len(UNSEEN_INSTRUMENT_DATASETS), 2)
        self.assertEqual(train_module.build_parser().parse_args([]).training_set, "original10")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_module.build_parser().parse_args(["--resume", "a", "--init-from", "b"])

    def test_expanded_family_schedule_fairness_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in EXPANDED_TRAIN_DATASETS:
                self._create_dataset_cache(root, dataset)
            for steps in (2000, 13, 3):
                kwargs = dict(train_fraction=.75, batch_reads=2, seed=17,
                              steps_per_epoch=steps, family_weights=EXPANDED_FAMILY_WEIGHTS)
                with self.subTest(steps=steps), BalancedTrainingSampler(
                    EXPANDED_TRAIN_DATASETS, root, **kwargs,
                ) as first, BalancedTrainingSampler(EXPANDED_TRAIN_DATASETS, root, **kwargs) as second:
                    for sampler in (first, second):
                        for reader in sampler._readers.values():
                            reader.read_range = mock.Mock(return_value=None)
                    for epoch in range(10):
                        for _ in range(steps):
                            first.sample_batch()
                        stats = first.statistics_dict()
                        for family in EXPANDED_FAMILY_WEIGHTS:
                            counts = [stats["datasets"][d.accession]["count"]
                                      for d in first.datasets if d.platform_family == family]
                            self.assertLessEqual(max(counts) - min(counts), 1)
                            if steps == 2000:
                                self.assertEqual(stats["families"][family]["count"],
                                                 (epoch + 1) * 200 * EXPANDED_FAMILY_WEIGHTS[family])
                    # Ten epochs span complete family cycles even for 3/13 steps.
                    for family, weight in EXPANDED_FAMILY_WEIGHTS.items():
                        self.assertEqual(stats["families"][family]["count"], steps * weight)
                    first.sample_batch()
                    second.load_state_dict(first.state_dict())
                    for _ in range(steps + 5):
                        self.assertEqual(first.sample_batch().dataset, second.sample_batch().dataset)
                    self.assertEqual(first.state_dict(), second.state_dict())
            for weights in ({"Illumina": 5}, {**EXPANDED_FAMILY_WEIGHTS, "MGI/BGI": 0},
                            {**EXPANDED_FAMILY_WEIGHTS, "MGI/BGI": .3}):
                with self.assertRaisesRegex(ValueError, "family weights"):
                    BalancedTrainingSampler(EXPANDED_TRAIN_DATASETS, root,
                                            train_fraction=.9, batch_reads=2, seed=1,
                                            family_weights=weights)

    def test_expanded_training_uses_twenty_validation_sources_and_counts_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in EXPANDED_TRAIN_DATASETS:
                self._create_dataset_cache(root, dataset, read_count=12)
            # No holdout caches exist: training must not try to open them.
            run_training(cache_dir=root, output_dir=root / "expanded", model_config=tiny_config(),
                         epochs=1, steps_per_epoch=20, batch_reads=2, train_fraction=.75,
                         validation_max_reads_per_file=2, learning_rate=.001,
                         weight_decay=0., grad_clip=1., device=torch.device("cpu"), seed=9,
                         train_datasets=EXPANDED_TRAIN_DATASETS, unseen_datasets=EXPANDED_UNSEEN_DATASETS,
                         family_weights=EXPANDED_FAMILY_WEIGHTS, progress=False)
            run = root / "expanded"
            payload = load_training_checkpoint(run / "last.pt").payload
            self.assertEqual(len(payload["validation_metrics"]["per_dataset"]), 20)
            self.assertEqual(len(payload["data_split"]["unseen_datasets"]), 14)
            self.assertEqual(payload["data_split"]["unseen_instrument_datasets"], [])
            corpus = json.loads((run / "training_corpus.json").read_text())
            self.assertEqual(corpus["train_reads"], 180)
            # Nine alternating length-1/2 reads contain thirteen quality tokens.
            self.assertEqual(corpus["train_quality_tokens"], 260)
            log = json.loads((run / "training_log.jsonl").read_text())
            self.assertEqual(log["training_quality_tokens"], log["epoch_training_quality_tokens"])
            self.assertEqual(log["training_quality_tokens"], sum(log["training_tokens_by_dataset"].values()))
            self.assertEqual(payload["training_state"]["training_quality_tokens"], log["training_quality_tokens"])

    def test_weight_initialization_resets_state_and_remains_exactly_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = self._control_training_kwargs(root)
            with mock.patch.object(train_module, "evaluate_validation", return_value=self._validation(1.)):
                run_training(output_dir=root / "old", epochs=3, **kwargs)
            source = root / "old/best.pt"
            source_payload = load_training_checkpoint(source).payload
            added = DirectQualityDataset("NEW", "Illumina", True)
            self._create_dataset_cache(root, added, read_count=640, offset=7)
            kwargs.update(train_datasets=[TRAIN_DATASETS[0], added],
                          family_weights={"MGI/BGI": 3, "Illumina": 5})
            first_step = []
            original_step = torch.optim.AdamW.step
            def check_step(optimizer, *args, **kw):
                if not first_step:
                    self.assertFalse(optimizer.state)
                    self.assertEqual(optimizer.param_groups[0]["lr"], .001)
                    weights = dict(zip(source_payload["model_state_dict"], optimizer.param_groups[0]["params"]))
                    for name, value in weights.items():
                        torch.testing.assert_close(value, source_payload["model_state_dict"][name], rtol=0, atol=0)
                    first_step.append(True)
                return original_step(optimizer, *args, **kw)
            with mock.patch.object(train_module, "evaluate_validation", return_value=self._validation(2.)), \
                    mock.patch.object(torch.optim.AdamW, "step", check_step):
                run_training(output_dir=root / "whole", epochs=2, init_from=source, **kwargs)
            self.assertTrue(first_step)
            with mock.patch.object(train_module, "evaluate_validation", return_value=self._validation(2.)):
                run_training(output_dir=root / "first", epochs=1, init_from=source, **kwargs)
                run_training(output_dir=root / "resumed", epochs=2, resume=root / "first/last.pt", **kwargs)
            whole = load_training_checkpoint(root / "whole/last.pt").payload
            resumed = load_training_checkpoint(root / "resumed/last.pt").payload
            self.assertEqual(whole["global_step"], 2)
            self.assertEqual(whole["sampler_state"], resumed["sampler_state"])
            self.assertEqual(whole["training_state"]["training_quality_tokens"], resumed["training_state"]["training_quality_tokens"])
            for key in whole["model_state_dict"]:
                torch.testing.assert_close(whole["model_state_dict"][key], resumed["model_state_dict"][key], rtol=0, atol=0)
            init = resumed["training_state"]["run_config"]["initialization"]
            self.assertEqual(init["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(init["source_global_step"], source_payload["global_step"])
            self.assertEqual(load_training_checkpoint(root / "first/best.pt").payload["best_validation_bits_per_quality"], 2.)
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                run_training(output_dir=root / "bad_resume", epochs=4, resume=root / "old/last.pt", **kwargs)
            with self.assertRaisesRegex(ValueError, "init-from model configuration"):
                run_training(output_dir=root / "bad_init", epochs=1, init_from=source,
                             **{**kwargs, "model_config": replace(kwargs["model_config"], num_layers=2)})

    def test_cli_selects_expanded_training_and_sampling(self):
        with mock.patch("sys.argv", ["codec.train", "--training-set", "expanded20", "--device", "cpu"]), \
                mock.patch.object(train_module, "run_training", return_value=Path("best.pt")) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train_module.main(), 0)
        self.assertEqual(run.call_args.kwargs["train_datasets"], EXPANDED_TRAIN_DATASETS)
        self.assertEqual(run.call_args.kwargs["unseen_datasets"], EXPANDED_UNSEEN_DATASETS)
        self.assertEqual(run.call_args.kwargs["family_weights"], EXPANDED_FAMILY_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
