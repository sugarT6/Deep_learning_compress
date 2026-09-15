import contextlib
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
    DATASETS,
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


if __name__ == "__main__":
    unittest.main()
