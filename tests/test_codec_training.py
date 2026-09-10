import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

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
from codec.model import DirectQualityModelConfig
from codec.train import BalancedTrainingSampler, run_training
from codec.training_cache import create_training_cache, default_cache_path


def small_fastq(read_count, offset=0):
    records = []
    for index in range(read_count):
        quality = bytes([33 + ((index + offset) % 42), 33 + ((index + offset + 1) % 42)])
        records.append(
            b"@r" + str(index).encode("ascii") + b"\nAC\n+\n" + quality + b"\n"
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

    def test_two_level_sampler_is_balanced_and_never_reads_validation(self):
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
            statistics["families"]["Illumina"]["proportion"], 0.5, delta=0.04
        )
        self.assertAlmostEqual(
            statistics["families"]["MGI/BGI"]["proportion"], 0.5, delta=0.04
        )
        a1 = statistics["datasets"]["A1"]["proportion"]
        a2 = statistics["datasets"]["A2"]["proportion"]
        self.assertAlmostEqual(a1, 0.25, delta=0.04)
        self.assertAlmostEqual(a2, 0.25, delta=0.04)

    def test_two_step_training_prediction_and_checkpoint_smoke(self):
        train_datasets = (
            DirectQualityDataset("TRAIN_A", "Illumina", True),
            DirectQualityDataset("TRAIN_B", "MGI/BGI", True),
        )
        unseen = (DirectQualityDataset("NEVER_USED", "Illumina", False, True),)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, dataset in enumerate(train_datasets):
                self._create_dataset_cache(root, dataset, read_count=8, offset=index)
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
                    batch_reads=2,
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


if __name__ == "__main__":
    unittest.main()
