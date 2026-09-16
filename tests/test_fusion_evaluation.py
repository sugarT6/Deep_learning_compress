import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from codec.checkpoint import LoadedCheckpoint
from codec.datasets import TRAIN_DATASETS
from codec.evaluate_fusion import evaluate_fusion
from codec.model import DirectQualityTransformer
from codec.training_cache import create_training_cache, TrainingCacheReader
from test_codec_training import tiny_config, small_fastq


class FusionEvaluationTest(unittest.TestCase):
    def test_fixed_ten_validation_only_one_forward_per_scored_batch(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "mock.pt"
                checkpoint.write_bytes(b"mock checkpoint hash")
                entries = {}
                for dataset in TRAIN_DATASETS:
                    source = root / f"{dataset.accession}.fq"
                    source.write_bytes(small_fastq(64))
                    create_training_cache(source, dataset.cache_path(root))
                    with TrainingCacheReader(dataset.cache_path(root)) as reader:
                        entries[dataset.accession] = {"train_range": [0, 32], "validation_range": [32, 64],
                            "read_count": 64, "source_sha256": reader.metadata.source_sha256}
                loaded = LoadedCheckpoint(DirectQualityTransformer(tiny_config()),
                    {"data_split": {"datasets": entries}})
                with mock.patch("codec.evaluate_fusion.load_training_checkpoint", return_value=loaded), \
                     mock.patch.object(loaded.model, "forward_full", wraps=loaded.model.forward_full) as forward, \
                     contextlib.redirect_stdout(io.StringIO()):
                    report = evaluate_fusion(checkpoint, root, batch_reads=8, warmup_reads=8, score_reads=8)
                self.assertEqual(forward.call_count, 10)
                self.assertEqual(report["model_forward_calls"], 10)
                self.assertEqual(set(report["ranges"]), {d.accession for d in TRAIN_DATASETS})
                for entry in report["ranges"].values():
                    self.assertEqual(entry["warmup"], [32, 40])
                    self.assertEqual(entry["score"], [40, 48])
                for name, summary in report["summaries"].items():
                    per_file = report["per_dataset"][name]
                    self.assertEqual(summary["dataset_count"], 10)
                    self.assertAlmostEqual(summary["dataset_macro_bits_per_quality"],
                        sum(item["bits_per_quality"] for item in per_file) / 10)
                    for item in per_file:
                        histogram = report["breakdown"][name][item["accession"]]["q"]
                        self.assertEqual(sum(r["symbols"] for r in histogram.values()), item["symbols"])
                        self.assertAlmostEqual(sum(r["bits"] for r in histogram.values()), item["total_bits"])
                eligible = report["candidate_profiles"]
                winner = min(eligible, key=lambda name: report["summaries"][name]["dataset_macro_bits_per_quality"])
                self.assertEqual(report["best_deployable_candidate"], winner)
                self.assertEqual(report["selected_profile"], eligible[winner])
                with mock.patch("codec.evaluate_fusion.load_training_checkpoint", return_value=loaded):
                    with self.assertRaises(ValueError):
                        evaluate_fusion(checkpoint, root, batch_reads=8, warmup_reads=7, score_reads=8)
                    with self.assertRaises(ValueError):
                        evaluate_fusion(checkpoint, root, batch_reads=8, warmup_reads=8, score_reads=64)
        finally:
            torch.set_num_threads(old_threads)
