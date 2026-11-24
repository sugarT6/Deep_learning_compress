"""Real-cache checks for read provenance, quotas and exact sampler continuation."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from codec.datasets import DirectQualityDataset, EXPANDED_FAMILY_WEIGHTS
from codec.fastq_stream import BASE_PAD_ID, QUALITY_PAD_ID
from codec.training_cache import create_training_cache
from codec.mixed_read_sampling import MixedReadTrainingSampler


class MixedSamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.datasets = []
        for family, count in (("MGI/BGI", 7), ("Illumina", 11),
                              ("ABI SOLiD", 1), ("Ion Torrent", 1)):
            for _ in range(count):
                source = len(cls.datasets)
                dataset = DirectQualityDataset(f"SOURCE_{source}", family, True)
                cls.datasets.append(dataset)
                records = []
                for read in range(80):
                    length = 1 + read % 7
                    sequence = bytes(b"ACGTNX"[(source + read + c) % 6]
                                     for c in range(length))
                    quality = bytes(33 + (7 * source + read + c) % 42
                                    for c in range(length))
                    records.append(b"@r" + str(read).encode() + b"\n" + sequence
                                   + b"\n+\n" + quality + b"\n")
                fastq = cls.root / f"{dataset.accession}.fastq"
                fastq.write_bytes(b"".join(records))
                create_training_cache(fastq, dataset.cache_path(cls.root))
        cls.datasets = tuple(cls.datasets)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def sampler(self, **changes):
        kwargs = dict(datasets=self.datasets, cache_dir=self.root,
                      train_fraction=.75, batch_reads=256, seed=20261123,
                      steps_per_epoch=3, family_weights=EXPANDED_FAMILY_WEIGHTS)
        kwargs.update(changes)
        result = MixedReadTrainingSampler(**kwargs)
        self.addCleanup(result.close)
        return result

    def assert_samples_equal(self, left, right):
        self.assertEqual(left.source_read_counts, right.source_read_counts)
        for name in ("row_dataset_indices", "row_read_indices"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        for name in ("bases", "qualities", "lengths", "active_mask", "read_indices"):
            np.testing.assert_array_equal(getattr(left.batch, name), getattr(right.batch, name))

    def test_every_batch_has_all_sources_and_read_quotas_keep_old_probabilities(self):
        sampler = self.sampler()
        for _ in range(20):
            sample = sampler.sample_batch()
            self.assertEqual(sample.batch.read_count, 256)
            self.assertEqual(len(set(sample.row_dataset_indices.tolist())), 20)
            self.assertEqual(sum(sample.source_read_counts.values()), 256)
            self.assertTrue(all(count > 0 for count in sample.source_read_counts.values()))
            # Rows are shuffled, rather than retaining source-grouped order.
            self.assertFalse(np.all(sample.row_dataset_indices[1:]
                                    >= sample.row_dataset_indices[:-1]))
        stats = sampler.statistics_dict()
        self.assertEqual(stats["count_unit"], "reads")
        self.assertEqual(stats["total_batches"], 20)
        self.assertEqual(stats["total_reads"], 5120)
        for family, weight in EXPANDED_FAMILY_WEIGHTS.items():
            self.assertEqual(stats["families"][family]["count"], 512 * weight)
        for family in EXPANDED_FAMILY_WEIGHTS:
            counts = [stats["datasets"][d.accession]["reads"]
                      for d in self.datasets if d.platform_family == family]
            self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertAlmostEqual(sampler.source_probabilities["SOURCE_0"], .3 / 7)
        self.assertAlmostEqual(sampler.source_probabilities["SOURCE_7"], .5 / 11)
        self.assertEqual(sampler.source_probabilities["SOURCE_18"], .1)
        self.assertAlmostEqual(sum(sampler.source_probabilities.values()), 1.)

    def test_shuffled_bases_q_lengths_and_source_indices_stay_paired(self):
        sampler = self.sampler()
        for _ in range(3):
            sample = sampler.sample_batch()
            self.assertEqual(sample.row_dataset_indices.dtype, np.int64)
            self.assertEqual(sample.row_read_indices.dtype, np.int64)
            np.testing.assert_array_equal(sample.batch.read_indices, sample.row_read_indices)
            for row, (source, read) in enumerate(zip(sample.row_dataset_indices,
                                                     sample.row_read_indices)):
                self.assertLess(read, 60)  # Training prefix only; validation begins at 60.
                length = 1 + read % 7
                self.assertEqual(sample.batch.lengths[row], length)
                np.testing.assert_array_equal(sample.batch.bases[row, :length],
                                              [(source + read + c) % 6 for c in range(length)])
                np.testing.assert_array_equal(sample.batch.qualities[row, :length],
                                              [(7 * source + read + c) % 42 for c in range(length)])
                self.assertTrue(sample.batch.active_mask[row, :length].all())
                self.assertFalse(sample.batch.active_mask[row, length:].any())
                self.assertTrue((sample.batch.bases[row, length:] == BASE_PAD_ID).all())
                self.assertTrue((sample.batch.qualities[row, length:] == QUALITY_PAD_ID).all())

    def test_read_permutation_finishes_before_repeating_across_epochs(self):
        sampler = self.sampler(datasets=self.datasets[:2], family_weights=None,
                               batch_reads=2)
        streams = [[], []]
        for _ in range(125):
            sample = sampler.sample_batch()
            for source, read in zip(sample.row_dataset_indices, sample.row_read_indices):
                streams[source].append(int(read))
        for reads in streams:
            self.assertEqual(sorted(reads[:60]), list(range(60)))
            self.assertEqual(sorted(reads[60:120]), list(range(60)))
            self.assertNotEqual(reads[:60], reads[60:120])
        self.assertEqual(sampler._epoch, 42)
        self.assertEqual(sampler._schedule_cursor, 2)
        self.assertEqual(sampler._total_samples, 125)

    def test_resume_reproduces_rows_and_read_round_boundaries_exactly(self):
        sampler = self.sampler()
        for _ in range(7):
            sampler.sample_batch()
        state = sampler.state_dict()
        # Serialized state has no 1.8M-read permutations or RAM-value arrays.
        self.assertNotIn("read_orders", state)
        self.assertNotIn("preloaded_values", state)
        future = [sampler.sample_batch() for _ in range(12)]
        resumed = self.sampler(preload_training_values=True)
        resumed.load_state_dict(state)
        for expected in future:
            self.assert_samples_equal(expected, resumed.sample_batch())
        self.assertEqual(sampler.state_dict(), resumed.state_dict())

    def test_numpy_global_rng_neither_changes_nor_controls_sampler(self):
        np.random.seed(123)
        before = np.random.get_state()
        first = self.sampler()
        observed = first.sample_batch()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.random.seed(9876)
        np.random.random(1000)
        second = self.sampler()
        self.assert_samples_equal(observed, second.sample_batch())

    def test_preloaded_training_prefix_is_byte_identical_and_memory_only(self):
        files_before = set(self.root.iterdir())
        disk = self.sampler()
        ram = self.sampler(preload_training_values=True)
        self.assertEqual(set(self.root.iterdir()), files_before)
        expected_bytes = 0
        for accession, (offsets, bases, qualities) in ram._preloaded_values.items():
            self.assertEqual(len(offsets), 61)
            self.assertEqual(len(bases), ram.reader_for(accession).count_symbols(0, 60))
            self.assertEqual(len(qualities), len(bases))
            self.assertFalse(bases.flags.writeable)
            expected_bytes += offsets.nbytes + bases.nbytes + qualities.nbytes
        self.assertEqual(ram.preloaded_training_bytes, expected_bytes)
        self.assertEqual(ram.preload_training_values(), expected_bytes)
        for _ in range(8):
            self.assert_samples_equal(disk.sample_batch(), ram.sample_batch())
        self.assertEqual(disk.state_dict(), ram.state_dict())

    def test_resume_rejects_corrupt_quota_and_cursor_without_changing_live_state(self):
        sampler = self.sampler()
        sampler.sample_batch()
        original = sampler.state_dict()
        for field in ("read_counts", "read_cursors", "source_batch_counts"):
            changed = copy.deepcopy(original)
            changed[field]["SOURCE_0"] += 1
            with self.assertRaises(ValueError):
                sampler.load_state_dict(changed)
            self.assertEqual(sampler.state_dict(), original)
        changed = copy.deepcopy(original)
        changed["schedule_cursor"] += 1
        with self.assertRaisesRegex(ValueError, "epoch/cursor"):
            sampler.load_state_dict(changed)
        changed = copy.deepcopy(original)
        changed["config"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            sampler.load_state_dict(changed)

    def test_small_batch_cannot_claim_to_include_every_source(self):
        with self.assertRaisesRegex(ValueError, "include every source"):
            self.sampler(batch_reads=20)
        with self.assertRaisesRegex(ValueError, "block features"):
            self.sampler(feature_block_reads=256)


if __name__ == "__main__":
    unittest.main()
