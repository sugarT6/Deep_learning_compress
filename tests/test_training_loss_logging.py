"""Check trace preservation, objective distinction and actual plot/event output."""

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from codec.training_loss_logging import TrainingLossLogger, read_jsonl, trailing_mean


def batch(step, epoch=1):
    return {
        "global_step": step, "epoch": epoch, "batch_in_epoch": step,
        "loss_nats": float(step), "pooled_loss_nats": float(step) + .5,
        "learning_rate": .001, "read_count": 4, "quality_symbols": 12,
        "source_read_counts": {"short": 2, "long": 2},
        "source_quality_symbols": {"short": 4, "long": 8},
        "source_loss_nats": {"short": float(step), "long": float(step) + 1},
    }


class TrainingLossLoggingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_raw_coordinates_and_short_initial_smoothing_windows(self):
        np.testing.assert_allclose(trailing_mean([1, 3, 5, 7], 3), [1, 2, 3, 5])
        np.testing.assert_allclose(trailing_mean([1, 3], 200), [1, 2])
        self.assertEqual(len(trailing_mean([], 200)), 0)
        with self.assertRaises(ValueError):
            trailing_mean([1], 0)
        long_trace = np.arange(200000, dtype=np.float64)
        long_smoothed = trailing_mean(long_trace, 200)
        self.assertEqual(long_smoothed.shape, (200000,))
        self.assertEqual(long_smoothed[0], 0.)
        self.assertEqual(long_smoothed[-1], np.mean(long_trace[-200:]))
        with TrainingLossLogger(self.root, smoothing_window=3, flush_every=2,
                                enable_tensorboard=False) as logger:
            for step in (1, 2, 4, 5):
                logger.log_batch(batch(step))
            series = logger.batch_series()
            np.testing.assert_array_equal(series["global_step"], [1, 2, 4, 5])
            np.testing.assert_allclose(series["loss_nats"], [1, 2, 4, 5])
            np.testing.assert_allclose(series["pooled_loss_nats"], [1.5, 2.5, 4.5, 5.5])
            np.testing.assert_allclose(series["smoothed_loss_nats"], [1, 1.5, 7 / 3, 11 / 3])
            self.assertEqual(len(list(read_jsonl(logger.batch_path))), 4)
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                logger.log_batch(batch(5))
        with (self.root / "training_batch_log.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["global_step"]) for row in rows], [1, 2, 4, 5])
        self.assertEqual(json.loads(rows[0]["source_quality_symbols"]), {"long": 8, "short": 4})

    def test_png_and_tensorboard_contain_distinct_complete_traces_without_rng_changes(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        py_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        with TrainingLossLogger(self.root, smoothing_window=2, require_tensorboard=True) as logger:
            logger.log_batch(batch(1))
            logger.log_batch(batch(2))
            epoch = {"epoch": 1, "global_step": 2, "train_loss": 2.,
                     "train_objective_loss": 1.5, "learning_rate": .001}
            # The normal training writer has already appended this epoch.
            (self.root / "training_log.jsonl").write_text(json.dumps(epoch) + "\n")
            logger.log_epoch(epoch)
            self.assertEqual(len(list(read_jsonl(self.root / "training_log.jsonl"))), 1)
        self.assertEqual(random.getstate(), py_rng)
        now = np.random.get_state()
        self.assertEqual(now[0], numpy_rng[0])
        np.testing.assert_array_equal(now[1], numpy_rng[1])
        self.assertEqual(now[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for name in ("loss_batch.png", "loss_epoch.png"):
            data = (self.root / name).read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(len(data), 1000)
        events = EventAccumulator(str(self.root / "tensorboard"), size_guidance={"scalars": 0})
        events.Reload()
        objective = events.Scalars("batch/source_weighted_loss_nats")
        pooled = events.Scalars("batch/pooled_loss_nats")
        self.assertEqual([(event.step, event.value) for event in objective], [(1, 1.), (2, 2.)])
        self.assertEqual([(event.step, event.value) for event in pooled], [(1, 1.5), (2, 2.5)])
        self.assertEqual(events.Scalars("epoch/source_weighted_loss_nats")[0].value, 1.5)
        self.assertEqual(events.Scalars("epoch/pooled_loss_nats")[0].value, 2.)

    def test_resume_import_excludes_updates_after_checkpoint_and_can_redraw_existing_logs(self):
        source = self.root / "source"
        with TrainingLossLogger(source, enable_tensorboard=False) as logger:
            for step in (1, 2, 3):
                logger.log_batch(batch(step, epoch=1 if step <= 2 else 2))
            logger.log_epoch({"epoch": 1, "global_step": 2, "train_loss": 2.,
                              "train_objective_loss": 1.5})
            logger.log_epoch({"epoch": 2, "global_step": 3, "train_loss": 3.5,
                              "train_objective_loss": 3.})
        destination = self.root / "resumed"
        with TrainingLossLogger(destination, enable_tensorboard=False) as logger:
            logger.import_history(source, max_global_step=2)
            np.testing.assert_array_equal(logger.batch_series()["global_step"], [1, 2])
            self.assertEqual([record["epoch"] for record in read_jsonl(logger.epoch_path)], [1])
            logger.log_batch(batch(3, epoch=2))
        with TrainingLossLogger(destination, enable_tensorboard=False) as logger:
            np.testing.assert_array_equal(logger.batch_series()["global_step"], [1, 2, 3])
            logger.redraw()
        self.assertEqual(len(list(read_jsonl(destination / "training_batch_log.jsonl"))), 3)

    def test_inconsistent_counts_and_nonfinite_losses_rejected_before_writing(self):
        with TrainingLossLogger(self.root, enable_tensorboard=False) as logger:
            record = batch(1)
            record["quality_symbols"] = 13
            with self.assertRaisesRegex(ValueError, "do not sum"):
                logger.log_batch(record)
            record = batch(1)
            record["loss_nats"] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                logger.log_batch(record)
            self.assertEqual(len(logger.batch_series()["global_step"]), 0)

    def test_import_replays_all_epoch_tensorboard_scalars_without_duplicate_jsonl(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        source = self.root / "source"
        source.mkdir()
        epochs = [
            {"epoch": index, "global_step": index, "train_loss": float(index) + .5,
             "train_objective_loss": float(index), "learning_rate": .001 / index,
             "validation": {"dataset_macro_bits_per_quality": float(index) + .25}}
            for index in (1, 2, 3)
        ]
        (source / "training_batch_log.jsonl").write_text(
            "".join(json.dumps(batch(index, epoch=index)) + "\n" for index in (1, 2, 3)))
        (source / "training_log.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in epochs))
        destination = self.root / "resumed"
        with TrainingLossLogger(destination, require_tensorboard=True) as logger:
            logger.import_history(source, max_global_step=2)
            logger.log_batch(batch(3, epoch=3))
            logger.log_epoch(epochs[2])
        records = list(read_jsonl(destination / "training_log.jsonl"))
        self.assertEqual([record["epoch"] for record in records], [1, 2, 3])
        self.assertEqual(records, epochs)
        events = EventAccumulator(str(destination / "tensorboard"), size_guidance={"scalars": 0})
        events.Reload()
        for tag, key in (("epoch/pooled_loss_nats", "train_loss"),
                         ("epoch/source_weighted_loss_nats", "train_objective_loss"),
                         ("epoch/learning_rate", "learning_rate")):
            values = events.Scalars(tag)
            self.assertEqual([value.step for value in values], [1, 2, 3])
            np.testing.assert_allclose([value.value for value in values],
                                       [record[key] for record in epochs], rtol=1e-6)
        validation = events.Scalars("epoch/validation_macro_bits_per_quality")
        self.assertEqual([(value.step, value.value) for value in validation],
                         [(1, 1.25), (2, 2.25), (3, 3.25)])


if __name__ == "__main__":
    unittest.main()
