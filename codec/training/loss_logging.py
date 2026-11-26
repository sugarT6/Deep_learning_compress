"""Loss traces for mixed-source training, without changing training RNG state.

Batch records contain the optimized source-weighted objective and the separate
pooled quality-symbol mean. Plots retain every recorded update and add an
explicitly labelled trailing mean; only the plots are refreshed each epoch.
"""

from array import array
import csv
import json
import math
from pathlib import Path
import warnings

import numpy as np


BATCH_FIELDS = (
    "global_step", "epoch", "batch_in_epoch", "loss_nats", "pooled_loss_nats",
    "learning_rate", "read_count", "quality_symbols", "source_read_counts",
    "source_quality_symbols", "source_loss_nats",
)
SOURCE_FIELDS = ("source_read_counts", "source_quality_symbols", "source_loss_nats")


def read_jsonl(path):
    """Read complete JSON records, refusing damaged or partially written logs."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{number}") from error


def trailing_mean(values, window):
    """Return one trailing-mean value per input, including initial short windows."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("smoothing window must be a positive integer")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("loss values must be one-dimensional")
    if not len(values):
        return values.copy()
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (prefix[ends] - prefix[starts]) / (ends - starts)


class TrainingLossLogger:
    """Append batch traces and produce batch/epoch PNGs and TensorBoard events.

    ``log_epoch`` also appends an epoch record when the training loop has not
    already written it to ``training_log.jsonl``. An existing record with the
    same epoch and step is reused, so it is safe to call after the normal epoch
    log write. ``import_history`` supports resuming into a fresh output directory.
    """

    def __init__(self, output_dir, smoothing_window=200, flush_every=50,
                 enable_tensorboard=True, require_tensorboard=False):
        if isinstance(flush_every, bool) or not isinstance(flush_every, int) or flush_every < 1:
            raise ValueError("flush_every must be a positive integer")
        trailing_mean([], smoothing_window)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.smoothing_window = smoothing_window
        self.flush_every = flush_every
        self.batch_path = self.output_dir / "training_batch_log.jsonl"
        self.csv_path = self.output_dir / "training_batch_log.csv"
        self.epoch_path = self.output_dir / "training_log.jsonl"
        self._steps = array("q")
        self._objective = array("d")
        self._pooled = array("d")
        self._last_step = 0
        self._pending = 0
        self._closed = False
        self._plots_dirty = False
        self._writer = None
        for record in read_jsonl(self.batch_path):
            record = self._normalize_batch(record)
            self._cache_batch(record)
        self._batch_file = self.batch_path.open("a", buffering=1024 * 1024)
        csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._csv_file = self.csv_path.open("a", newline="", buffering=1024 * 1024)
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=BATCH_FIELDS)
        if not csv_exists:
            self._csv_writer.writeheader()
        if require_tensorboard and not enable_tensorboard:
            raise ValueError("require_tensorboard cannot be used with TensorBoard disabled")
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                if require_tensorboard:
                    self._batch_file.close()
                    self._csv_file.close()
                    raise
                warnings.warn("TensorBoard unavailable; JSONL, CSV and PNG loss logs remain enabled")
            else:
                self._writer = SummaryWriter(str(self.output_dir / "tensorboard"),
                                             max_queue=1000, flush_secs=30)
        metadata = {
            "version": 1,
            "batch_objective": "loss_nats: source-weighted loss used for backward",
            "batch_pooled": "pooled_loss_nats: sum of valid-Q cross entropy / valid Q count",
            "epoch_objective": "train_objective_loss: arithmetic mean of batch objectives",
            "epoch_pooled": "train_loss: valid-Q-weighted mean across the epoch",
            "batch_smoothing": {"kind": "trailing arithmetic mean", "window": smoothing_window,
                                "initial_windows": "all available updates; no raw points removed"},
            "flush_every_batches": flush_every,
            "tensorboard_enabled": self._writer is not None,
            "batch_jsonl": self.batch_path.name, "batch_csv": self.csv_path.name,
            "epoch_jsonl": self.epoch_path.name,
            "batch_plot": "loss_batch.png", "epoch_plot": "loss_epoch.png",
        }
        (self.output_dir / "loss_logging.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _normalize_batch(record):
        missing = set(BATCH_FIELDS) - record.keys()
        if missing:
            raise ValueError(f"Missing batch log fields: {sorted(missing)}")
        result = {}
        for name in ("global_step", "epoch", "batch_in_epoch", "read_count", "quality_symbols"):
            value = record[name]
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            result[name] = int(value)
        for name in ("loss_nats", "pooled_loss_nats", "learning_rate"):
            value = float(record[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            result[name] = value
        for name in SOURCE_FIELDS:
            values = record[name]
            if not isinstance(values, dict):
                raise ValueError(f"{name} must be a source-to-value mapping")
            result[name] = {}
            for source, value in sorted(values.items()):
                if name == "source_loss_nats":
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{name} must contain finite nonnegative losses")
                else:
                    if isinstance(value, bool) or int(value) != value or int(value) < 0:
                        raise ValueError(f"{name} must contain nonnegative integer counts")
                    value = int(value)
                result[name][str(source)] = value
        if sum(result["source_read_counts"].values()) != result["read_count"]:
            raise ValueError("source_read_counts do not sum to read_count")
        if sum(result["source_quality_symbols"].values()) != result["quality_symbols"]:
            raise ValueError("source_quality_symbols do not sum to quality_symbols")
        represented = {source for source, count in result["source_quality_symbols"].items() if count}
        if represented != set(result["source_loss_nats"]):
            raise ValueError("source_loss_nats must identify every source with valid Q symbols")
        return result

    def _cache_batch(self, record):
        if record["global_step"] <= self._last_step:
            raise ValueError("Batch global_step must increase strictly")
        self._steps.append(record["global_step"])
        self._objective.append(record["loss_nats"])
        self._pooled.append(record["pooled_loss_nats"])
        self._last_step = record["global_step"]
        self._plots_dirty = True

    def log_batch(self, record):
        if self._closed:
            raise RuntimeError("Loss logger is closed")
        record = self._normalize_batch(record)
        self._cache_batch(record)
        self._batch_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        csv_record = dict(record)
        for name in SOURCE_FIELDS:
            csv_record[name] = json.dumps(record[name], sort_keys=True, separators=(",", ":"))
        self._csv_writer.writerow(csv_record)
        if self._writer is not None:
            step = record["global_step"]
            for tag, value in (("batch/source_weighted_loss_nats", record["loss_nats"]),
                               ("batch/pooled_loss_nats", record["pooled_loss_nats"]),
                               ("batch/learning_rate", record["learning_rate"]),
                               ("batch/quality_symbols", record["quality_symbols"]),
                               ("batch/read_count", record["read_count"])):
                self._writer.add_scalar(tag, value, step)
            for source, value in record["source_loss_nats"].items():
                self._writer.add_scalar(f"batch/source_loss_nats/{source}", value, step)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def log_epoch(self, record):
        if self._closed:
            raise RuntimeError("Loss logger is closed")
        required = ("epoch", "global_step", "train_loss")
        if any(name not in record for name in required):
            raise ValueError(f"Epoch record requires {required}")
        existing = [item for item in read_jsonl(self.epoch_path) if item["epoch"] == record["epoch"]]
        if existing:
            if len(existing) != 1 or existing[0]["global_step"] != record["global_step"]:
                raise ValueError("Conflicting existing epoch record")
        else:
            with self.epoch_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._log_epoch_scalars(record)
        self._plots_dirty = True
        self.flush()
        self.redraw()

    def _log_epoch_scalars(self, record):
        """Write fresh or imported epoch scalars without appending JSONL again."""
        if self._writer is not None:
            epoch = record["epoch"]
            self._writer.add_scalar("epoch/pooled_loss_nats", record["train_loss"], epoch)
            if "train_objective_loss" in record:
                self._writer.add_scalar("epoch/source_weighted_loss_nats",
                                        record["train_objective_loss"], epoch)
            if "learning_rate" in record:
                self._writer.add_scalar("epoch/learning_rate", record["learning_rate"], epoch)
            validation = record.get("validation", {})
            if "dataset_macro_bits_per_quality" in validation:
                self._writer.add_scalar("epoch/validation_macro_bits_per_quality",
                                        validation["dataset_macro_bits_per_quality"], epoch)

    def batch_series(self):
        """Return all raw coordinates and full-length trailing means for inspection."""
        steps = np.asarray(self._steps, dtype=np.int64)
        objective = np.asarray(self._objective, dtype=np.float64)
        pooled = np.asarray(self._pooled, dtype=np.float64)
        return {
            "global_step": steps.copy(), "loss_nats": objective.copy(),
            "pooled_loss_nats": pooled.copy(),
            "smoothed_loss_nats": trailing_mean(objective, self.smoothing_window),
            "smoothed_pooled_loss_nats": trailing_mean(pooled, self.smoothing_window),
        }

    def redraw(self):
        """Regenerate plots from complete traces; this does not sample or decimate."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        series = self.batch_series()
        if len(series["global_step"]):
            figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            for axis, field, smooth, title in (
                (axes[0], "loss_nats", "smoothed_loss_nats", "Source-weighted training objective"),
                (axes[1], "pooled_loss_nats", "smoothed_pooled_loss_nats", "Pooled quality-symbol loss"),
            ):
                axis.plot(series["global_step"], series[field], color="#cc6699", alpha=.27,
                          linewidth=.45, label="Raw batch loss (all updates)")
                axis.plot(series["global_step"], series[smooth], color="#245b98", linewidth=1.1,
                          label=f"Trailing {self.smoothing_window}-batch mean (initial partial windows)")
                axis.set_title(title)
                axis.set_ylabel("Loss (nats/Q)")
                axis.grid(alpha=.2)
                axis.legend(loc="best", fontsize=8)
            axes[-1].set_xlabel("Optimizer update (global step)")
            figure.suptitle(f"Batch training loss - {len(series['global_step']):,} recorded updates")
            figure.tight_layout()
            self._save_figure(figure, self.output_dir / "loss_batch.png")
            plt.close(figure)
        epochs = list(read_jsonl(self.epoch_path))
        if epochs:
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.plot([record["epoch"] for record in epochs],
                      [record["train_loss"] for record in epochs], marker=".", linewidth=1.2,
                      label="Pooled Q mean (Q-weighted across batches)")
            objective_records = [record for record in epochs if "train_objective_loss" in record]
            if objective_records:
                axis.plot([record["epoch"] for record in objective_records],
                          [record["train_objective_loss"] for record in objective_records],
                          marker=".", linewidth=1.2,
                          label="Source-weighted objective (mean across batches)")
            axis.set_title("Training loss by epoch")
            axis.set_xlabel("Logging epoch")
            axis.set_ylabel("Loss (nats/Q)")
            axis.grid(alpha=.2)
            axis.legend()
            figure.tight_layout()
            self._save_figure(figure, self.output_dir / "loss_epoch.png")
            plt.close(figure)
        self._plots_dirty = False

    @staticmethod
    def _save_figure(figure, path):
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        try:
            figure.savefig(temporary, dpi=150)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def import_history(self, source_dir, max_global_step):
        """Import completed history up to a saved checkpoint into this fresh run."""
        if self._last_step or any(read_jsonl(self.epoch_path)):
            raise ValueError("History import requires empty batch and epoch traces")
        source_dir = Path(source_dir)
        for record in read_jsonl(source_dir / self.batch_path.name):
            if record["global_step"] > max_global_step:
                break
            self.log_batch(record)
            if record["global_step"] == max_global_step:
                # A killed run may have a damaged final line after the saved
                # checkpoint; do not inspect uncommitted updates past it.
                break
        epochs = [record for record in read_jsonl(source_dir / self.epoch_path.name)
                  if record["global_step"] <= max_global_step]
        with self.epoch_path.open("a") as handle:
            for record in epochs:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                self._log_epoch_scalars(record)
        self.flush()
        self._plots_dirty = bool(self._last_step or epochs)
        if self._plots_dirty:
            self.redraw()

    def flush(self):
        if self._closed:
            return
        self._batch_file.flush()
        self._csv_file.flush()
        if self._writer is not None:
            self._writer.flush()
        self._pending = 0

    def close(self):
        if not self._closed:
            try:
                self.flush()
                if self._plots_dirty:
                    self.redraw()
            finally:
                self._batch_file.close()
                self._csv_file.close()
                if self._writer is not None:
                    self._writer.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
