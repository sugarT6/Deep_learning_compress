"""Validation-driven learning-rate reduction and resumable early stopping."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class TrainingControlConfig:
    lr_scheduler: str = "none"
    lr_factor: float = 0.5
    lr_patience: int = 2
    min_lr: float = 1e-5
    early_stopping_patience: int = 0
    min_delta: float = 1e-4

    def __post_init__(self):
        if self.lr_scheduler not in ("none", "plateau"):
            raise ValueError("lr_scheduler must be none or plateau")
        for name in ("lr_patience", "early_stopping_patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("lr_factor", "min_lr", "min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.lr_factor < 1 or self.min_lr < 0 or self.min_delta < 0:
            raise ValueError("invalid learning-rate factor, floor, or minimum improvement")

    def to_dict(self):
        return asdict(self)


class TrainingControl:
    """Uses only the training-source validation macro BPQ, never test files.

    min_delta controls scheduling/stopping, not best.pt selection: any strict
    improvement still deserves its checkpoint. A legacy resume initializes
    both controls against the old best, with fresh patience counters.
    """

    def __init__(self, optimizer, config, *, best_bits=float("inf")):
        self.config = config
        self.best_bits = best_bits
        self.bad_epochs = 0
        self.scheduler = None
        if config.lr_scheduler == "plateau":
            if any(config.min_lr > group["lr"] for group in optimizer.param_groups):
                raise ValueError("min_lr must not exceed the current learning rate")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=config.lr_factor,
                patience=config.lr_patience, threshold=config.min_delta,
                threshold_mode="abs", min_lr=config.min_lr,
            )
            self.scheduler.best = best_bits

    def step(self, bits_per_quality):
        if not math.isfinite(bits_per_quality):
            raise ValueError("validation metric must be finite")
        if bits_per_quality < self.best_bits - self.config.min_delta:
            self.best_bits = bits_per_quality
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.scheduler is not None:
            self.scheduler.step(bits_per_quality)
        return (self.config.early_stopping_patience > 0
                and self.bad_epochs >= self.config.early_stopping_patience)

    def state_dict(self):
        return {
            "version": 1, "config": self.config.to_dict(),
            "best_bits": self.best_bits, "bad_epochs": self.bad_epochs,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("config") != self.config.to_dict()
                or type(state.get("bad_epochs")) is not int or state["bad_epochs"] < 0
                or not isinstance(state.get("best_bits"), (int, float))
                or not math.isfinite(state["best_bits"])
                or (state.get("scheduler") is None) != (self.scheduler is None)):
            raise ValueError("invalid or mismatched training control state")
        self.best_bits = state["best_bits"]
        self.bad_epochs = state["bad_epochs"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
