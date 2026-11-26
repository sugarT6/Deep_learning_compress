"""Stable resource locations, independent of individual module depth."""

from pathlib import Path

CODEC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODEC_ROOT.parent
CONFIG_ROOT = CODEC_ROOT / "configs"
NATIVE_ROOT = CODEC_ROOT / "native"
RUNS_ROOT = CODEC_ROOT / "runs"
OUTPUT_ROOT = CODEC_ROOT / "output"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "2nd" / "training_cache"
