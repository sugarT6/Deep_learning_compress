"""Lazy imports and module execution for the previous flat codec layout.

Aliases resolve to the same module objects as their canonical replacements,
so historical imports, pickles and mock.patch targets continue to work.
Only the explicitly listed codec names are intercepted.
"""

from importlib import abc, import_module, util
from pathlib import Path
import sys

_MODULE_TARGETS = {
    "_native_range": "codec.coding._native_range",
    "_range_common": "codec.coding._range_common",
    "ablate_adaptation": "codec.experiments.ablate_adaptation",
    "adaptation_diagnostic": "codec.diagnostics.adaptation_diagnostic",
    "adaptive_prior": "codec.priors.adaptive_prior",
    "backbone_adapter": "codec.adaptation.backbone_adapter",
    "base_memory": "codec.models.base_memory",
    "benchmark_floor_chunks": "codec.experiments.benchmark_floor_chunks",
    "benchmark_lora": "codec.experiments.benchmark_lora",
    "benchmark_quantizer": "codec.experiments.benchmark_quantizer",
    "benchmark_range_coder": "codec.experiments.benchmark_range_coder",
    "block_adaptation": "codec.adaptation.block_adaptation",
    "block_conditioning": "codec.models.block_conditioning",
    "block_features": "codec.data.block_features",
    "block_training": "codec.training.block_training",
    "build_native": "codec.coding.build_native",
    "checkpoint": "codec.models.checkpoint",
    "compact_adapter": "codec.adaptation.compact_adapter",
    "container": "codec.coding.container",
    "datasets": "codec.data.datasets",
    "decode": "codec.coding.decode",
    "diagnose_adaptation": "codec.diagnostics.diagnose_adaptation",
    "diagnose_code_lengths": "codec.diagnostics.diagnose_code_lengths",
    "encode": "codec.coding.encode",
    "encode_fastpath": "codec.coding.encode_fastpath",
    "evaluate": "codec.diagnostics.evaluate",
    "evaluate_fusion": "codec.diagnostics.evaluate_fusion",
    "fastq_stream": "codec.data.fastq_stream",
    "head_adapter": "codec.adaptation.head_adapter",
    "lora_adapter": "codec.adaptation.lora_adapter",
    "mixed_read_loss": "codec.training.mixed_read_loss",
    "mixed_read_sampling": "codec.training.mixed_read_sampling",
    "mixed_read_training": "codec.training.mixed_read_training",
    "mixture_prior": "codec.priors.mixture_prior",
    "model": "codec.models.transformer",
    "online_prior": "codec.priors.online_prior",
    "prediction_report": "codec.diagnostics.prediction_report",
    "probability_quantization": "codec.coding.probability_quantization",
    "quality_history_features": "codec.models.quality_history_features",
    "quality_sampling": "codec.training.quality_sampling",
    "range_decoder": "codec.coding.range_decoder",
    "range_encoder": "codec.coding.range_encoder",
    "rank_coding": "codec.coding.rank_coding",
    "read_state": "codec.adaptation.read_state",
    "relative_attention": "codec.models.relative_attention",
    "running_delta": "codec.priors.running_delta",
    "suffix_lora_adapter": "codec.adaptation.suffix_lora_adapter",
    "train": "codec.training.trainer",
    "train_adaptation_ab": "codec.experiments.train_adaptation_ab",
    "train_experiments": "codec.experiments.train_experiments",
    "train_feature_ablation": "codec.experiments.train_feature_ablation",
    "train_mixed_reads": "codec.training.train_mixed_reads",
    "train_sampling_ab": "codec.experiments.train_sampling_ab",
    "training_cache": "codec.data.training_cache",
    "training_control": "codec.training.control",
    "training_loss_logging": "codec.training.loss_logging"
}

LEGACY_MODULES = {"codec." + name: target for name, target in _MODULE_TARGETS.items()}

COMMANDS = {
    "encode": "codec.coding.encode",
    "decode": "codec.coding.decode",
    "train": "codec.training.trainer",
    "train-mixed": "codec.training.train_mixed_reads",
    "evaluate": "codec.diagnostics.evaluate",
    "cache": "codec.data.training_cache",
    "build-native": "codec.coding.build_native",
    "compact-adapter": "codec.adaptation.compact_adapter",
    "diagnose-adaptation": "codec.diagnostics.diagnose_adaptation",
    "diagnose-code-lengths": "codec.diagnostics.diagnose_code_lengths",
    "evaluate-fusion": "codec.diagnostics.evaluate_fusion",
    "ablate-adaptation": "codec.experiments.ablate_adaptation",
    "train-experiments": "codec.experiments.train_experiments",
    "train-adaptation-ab": "codec.experiments.train_adaptation_ab",
    "train-sampling-ab": "codec.experiments.train_sampling_ab",
    "benchmark-range-coder": "codec.experiments.benchmark_range_coder",
    "benchmark-quantizer": "codec.experiments.benchmark_quantizer",
    "benchmark-floor-chunks": "codec.experiments.benchmark_floor_chunks",
    "benchmark-lora": "codec.experiments.benchmark_lora",
}
_CLI_MODULES = frozenset(COMMANDS.values()) | {
    # This optional local experiment keeps its historical module invocation,
    # without advertising an untracked module in the installed CLI.
    "codec.experiments.train_feature_ablation",
}


class _LegacyLoader(abc.Loader):
    def __init__(self, fullname, target):
        self.fullname = fullname
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Do not copy a module dictionary: functions must retain the same
        # globals as the module reached through either public import path.
        sys.modules[self.fullname] = import_module(self.target)

    def is_package(self, fullname):
        return False

    def get_filename(self, fullname):
        relative = Path(*self.target.split(".")[1:]).with_suffix(".py")
        return str(Path(__file__).resolve().parents[1] / relative)

    def get_code(self, fullname):
        if self.target not in _CLI_MODULES:
            raise ImportError(
                f"{fullname} is a library module; import {self.target} instead")
        # runpy supplies the original arguments. Importing the implementation
        # preserves its package, resource locations and monkeypatch identity.
        source = (
            "from importlib import import_module as _import_module\n"
            f"raise SystemExit(_import_module({self.target!r}).main())\n"
        )
        return compile(source, self.get_filename(fullname), "exec")


class _LegacyFinder(abc.MetaPathFinder):
    _codec_legacy_imports = True

    def find_spec(self, fullname, path=None, target=None):
        replacement = LEGACY_MODULES.get(fullname)
        if replacement is None:
            return None
        return util.spec_from_loader(fullname, _LegacyLoader(fullname, replacement))


def install_legacy_imports():
    """Register the exact-name compatibility finder once, without imports."""
    if not any(getattr(finder, "_codec_legacy_imports", False)
               for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
