"""The functional package layout retains public imports, patches and commands."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import codec
from codec.utils.legacy_imports import LEGACY_MODULES, install_legacy_imports


PROJECT = Path(__file__).resolve().parents[1]
CODEC = PROJECT / "codec"


class PackageLayoutTest(unittest.TestCase):
    def run_python(self, *arguments):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1")
        result = subprocess.run([sys.executable, *arguments], cwd=PROJECT,
                                env=environment, text=True, capture_output=True,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_legacy_imports_are_the_canonical_module_objects(self):
        optional = {"codec.quality_history_features", "codec.train_feature_ablation"}
        for legacy, canonical in LEGACY_MODULES.items():
            path = PROJECT.joinpath(*canonical.split(".")).with_suffix(".py")
            if legacy in optional and not path.exists():
                continue
            with self.subTest(legacy=legacy):
                new = importlib.import_module(canonical)
                old = importlib.import_module(legacy)
                self.assertIs(old, new)
                self.assertIs(sys.modules[legacy], sys.modules[canonical])
                self.assertIs(getattr(codec, legacy.split(".")[-1]), new)
                self.assertEqual(new.__spec__.name, canonical)

    def test_legacy_patch_changes_the_implementation_used_by_the_new_path(self):
        encoder = importlib.import_module("codec.coding.encode")
        with mock.patch("codec.encode.load_training_checkpoint") as patched:
            self.assertIs(encoder.encode_fastq.__globals__["load_training_checkpoint"],
                          patched)
        model = importlib.import_module("codec.models.transformer")
        self.assertIs(importlib.import_module("codec.model").DirectQualityTransformer,
                      model.DirectQualityTransformer)

    def test_compatibility_finder_is_idempotent_and_does_not_capture_new_names(self):
        install_legacy_imports()
        install_legacy_imports()
        finders = [item for item in sys.meta_path
                   if getattr(item, "_codec_legacy_imports", False)]
        self.assertEqual(len(finders), 1)
        self.assertIsNone(finders[0].find_spec("codec.not_a_legacy_module"))
        self.assertIsNone(finders[0].find_spec("codec.models.transformer"))

    def test_top_level_help_does_not_import_torch_or_flash_attention(self):
        result = self.run_python(
            "-c", "import runpy, sys; sys.argv=['codec', '--help']; "
            "runpy.run_module('codec', run_name='__main__')")
        self.assertIn("train-mixed", result.stdout)
        self.assertIn("build-native", result.stdout)
        self.run_python(
            "-c", "import codec, codec.__main__, sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'flash_attn' not in sys.modules")

    def test_old_canonical_and_unified_cli_accept_help(self):
        cases = (
            ("encode", "codec.coding.encode", "encode"),
            ("decode", "codec.coding.decode", "decode"),
            ("train_mixed_reads", "codec.training.train_mixed_reads", "train-mixed"),
            ("training_cache", "codec.data.training_cache", "cache"),
        )
        for legacy, canonical, command in cases:
            for arguments in (("-m", "codec." + legacy, "--help"),
                              ("-m", canonical, "--help"),
                              ("-m", "codec", command, "--help")):
                with self.subTest(arguments=arguments):
                    result = self.run_python(*arguments)
                    self.assertIn("usage:", result.stdout.lower())
                    self.assertNotIn("RuntimeWarning", result.stderr)

    def test_unified_dispatch_preserves_command_arguments(self):
        from codec.__main__ import main
        encoder = importlib.import_module("codec.coding.encode")
        original = sys.argv
        seen = []
        with mock.patch.object(encoder, "main",
                               side_effect=lambda: seen.append(list(sys.argv)) or 0):
            self.assertEqual(main(["encode", "in.fastq", "out.fqdc", "base.pt"]), 0)
        self.assertEqual(seen[0][1:], ["in.fastq", "out.fqdc", "base.pt"])
        self.assertIs(sys.argv, original)

    def test_moved_modules_keep_existing_data_output_and_native_locations(self):
        from codec.utils import paths
        cache = importlib.import_module("codec.data.training_cache")
        native = importlib.import_module("codec.coding._native_range")
        experiments = importlib.import_module("codec.experiments.train_experiments")
        diagnostic = importlib.import_module("codec.diagnostics.diagnose_adaptation")
        self.assertEqual(paths.PROJECT_ROOT, PROJECT)
        self.assertEqual(paths.CODEC_ROOT, CODEC)
        self.assertEqual(paths.CONFIG_ROOT, CODEC / "configs")
        self.assertEqual(cache.DEFAULT_CACHE_DIR, PROJECT / "data/2nd/training_cache")
        self.assertEqual(experiments.REPOSITORY_ROOT, PROJECT)
        self.assertEqual(diagnostic.REPOSITORY_ROOT, PROJECT)
        self.assertEqual(paths.RUNS_ROOT, CODEC / "runs")
        self.assertEqual(paths.OUTPUT_ROOT, CODEC / "output")
        self.assertEqual(native.native_source_path(), CODEC / "native/range_encoder.cpp")
        self.assertTrue(native.native_source_path().is_file())
        self.assertEqual(native.native_library_path().parent, CODEC / "native/_build")


if __name__ == "__main__":
    unittest.main()
