"""Check split isolation, matched loss tokens and exact continuation recovery."""

import contextlib
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from codec.checkpoint import load_training_checkpoint, save_training_checkpoint
from codec.datasets import DirectQualityDataset
from codec.fastq_stream import make_fastq_batch
from codec.model import DirectQualityModelConfig, DirectQualityTransformer, fastq_batch_to_tensors, masked_cross_entropy
from codec.quality_sampling import (QualityMixtureSampler, cluster_profiles, load_profile,
    mixture_probabilities, prepare_profile, profile_ranges, summarize_quality_batches)
import codec.train_sampling_ab as training
from codec.training_cache import TrainingCacheReader, create_training_cache


def fixture(root, mode_fraction=.5):
    datasets = (DirectQualityDataset("A", "MGI/BGI", True),
                DirectQualityDataset("B", "Illumina", True),
                DirectQualityDataset("C", "Illumina", True))
    families = {"MGI/BGI": 1, "Illumina": 3}
    cache = root / "cache"
    cache.mkdir()
    for source_index, dataset in enumerate(datasets):
        rows = []
        for index in range(24):
            length = 1 + (index + source_index) % (3 + source_index)
            quality = bytes(33 + (source_index * 15 + t + index % 3) % 42 for t in range(length))
            rows.append(b"@r\n" + b"A" * length + b"\n+\n" + quality + b"\n")
        path = root / (dataset.accession + ".fq")
        path.write_bytes(b"".join(rows))
        create_training_cache(path, dataset.cache_path(cache))
    profile_path = root / "profile.json"
    profile = prepare_profile(profile_path, datasets=datasets, cache_dir=cache,
        family_weights=families, train_fraction=.75, profile_reads=12, strata=4,
        clusters=2, mode_fraction=mode_fraction)
    torch.manual_seed(17)
    model = DirectQualityTransformer(DirectQualityModelConfig(
        num_layers=4, d_model=8, num_heads=2, feedforward_dim=16, context_length=16,
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3, base_conv_channels=3,
        base_context_dim=4, dropout=.1))
    checkpoint = root / "public.pt"
    save_training_checkpoint(checkpoint, model=model, optimizer=None, epoch=1,
        global_step=200000, data_split={}, sampler_config={}, sampler_statistics={},
        best_validation_bits_per_quality=1., validation_metrics={})
    config = training.SamplingConfig(updates=4, quality_tokens_per_update=11, eval_every=2,
        batch_reads=4, schedule_batches=8, validation_reads_per_file=3, train_fraction=.75,
        learning_rate=.0003, min_learning_rate=.00003, seed=19)
    return datasets, families, cache, profile_path, profile, checkpoint, config


def sampler_for(data, mode="quality"):
    datasets, _, cache, _, profile, _, config = data
    return QualityMixtureSampler(datasets, cache, train_fraction=config.train_fraction,
        batch_reads=config.batch_reads, seed=config.seed, steps_per_epoch=config.schedule_batches,
        probabilities=profile[mode + "_probabilities"], profile_sha256=profile["sha256"])


def run(data, mode, output, resume=None, config=None):
    datasets, families, cache, path, _, checkpoint, original = data
    return training.run_sampling(mode, output_dir=output, cache_dir=cache, profile_path=path,
        init_from=checkpoint if resume is None else None, resume=resume,
        config=config or original, datasets=datasets, unseen_datasets=(),
        family_weights=families, progress=False)


class QualitySamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_profile_ranges_cover_exact_quota_without_crossing_split(self):
        for stop in range(1, 27):
            for requested in range(1, 30):
                for strata in (1, 3, 6, 16):
                    ranges = profile_ranges(stop, requested, strata)
                    indices = [i for begin, end in ranges for i in range(begin, end)]
                    self.assertEqual(len(indices), min(stop, requested))
                    self.assertEqual(len(set(indices)), len(indices))
                    self.assertTrue(all(0 <= i < stop for i in indices))
                    self.assertTrue(all(0 < end - begin <= 256 for begin, end in ranges))

    def test_quality_statistics_exclude_padding_read_boundaries_and_uint8_wrap(self):
        batch = make_fastq_batch([np.zeros(3, np.uint8), np.zeros(1, np.uint8)],
            [np.array([40, 0, 1], np.uint8), np.array([37], np.uint8)], [0, 1], source_name="fixture")
        result = summarize_quality_batches([batch])
        self.assertEqual(result["quality_symbols"], 4)
        self.assertEqual(result["change_counts"], [0, 1, 0, 0, 0, 1])
        self.assertEqual(sum(result["histogram"]), 4)
        self.assertEqual(sum(result["cycle_counts"]), 4)
        self.assertEqual(len(result["features"]), 68)
        self.assertTrue(np.isfinite(result["features"]).all())

    def test_profile_is_training_only_and_ignores_unlisted_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            datasets, families, cache, _, _, _, _ = data
            (cache / "CNR0847462.direct_quality.h5").write_bytes(b"must not be opened")
            observed = []
            original = TrainingCacheReader.read_range
            def read(reader, start, stop):
                observed.append((reader.cache_path.name, start, stop))
                return original(reader, start, stop)
            with mock.patch.object(TrainingCacheReader, "read_range", read):
                result = prepare_profile(Path(directory) / "second.json", datasets=datasets,
                    cache_dir=cache, family_weights=families, train_fraction=.75,
                    profile_reads=12, strata=4, clusters=2)
            self.assertEqual(len(observed), 12)
            self.assertTrue(all(stop <= 18 for _, _, stop in observed))
            self.assertEqual(set(result["profiles"]), {"A", "B", "C"})
            with self.assertRaisesRegex(ValueError, "training sources only"):
                prepare_profile(Path(directory) / "bad.json", datasets=(DirectQualityDataset("TEST", "MGI/BGI", False),),
                    cache_dir=cache, family_weights={"MGI/BGI": 1})

    def test_clustering_is_order_invariant_and_handles_identical_profiles(self):
        profiles = {"C": {"features": [9., 9.]}, "A": {"features": [0., 0.]},
                    "B": {"features": [.1, 0.]}}
        groups = cluster_profiles(profiles, 2)
        self.assertEqual(groups, cluster_profiles(dict(reversed(list(profiles.items()))), 2))
        self.assertEqual(sorted(name for group in groups for name in group), ["A", "B", "C"])
        same = {name: {"features": [1., 1.]} for name in profiles}
        self.assertEqual(cluster_profiles(same, 4), [["A", "B", "C"]])

    def test_mixture_has_declared_half_original_half_mode_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = fixture(Path(directory))[4]
            self.assertEqual(profile["original_probabilities"], {"A": .25, "B": .375, "C": .375})
            for name in profile["profiles"]:
                self.assertAlmostEqual(profile["quality_probabilities"][name],
                    .5 * profile["original_probabilities"][name] + .5 * profile["mode_balanced_probabilities"][name])
            self.assertAlmostEqual(sum(profile["quality_probabilities"].values()), 1.)

    def test_source_quotas_and_sampler_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            with sampler_for(data) as sampler:
                for _ in range(240):
                    sample = sampler.sample_batch()
                    self.assertLess(int(sample.batch.read_indices.max()), 18)
                stats = sampler.statistics_dict()
                for name, p in data[4]["quality_probabilities"].items():
                    self.assertAlmostEqual(stats["datasets"][name]["proportion"], p, delta=2 / 240)
                state = sampler.state_dict()
                expected = [(x.dataset.accession, x.batch.read_indices.tolist())
                            for x in (sampler.sample_batch() for _ in range(13))]
            with sampler_for(data) as restored:
                restored.load_state_dict(state)
                actual = [(x.dataset.accession, x.batch.read_indices.tolist())
                          for x in (restored.sample_batch() for _ in range(13))]
                self.assertEqual(actual, expected)

    def test_singleton_modes_cannot_boost_a_source_above_twice_control(self):
        original = {"A": .9, "B": .05, "C": .05}
        for fraction in (0., .5, 1.):
            _, balanced, mixed = mixture_probabilities(original, [["A"], ["B"], ["C"]], fraction)
            self.assertAlmostEqual(sum(mixed.values()), 1.)
            self.assertAlmostEqual(sum(balanced.values()), 1.)
            for name, p in mixed.items():
                self.assertLessEqual(p, 2 * original[name] + 1e-12)
                self.assertGreaterEqual(p, (1 - fraction) * original[name])

    def test_zero_mode_fraction_produces_identical_control_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), mode_fraction=0.)
            with sampler_for(data, "original") as a, sampler_for(data, "quality") as b:
                for _ in range(24):
                    left, right = a.sample_batch(), b.sample_batch()
                    self.assertEqual(left.dataset, right.dataset)
                    np.testing.assert_array_equal(left.batch.qualities, right.batch.qualities)
                    np.testing.assert_array_equal(left.batch.read_indices, right.batch.read_indices)

    def test_partial_loss_batch_resume_is_exact_and_keeps_context(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            with sampler_for(data) as sampler:
                stream = training.TokenStream(sampler)
                batch, selected, count = stream.take(1)
                self.assertEqual(count, 1)
                self.assertTrue(batch.active_mask.reshape(-1)[selected].all())
                self.assertGreater(int(batch.active_mask.sum()), count)
                saved_sampler, saved_stream = sampler.state_dict(), stream.state_dict()
                expected = []
                for _ in range(9):
                    part, positions, tokens = stream.take(7)
                    expected.append((part.qualities.copy(), positions.copy(), tokens))
                final = stream.state_dict()
            with sampler_for(data) as sampler:
                sampler.load_state_dict(saved_sampler)
                stream = training.TokenStream(sampler)
                stream.load_state_dict(saved_stream)
                for wanted in expected:
                    part, positions, count = stream.take(7)
                    np.testing.assert_array_equal(part.qualities, wanted[0])
                    np.testing.assert_array_equal(positions, wanted[1])
                    self.assertEqual(count, wanted[2])
                self.assertEqual(stream.state_dict(), final)

    def test_token_update_matches_full_cross_entropy_on_identical_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            with sampler_for(data) as source:
                first = source.sample_batch().batch
            config = replace(data[6], quality_tokens_per_update=int(first.active_mask.sum()), grad_clip=1e6)
            original = load_training_checkpoint(data[5]).model
            original.config = replace(original.config, dropout=0.)
            for module in original.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.dropout = 0.
            candidate = copy.deepcopy(original)
            tensors = fastq_batch_to_tensors(first, torch.device("cpu"))
            original.train()
            loss = masked_cross_entropy(original.forward_full(**tensors), tensors["qualities"], tensors["active_mask"])
            loss.backward()
            optimizer = torch.optim.SGD(candidate.parameters(), lr=0.)
            with sampler_for(data) as sampler:
                stream = training.TokenStream(sampler)
                nats = training.update_on_tokens(candidate, optimizer, stream, config, torch.device("cpu"))
                self.assertEqual(stream.tokens, config.quality_tokens_per_update)
            self.assertAlmostEqual(nats / config.quality_tokens_per_update, loss.item(), places=6)
            for left, right in zip(original.parameters(), candidate.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)

    def test_both_arms_match_update_and_loss_token_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            for mode in ("original", "quality"):
                path = run(data, mode, root / mode)
                saved = load_training_checkpoint(path).payload
                self.assertEqual(saved["global_step"], 4)
                self.assertEqual(saved["training_state"]["stream"]["tokens"], 44)
                self.assertEqual(sum(saved["training_state"]["stream"]["tokens_by_dataset"].values()), 44)
                self.assertEqual(saved["model_config"]["num_layers"], 4)
                self.assertGreater(load_training_checkpoint(path.parent / "best.pt").payload["global_step"], 0)
            result = training.compare_runs(root / "original", root / "quality")
            self.assertTrue(np.isfinite(result["quality_gain_bpq"]))
            left = json.loads((root / "original/initial_validation.json").read_text())
            right = json.loads((root / "quality/initial_validation.json").read_text())
            self.assertEqual(left, right)
            summary_path = root / "quality/training_summary.json"
            incomplete = json.loads(summary_path.read_text())
            incomplete["global_step"] = 2
            summary_path.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError, "complete matched"):
                training.compare_runs(root / "original", root / "quality")

    def test_resume_rejects_changed_mode_budget_and_finished_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            path = run(data, "original", root / "original")
            with self.assertRaisesRegex(ValueError, "contract differs"):
                run(data, "quality", root / "changed_mode", resume=path)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                run(data, "original", root / "changed_budget", resume=path,
                    config=replace(data[6], updates=6))
            with self.assertRaisesRegex(ValueError, "unfinished"):
                run(data, "original", root / "finished", resume=path)

    def test_interrupted_resume_matches_continuous_training_including_dropout(self):
        for mode in ("original", "quality"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = fixture(root)
                continuous = run(data, mode, root / "continuous")
                original_save = training.save_training_checkpoint
                def interrupted(path, **kwargs):
                    original_save(path, **kwargs)
                    if Path(path).name == "last.pt" and kwargs["global_step"] == 2:
                        raise RuntimeError("test interruption")
                with mock.patch.object(training, "save_training_checkpoint", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "test interruption"):
                        run(data, mode, root / "partial")
                resumed = run(data, mode, root / "resumed", resume=root / "partial/last.pt")
                a, b = [load_training_checkpoint(path).payload for path in (continuous, resumed)]
                for name in a["model_state_dict"]:
                    self.assertTrue(torch.equal(a["model_state_dict"][name], b["model_state_dict"][name]), name)
                self.assertEqual(a["training_state"]["stream"], b["training_state"]["stream"])
                self.assertEqual(a["sampler_state"], b["sampler_state"])
                for key, item in a["optimizer_state_dict"]["state"].items():
                    for field, value in item.items():
                        self.assertTrue(torch.equal(value, b["optimizer_state_dict"]["state"][key][field]))

    def test_profile_tampering_or_changed_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            profile_path = data[3]
            original = profile_path.read_text()
            edited = json.loads(original)
            edited["quality_probabilities"]["A"] += .1
            profile_path.write_text(json.dumps(edited))
            with sampler_for(data) as sampler:
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    load_profile(profile_path, sampler)
            profile_path.write_text(original)
            cache = data[0][0].cache_path(data[2])
            import os
            stat = cache.stat()
            os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            with sampler_for(data) as sampler:
                with self.assertRaisesRegex(ValueError, "cache identity mismatch"):
                    load_profile(profile_path, sampler)

    def test_dry_run_does_not_need_cache_gpu_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with contextlib.redirect_stdout(io.StringIO()) as text, mock.patch.object(training, "choose_device", side_effect=AssertionError("GPU touched")):
                self.assertEqual(training.main(["quality", "--dry-run", "--output-root", str(output)]), 0)
            value = json.loads(text.getvalue())
            self.assertEqual(value["quality_token_budget"], 327680000)
            self.assertNotIn("CNR0847462", value["training_accessions"])
            self.assertFalse(output.exists())

    def test_invalid_optimizer_or_budget_configuration_is_rejected(self):
        for kwargs in ({"quality_tokens_per_update": 0}, {"updates": 3, "eval_every": 2},
                       {"learning_rate": float("nan")}, {"batch_reads": 257}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                training.SamplingConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
