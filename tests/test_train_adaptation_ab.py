"""Check support/query isolation, FO gradient mapping and matched/resumable A/B runs."""

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

import codec.train_adaptation_ab as ab
from codec.checkpoint import load_training_checkpoint
from codec.datasets import DirectQualityDataset
from codec.train import BalancedTrainingSampler
from codec.training_cache import TrainingCacheReader, create_training_cache
from test_adaptation_diagnostic import tiny_model, write_source


def fixture(root):
    datasets = (DirectQualityDataset("A", "MGI/BGI", True),
                DirectQualityDataset("B", "Illumina", True))
    cache = root / "cache"
    cache.mkdir()
    for dataset in datasets:
        source = write_source(root, name=f"{dataset.accession}.fq", count=96)
        create_training_cache(source, dataset.cache_path(cache))
    model, checkpoint = tiny_model(root)
    config = ab.ABConfig(epochs=2, steps_per_epoch=2, batch_reads=8, inner_batch_reads=2,
                        validation_episodes_per_file=1, learning_rate=.0003, min_learning_rate=.00003,
                        train_fraction=.75, lora_rank=2, residual_dim=16, cross_dim=4, seed=29)
    return model, checkpoint, cache, datasets, config


def run(mode, root, checkpoint, cache, datasets, config, *, resume=None):
    return ab.run_continuation(mode, output_dir=root, cache_dir=cache, config=config,
        init_from=checkpoint if resume is None else None, resume=resume, datasets=datasets,
        unseen_datasets=(), family_weights={"MGI/BGI": 1, "Illumina": 1}, progress=False)


class AdaptationTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_episode_partition_and_validation_never_overlap_training(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, cache, datasets, config = fixture(Path(directory))
            with BalancedTrainingSampler(datasets, cache, train_fraction=.75, batch_reads=8,
                    seed=config.seed, steps_per_epoch=2, family_weights={"MGI/BGI": 1, "Illumina": 1}) as sampler:
                batches, indices = ab.validation_episodes(sampler, config)
                for accession, episodes in indices.items():
                    all_indices = []
                    for episode in episodes:
                        self.assertFalse(set(episode["support"]) & set(episode["query"]))
                        all_indices.extend(episode["support"] + episode["query"])
                    self.assertEqual(len(all_indices), len(set(all_indices)))
                    self.assertTrue(all(i >= sampler.train_stop_for(accession) for i in all_indices))
                sample = sampler.sample_batch().batch
                support, query = ab.split_episode(sample, seed=config.seed, step=1)
                self.assertFalse(set(support.read_indices) & set(query.read_indices))
                self.assertEqual(set(sample.read_indices), set(support.read_indices) | set(query.read_indices))
                self.assertLess(max(sample.read_indices), 72)
                self.assertEqual(len(batches), 2)

    def test_episode_spans_training_file_strata_and_records_actual_exposures(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, cache, datasets, config = fixture(Path(directory))
            with BalancedTrainingSampler(datasets, cache, train_fraction=.75, batch_reads=8,
                    seed=config.seed, steps_per_epoch=2, family_weights={"MGI/BGI": 1, "Illumina": 1}) as sampler:
                sample = sampler.sample_batch()
                first = ab.spread_episode(sample, sampler, config, step=1)
                second = ab.spread_episode(sample, sampler, config, step=1)
                np.testing.assert_array_equal(first.read_indices, second.read_indices)
                self.assertEqual(first.read_count, sample.batch.read_count)
                self.assertEqual(len(np.unique(first.read_indices)), first.read_count)
                self.assertTrue(np.all(first.read_indices < 72))
                np.testing.assert_array_equal(first.read_indices * 8 // 72, np.arange(8))
                self.assertIsNone(ab.sampling_statistics(sampler)["unique_reads"])

    def test_zero_start_adapter_maps_query_gradients_to_plain_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, cache, datasets, config = fixture(Path(directory))
            with TrainingCacheReader(datasets[0].cache_path(cache)) as reader:
                _, query = ab.split_episode(reader.read_range(0, 8), seed=config.seed, step=1)
            plain = copy.deepcopy(model).eval().requires_grad_(True)
            ce, count = ab.batch_loss(plain, query, "cpu")
            ce.backward()
            expected = {name: p.grad.clone() if p.grad is not None else None for name, p in plain.named_parameters()}
            candidate = ab.prepare_adapter(model, config)
            actual_loss, actual_count = ab.query_gradients(model, candidate, query, "cpu")
            self.assertEqual(count, actual_count)
            self.assertAlmostEqual(float(ce), actual_loss, places=7)
            for name, p in model.named_parameters():
                if expected[name] is None:
                    self.assertIsNone(p.grad)
                else:
                    torch.testing.assert_close(p.grad, expected[name], rtol=1e-5, atol=2e-7)
            mapping = ab.shared_parameter_map(model, candidate)
            mapped_ids = {id(p) for p in mapping.values()}
            for p in candidate.parameters():
                self.assertEqual(p.requires_grad, id(p) in mapped_ids)

    def test_support_changes_only_file_variables_then_updates_clean_shared_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, cache, datasets, config = fixture(Path(directory))
            original = {name: p.detach().clone() for name, p in model.state_dict().items()}
            with TrainingCacheReader(datasets[0].cache_path(cache)) as reader:
                batch = reader.read_range(0, 8)
            support, query = ab.split_episode(batch, seed=config.seed, step=1)
            adapted, stats = ab.fit_support(model, support, config, "cpu")
            self.assertEqual(stats["inner_steps"], 2)
            self.assertEqual(stats["inner_backward_symbols"], int(support.active_mask.sum()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            # Inner fitting freezes every shared backbone parameter except norm;
            # fitting never puts a support gradient into the public model.
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            mapping = ab.shared_parameter_map(model, adapted)
            for name, p in mapping.items():
                if not name.startswith(("output_head.", "transformer.norm.")):
                    torch.testing.assert_close(p, original[name], rtol=0, atol=0)
            ab.query_gradients(model, adapted, query, "cpu")
            early = model.transformer.layers[0].self_attn.in_proj_weight.grad
            self.assertIsNotNone(early)
            self.assertGreater(float(early.abs().sum()), 0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
            optimizer.step()
            self.assertTrue(any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()))
            self.assertEqual(set(model.state_dict()), set(original))
            self.assertIs(type(model.output_head), torch.nn.Linear)

    def test_modes_match_data_exposure_validation_and_export_codec_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, checkpoint, cache, datasets, config = fixture(root)
            original_state = {name: value.clone() for name, value in original.state_dict().items()}
            outputs = {}
            for mode in ("ordinary", "adaptation"):
                path = run(mode, root / mode, checkpoint, cache, datasets, config)
                outputs[mode] = load_training_checkpoint(path).payload
                loaded = load_training_checkpoint(path).model
                self.assertEqual(loaded.config, original.config)
                self.assertEqual(set(loaded.state_dict()), set(original_state))
                self.assertTrue(any(not torch.equal(v, original_state[k]) for k, v in loaded.state_dict().items()))
                self.assertEqual(outputs[mode]["global_step"], 4)
                self.assertEqual(outputs[mode]["selection_metric"], ab.SELECTION_METRIC)
                self.assertTrue((root / mode / "best.pt").is_file())
            a = outputs["ordinary"]["training_state"]
            b = outputs["adaptation"]["training_state"]
            self.assertEqual(a["initial_validation"], b["initial_validation"])
            self.assertEqual(a["contract"]["validation_indices_sha256"], b["contract"]["validation_indices_sha256"])
            for key in ("data_quality_tokens", "support_quality_tokens", "query_quality_tokens",
                        "tokens_by_dataset", "data_footprint_sha256"):
                self.assertEqual(a["counters"][key], b["counters"][key])
            self.assertEqual(a["counters"]["inner_steps"], 0)
            self.assertGreater(b["counters"]["inner_steps"], 0)
            self.assertEqual(a["counters"]["data_quality_tokens"], a["counters"]["outer_backward_symbols"])
            self.assertEqual(b["counters"]["data_quality_tokens"],
                             b["counters"]["outer_backward_symbols"] + b["counters"]["inner_backward_symbols"])

    def test_resume_reproduces_uninterrupted_weights_and_data_for_both_modes(self):
        for mode in ("ordinary", "adaptation"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, checkpoint, cache, datasets, config = fixture(root)
                complete = run(mode, root / "complete", checkpoint, cache, datasets, config)
                save = ab.save_training_checkpoint

                def interrupt(path, **kwargs):
                    if Path(path).name == "last.pt" and kwargs["epoch"] == 2:
                        raise RuntimeError("simulated interruption before epoch 2 save")
                    return save(path, **kwargs)

                with mock.patch.object(ab, "save_training_checkpoint", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run(mode, root / "partial", checkpoint, cache, datasets, config)
                resumed = run(mode, root / "resumed", checkpoint, cache, datasets, config,
                              resume=root / "partial/last.pt")
                expected, actual = load_training_checkpoint(complete).payload, load_training_checkpoint(resumed).payload
                for name, value in expected["model_state_dict"].items():
                    torch.testing.assert_close(actual["model_state_dict"][name], value, rtol=0, atol=0)
                for key in ("data_quality_tokens", "inner_steps", "data_footprint_sha256", "tokens_by_dataset"):
                    self.assertEqual(expected["training_state"]["counters"][key], actual["training_state"]["counters"][key])
                with self.assertRaisesRegex(ValueError, "contract"):
                    run(mode, root / "wrong_budget", checkpoint, cache, datasets, replace(config, epochs=3),
                        resume=root / "partial/last.pt")

    def test_validation_does_not_modify_shared_weights_or_leak_query_to_support(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, cache, datasets, config = fixture(Path(directory))
            original = {name: v.clone() for name, v in model.state_dict().items()}
            with BalancedTrainingSampler(datasets, cache, train_fraction=.75, batch_reads=8,
                    seed=config.seed, steps_per_epoch=2, family_weights={"MGI/BGI": 1, "Illumina": 1}) as sampler:
                episodes, _ = ab.validation_episodes(sampler, config)
                with mock.patch.object(ab, "fit_support", wraps=ab.fit_support) as fit:
                    first = ab.evaluate_proxy(model, episodes, config, "cpu")
                for call, pairs in zip(fit.call_args_list, episodes.values()):
                    fitted_support = call.args[1]
                    self.assertFalse(set(fitted_support.read_indices) & set(pairs[0][1].read_indices))
                second = ab.evaluate_proxy(model, episodes, config, "cpu")
                self.assertEqual(first, second)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_cli_dry_run_and_invalid_configuration(self):
        for mode in ("ordinary", "adaptation"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(ab.main([mode, "--dry-run"]), 0)
            description = json.loads(output.getvalue())
            self.assertEqual(description["mode"], mode)
            self.assertEqual(len(description["training_accessions"]), 20)
        for changes in (dict(batch_reads=1), dict(inner_batch_reads=0), dict(learning_rate=float("nan")),
                        dict(min_learning_rate=.01), dict(lora_rank=17), dict(train_fraction=1.0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ab.ABConfig(**changes)


if __name__ == "__main__":
    unittest.main()
