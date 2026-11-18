"""Real three-rank CPU regression for the large Transformer training path.

The tiny cache intentionally ends in a two-read global batch. With three ranks
and microbatches of two reads, this exercises both unequal accumulation counts
and a rank without real examples. No CUDA device is visible to these workers.
"""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from codec.checkpoint import load_training_checkpoint as load_checkpoint
from codec.datasets import DirectQualityDataset
from codec.training_cache import create_training_cache, TrainingCacheReader
from codec.model import DirectQualityModelConfig, fastq_batch_to_tensors, masked_cross_entropy
from codec.large_transformer import LargeQualityTransformer
from codec.train import BalancedTrainingSampler
from codec.training_control import TrainingControlConfig


def _single_device_reference(root):
    """Independent ordinary full-batch optimizer updates, with no DDP helper."""
    args = _arguments(root)
    torch.manual_seed(args["seed"])
    model = LargeQualityTransformer(args["model_config"], attention_backend="math")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"],
                                  weight_decay=args["weight_decay"])
    total_symbols, total_nats = 0, 0.
    with BalancedTrainingSampler(args["train_datasets"], args["cache_dir"],
            train_fraction=args["train_fraction"], batch_reads=args["batch_reads"],
            seed=args["seed"], steps_per_epoch=args["steps_per_epoch"],
            family_weights=None) as sampler:
        model.train()
        for _ in range(args["steps_per_epoch"]):
            tensors = fastq_batch_to_tensors(sampler.sample_batch().batch, torch.device("cpu"))
            optimizer.zero_grad(set_to_none=True)
            loss = masked_cross_entropy(model(**tensors), tensors["qualities"], tensors["active_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            symbols = int(tensors["active_mask"].sum())
            total_symbols += symbols
            total_nats += float(loss.detach()) * symbols
        return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "parameter_names": [name for name, _ in model.named_parameters()],
                "sampler_state": sampler.state_dict(), "global_step": args["steps_per_epoch"],
                "training_state": {"training_quality_tokens": total_symbols,
                    "training_tokens_by_dataset": {"DDP_TINY": total_symbols}},
                "train_loss": total_nats / total_symbols}


def _arguments(root, *, dropout=0.):
    config = DirectQualityModelConfig(d_model=8, num_heads=2, num_layers=2,
        feedforward_dim=16, context_length=3, dropout=dropout)
    return dict(model_config=config, cache_dir=root,
        init_mode="random", epochs=1, steps_per_epoch=2, batch_reads=7,
        train_fraction=.75, validation_max_reads_per_file=3,
        train_datasets=(DirectQualityDataset("DDP_TINY", "Illumina", True),),
        unseen_datasets=(), family_weights=None, device=torch.device("cpu"),
        seed=117, progress=False, learning_rate=.001, weight_decay=.01,
        grad_clip=1., control_config=TrainingControlConfig(lr_scheduler="plateau",
            lr_patience=0, min_delta=1000., min_lr=1e-5))


def _distributed_worker(rank, world_size, root_text, scenario):
    """Runs in a fresh interpreter; failures propagate through mp.spawn."""
    root = Path(root_text)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
        init_method=f"file://{root / (scenario + '.rendezvous')}",
        timeout=timedelta(seconds=60))
    try:
        import codec.large_transformer_training as distributed

        # Guard the actual artifact boundaries, not each rank's test receipts.
        calls = {"save_checkpoint": 0, "write_json": 0}
        for name in tuple(calls):
            original = getattr(distributed, name)

            def guarded(*args, _name=name, _original=original, **kwargs):
                if rank != 0:
                    raise AssertionError(f"rank {rank} attempted {_name}")
                calls[_name] += 1
                return _original(*args, **kwargs)

            setattr(distributed, name, guarded)

        kwargs = {**_arguments(root, dropout=.2 if scenario == "resume" else 0.),
            "precision": "float32", "activation_checkpointing": True, "attention_backend": "math",
            "warmup_steps": 3 if scenario == "resume" else 0,
            "validation_microbatch_reads": 2,
            "microbatch_reads": 2}
        if scenario == "weighted":
            distributed.run_distributed_training(output_dir=root / "ddp", **kwargs)
        elif scenario == "resume":
            distributed.run_distributed_training(output_dir=root / "full",
                **{**kwargs, "epochs": 2})
            distributed.run_distributed_training(output_dir=root / "first", **kwargs)
            distributed.run_distributed_training(output_dir=root / "resumed",
                resume=root / "first" / "last.pt", **{**kwargs, "epochs": 2})
            rejected = root / "wrong_contract"
            try:
                distributed.run_distributed_training(output_dir=rejected,
                    resume=root / "first" / "last.pt",
                    **{**kwargs, "epochs": 2, "learning_rate": .0005})
            except ValueError as exc:
                if "configuration differs" not in str(exc):
                    raise
            else:
                raise AssertionError("changed learning rate was accepted on resume")
            if rejected.exists():
                raise AssertionError("rejected resume created output artifacts")
            cpu_bf16_output = root / "cpu_bf16"
            try:
                distributed.run_distributed_training(output_dir=cpu_bf16_output,
                    **{**kwargs, "precision": "bf16"})
            except ValueError as exc:
                if "CUDA" not in str(exc) and "cuda" not in str(exc):
                    raise
            else:
                raise AssertionError("bf16 CUDA training was accepted on CPU")
            if cpu_bf16_output.exists():
                raise AssertionError("rejected CPU bf16 run created output artifacts")
        else:
            raise AssertionError(f"unknown test scenario: {scenario}")
        dist.barrier()
        (root / f"{scenario}.rank{rank}.json").write_text(json.dumps(calls))
    finally:
        dist.destroy_process_group()


class DistributedTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = _arguments(self.root)["train_datasets"][0]
        records = []
        # The training prefix has 34 Q, split into global batches of 7 and 2
        # reads. Rank losses consequently must not be averaged equally.
        lengths = (1, 6, 2, 5, 3, 4, 6, 2, 5, 4, 1, 6)
        for index, length in enumerate(lengths):
            bases = (b"ACGTNACGTNACGTNA")[:length]
            qualities = bytes(33 + (index * 7 + cycle * 3) % 42
                              for cycle in range(length))
            records.append(b"@r" + str(index).encode() + b"\n" + bases +
                           b"\n+\n" + qualities + b"\n")
        fastq = self.root / "tiny.fastq"
        fastq.write_bytes(b"".join(records))
        create_training_cache(fastq, dataset.cache_path(self.root))

    def _launch(self, scenario):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            GLOO_SOCKET_IFNAME="lo")
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--worker", str(self.root), scenario], cwd=PROJECT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        receipts = [json.loads((self.root / f"{scenario}.rank{rank}.json").read_text())
                    for rank in range(3)]
        self.assertGreater(receipts[0]["save_checkpoint"], 0)
        self.assertGreater(receipts[0]["write_json"], 0)
        self.assertEqual(receipts[1:], [dict.fromkeys(receipts[0], 0)] * 2)

    def test_unequal_microbatches_and_empty_rank_match_global_q_reference(self):
        reference = _single_device_reference(self.root)
        self._launch("weighted")
        actual = load_checkpoint(self.root / "ddp" / "last.pt").payload
        self.assertEqual(actual["global_step"], 2)
        self.assertEqual(reference["sampler_state"], actual["sampler_state"])
        for field in ("training_quality_tokens", "training_tokens_by_dataset"):
            self.assertEqual(reference["training_state"][field],
                             actual["training_state"][field])
        self.assertEqual(actual["training_state"]["training_quality_tokens"], 34)
        for name, value in reference["model_state_dict"].items():
            obtained = actual["model_state_dict"][name]
            if name.endswith("self_attn.in_proj_bias"):
                dimension = value.numel() // 3
                # Adding the same K bias to every allowed key adds one
                # constant to each softmax row, so its exact gradient is zero.
                # FP32 reduction leaves ~1e-10 gradients; Adam eps=1e-8 can
                # amplify that null direction. Q/V biases are still compared
                # with the ordinary tolerance, as are all Adam moments below.
                torch.testing.assert_close(obtained[dimension:2 * dimension],
                    value[dimension:2 * dimension], rtol=0, atol=1e-4, msg=name)
                obtained = torch.cat((obtained[:dimension], obtained[2 * dimension:]))
                value = torch.cat((value[:dimension], value[2 * dimension:]))
            torch.testing.assert_close(obtained, value, rtol=3e-4, atol=3e-6,
                                       msg=f"parameter {name}")
        # Adam's moments expose a wrong gradient scale even if the first Adam
        # parameter update hides it through the adaptive denominator.
        for key, state in reference["optimizer_state_dict"]["state"].items():
            for name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual["optimizer_state_dict"]["state"][key][name], state[name],
                    rtol=1e-4, atol=1e-8, msg=f"optimizer {key}.{name}")
            if reference["parameter_names"][key].endswith("self_attn.in_proj_bias"):
                dimension = state["exp_avg"].numel() // 3
                for moments in (state, actual["optimizer_state_dict"]["state"][key]):
                    self.assertLess(float(moments["exp_avg"][dimension:2 * dimension].abs().max()), 1e-8)
        # The tolerated K-bias null direction must not change predictions or CE.
        config = _arguments(self.root)["model_config"]
        models = [LargeQualityTransformer(config, attention_backend="math").eval() for _ in range(2)]
        for model, payload in zip(models, (reference, actual)):
            model.load_state_dict(payload["model_state_dict"], strict=True)
        dataset = _arguments(self.root)["train_datasets"][0]
        with TrainingCacheReader(dataset.cache_path(self.root)) as reader:
            tensors = fastq_batch_to_tensors(reader.read_range(0, 12), torch.device("cpu"))
        with torch.inference_mode():
            logits = [model(**tensors) for model in models]
            torch.testing.assert_close(logits[0], logits[1], rtol=1e-5, atol=1e-6)
            losses = [masked_cross_entropy(value, tensors["qualities"], tensors["active_mask"])
                      for value in logits]
            torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-6)
        log = [json.loads(line) for line in
               (self.root / "ddp" / "training_log.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["epoch_training_quality_tokens"], 34)
        self.assertAlmostEqual(log[0]["train_loss"], reference["train_loss"], places=6)
        self.assertEqual(actual["model_config"]["architecture"], "direct-quality-causal-transformer")
        self.assertTrue(all(value.dtype == torch.float32 for value in actual["model_state_dict"].values()))

    def test_dropout_resume_restores_each_rank_and_rejects_changed_contract(self):
        self._launch("resume")
        full = load_checkpoint(self.root / "full" / "last.pt").payload
        resumed = load_checkpoint(self.root / "resumed" / "last.pt").payload
        for field in ("global_step", "sampler_state", "validation_metrics"):
            self.assertEqual(full[field], resumed[field])
        for name, tensor in full["model_state_dict"].items():
            self.assertTrue(torch.equal(tensor, resumed["model_state_dict"][name]), name)
        for key, state in full["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                actual = resumed["optimizer_state_dict"]["state"][key][name]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, actual), (key, name))
                else:
                    self.assertEqual(value, actual)
        self.assertEqual(len(full["training_state"]["rank_rng_states"]), 3)
        full_rng = full["training_state"]["rank_rng_states"]
        resumed_rng = resumed["training_state"]["rank_rng_states"]
        for expected, actual in zip(full_rng, resumed_rng):
            self.assertTrue(torch.equal(expected["torch_rng"], actual["torch_rng"]))
            self.assertEqual(expected["python_rng"], actual["python_rng"])
        self.assertFalse(torch.equal(full_rng[0]["torch_rng"], full_rng[1]["torch_rng"]))
        for field in ("training_quality_tokens", "training_tokens_by_dataset", "control_state"):
            self.assertEqual(full["training_state"][field], resumed["training_state"][field])
        summary = json.loads((self.root / "resumed" / "training_summary.json").read_text())
        self.assertEqual(summary["start_epoch"], 2)
        self.assertEqual(summary["completed_epoch"], 2)
        self.assertEqual(full["training_state"]["control_state"]["scheduler"],
                         resumed["training_state"]["control_state"]["scheduler"])
        self.assertLess(summary["final_learning_rate"], .001)
        self.assertTrue((self.root / "resumed" / "best.pt").is_file())
        self.assertFalse((self.root / "wrong_contract").exists())
        self.assertFalse((self.root / "cpu_bf16").exists())


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        mp.spawn(_distributed_worker, args=(3, sys.argv[2], sys.argv[3]),
                 nprocs=3, join=True)
    else:
        unittest.main()
