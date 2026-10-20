"""Decoder safety and compatibility checks for the architecture experiments."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from codec.checkpoint import load_training_checkpoint, save_training_checkpoint
from codec.decode import decode_fastq
from codec.datasets import DirectQualityDataset
from codec.encode import encode_fastq
from codec.fastq_stream import make_fastq_batch
from codec.model import (
    DirectQualityModelConfig,
    DirectQualityTransformer,
    build_quality_prefix_features,
    fastq_batch_to_tensors,
    feature_schema,
    masked_cross_entropy,
)
from codec.probability_quantization import logits_to_cdfs
from codec.train import run_training
from codec.training_cache import create_training_cache, default_cache_path


def tiny_config(**overrides):
    values = dict(
        prev_q_embed_dim=4, qmer_embed_dim=3, base_embed_dim=4,
        base_conv_channels=4, base_context_dim=8, d_model=16,
        num_heads=4, num_layers=1, feedforward_dim=32,
        context_length=4, dropout=0.0,
    )
    values.update(overrides)
    return DirectQualityModelConfig(**values)


VARIANTS = (
    ("wider_ffn", dict(feedforward_dim=64)),
    ("wider_hidden", dict(d_model=24, num_heads=6, feedforward_dim=48)),
    ("dilated_bases", dict(base_encoder_type="dilated_cnn")),
    ("swiglu", dict(ffn_type="swiglu")),
    ("quality_prefix", dict(quality_prefix_features=True)),
)


def batch_tensors(lengths=(11, 6)):
    bases = [np.arange(n, dtype=np.uint8) % 5 for n in lengths]
    qualities = [(np.arange(n, dtype=np.uint8) * 7 + 3) % 42 for n in lengths]
    batch = make_fastq_batch(bases, qualities, list(range(len(lengths))), source_name="test.fastq")
    return fastq_batch_to_tensors(batch, torch.device("cpu"))


def save_checkpoint(path, model):
    save_training_checkpoint(
        path, model=model, optimizer=None, epoch=1, global_step=1,
        data_split={}, sampler_config={}, sampler_statistics={},
        best_validation_bits_per_quality=2.0, validation_metrics={},
    )


class ModelVariantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(412)

    def test_default_serialization_keeps_the_existing_checkpoint_contract(self):
        # These are the public default config/schema before experiment fields
        # were introduced. Existing checkpoints must retain exactly this shape.
        expected = dict(
            architecture="direct-quality-causal-transformer", architecture_version=1,
            quality_alphabet_size=42, phred_offset=33, qmer_ks=(2, 3, 4),
            prev_q_embed_dim=16, qmer_embed_dim=8, base_embed_dim=16,
            base_conv_kernels=(3, 5, 7), base_conv_channels=16,
            base_context_dim=32, position_length_scale=512.0, d_model=256,
            num_heads=4, num_layers=4, feedforward_dim=512,
            context_length=256, dropout=0.1,
        )
        default = DirectQualityModelConfig()
        self.assertEqual(default.to_dict(), expected)
        self.assertEqual(DirectQualityModelConfig.from_dict(expected), default)
        self.assertEqual(feature_schema(default), feature_schema())
        encoded = json.dumps(feature_schema(), sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(),
                         "d012ae4c291a4ccb2c3a7bcd97b4e6e76547e24491fa984e8ff50837a8768f2e")

    def test_experiment_configuration_is_serialized_and_invalid_values_rejected(self):
        for name, options in VARIANTS:
            with self.subTest(name=name):
                config = tiny_config(**options)
                self.assertEqual(DirectQualityModelConfig.from_dict(config.to_dict()), config)
                for key, value in options.items():
                    self.assertEqual(config.to_dict()[key], value)
        for options in (dict(base_encoder_type="unknown"), dict(ffn_type="relu"),
                        dict(quality_prefix_features=1)):
            with self.subTest(invalid=options), self.assertRaises(ValueError):
                tiny_config(**options)

    def test_legacy_checkpoint_without_experiment_fields_still_loads(self):
        config = tiny_config()
        model = DirectQualityTransformer(config).eval()
        tensors = batch_tensors()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu")
            self.assertNotIn("base_encoder_type", payload["model_config"])
            self.assertNotIn("ffn_type", payload["model_config"])
            self.assertNotIn("quality_prefix_features", payload["model_config"])
            loaded = load_training_checkpoint(path).model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model.forward_full(**tensors),
                                       loaded.forward_full(**tensors), rtol=0, atol=0)

    def test_prefix_statistics_are_shifted_and_have_the_documented_values(self):
        qualities = torch.tensor([[10, 30, 5, 0, 41], [41, 0, 42, 42, 42]])
        active = torch.tensor([[True] * 5, [True, True, False, False, False]])
        actual = build_quality_prefix_features(qualities, active)
        expected = torch.tensor([
            [[0, 0, 0, 0], [10 / 41, 0, 1, 1 / 512],
             [20 / 41, 100 / 41**2, .5, 0],
             [15 / 41, (350 / 3) / 41**2, 2 / 3, 1 / 512],
             [11.25 / 41, 129.6875 / 41**2, .75, 2 / 512]],
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 0], [0, 0, 0, 0]],
        ])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        changed = qualities.clone()
        changed[0, 3:] = torch.tensor([41, 40])
        torch.testing.assert_close(actual[:, :4],
                                   build_quality_prefix_features(changed, active)[:, :4],
                                   rtol=0, atol=0)

    def test_prefix_model_is_causal_and_uses_older_history_than_local_qmers(self):
        tensors = batch_tensors((11,))
        prefix_model = DirectQualityTransformer(
            tiny_config(quality_prefix_features=True, context_length=2)).eval()
        changed = dict(tensors)
        changed["qualities"] = tensors["qualities"].clone()
        changed["qualities"][0, 5:] = 41 - changed["qualities"][0, 5:]
        with torch.no_grad():
            original = prefix_model.forward_full(**tensors)
            future_changed = prefix_model.forward_full(**changed)
        torch.testing.assert_close(original[:, :6], future_changed[:, :6], rtol=0, atol=0)

        changed["qualities"] = tensors["qualities"].clone()
        changed["qualities"][0, 0] = 41
        with torch.no_grad():
            older_changed = prefix_model.forward_full(**changed)
        self.assertFalse(torch.equal(original[:, 8], older_changed[:, 8]))

        # In a one-block model this old value is outside both the Q-mer input
        # history and the local attention receptive field at cycle eight.
        ordinary_model = DirectQualityTransformer(tiny_config(context_length=2)).eval()
        with torch.no_grad():
            torch.testing.assert_close(ordinary_model.forward_full(**tensors)[:, 8],
                                       ordinary_model.forward_full(**changed)[:, 8],
                                       rtol=0, atol=0)

    def test_dilated_encoder_masks_padding_between_every_convolution(self):
        model = DirectQualityTransformer(tiny_config(base_encoder_type="dilated_cnn")).eval()
        # Learned biases make accidentally feeding padded hidden states back
        # into a later convolution observable even before model training.
        with torch.no_grad():
            for block in model.base_encoder.blocks:
                block.norm.bias.fill_(.6)
                block.convolution.bias.fill_(.4)
        short = batch_tensors((7,))
        mixed = batch_tensors((7, 23))
        with torch.no_grad():
            short_base = model.base_encoder(short["bases"], short["active_mask"])
            mixed_base = model.base_encoder(mixed["bases"], mixed["active_mask"])
            short_logits = model.forward_full(**short)
            mixed_logits = model.forward_full(**mixed)
        torch.testing.assert_close(short_base[0], mixed_base[0, :7], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(short_logits[0], mixed_logits[0, :7], rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(mixed_base[0, 7:], torch.zeros_like(mixed_base[0, 7:])))

    def test_dilated_base_features_can_use_distant_decoder_known_future_bases(self):
        model = DirectQualityTransformer(tiny_config(base_encoder_type="dilated_cnn")).eval()
        tensors = batch_tensors((23,))
        changed = dict(tensors)
        changed["bases"] = tensors["bases"].clone()
        changed["bases"][0, 12] = (changed["bases"][0, 12] + 1) % 5
        with torch.no_grad():
            original = model.forward_full(**tensors)
            different = model.forward_full(**changed)
        self.assertFalse(torch.equal(original[:, 0], different[:, 0]))

    def test_swiglu_gate_affects_eval_and_receives_training_gradients(self):
        model = DirectQualityTransformer(tiny_config(ffn_type="swiglu")).eval()
        tensors = batch_tensors()
        layer = model.transformer.layers[0]
        with torch.no_grad():
            initial = model.forward_full(**tensors)
            layer.linear_gate.weight.zero_()
            layer.linear_gate.bias.fill_(2)
            changed = model.forward_full(**tensors)
        self.assertFalse(torch.equal(initial, changed))
        model.train()
        loss = masked_cross_entropy(model.forward_full(**tensors),
                                    tensors["qualities"], tensors["active_mask"])
        loss.backward()
        self.assertIsNotNone(layer.linear_gate.weight.grad)
        self.assertGreater(float(layer.linear_gate.weight.grad.abs().sum()), 0)

    def test_all_variants_have_identical_full_and_incremental_integer_cdfs(self):
        tensors = batch_tensors()
        variants = VARIANTS + (("combined", dict(
            base_encoder_type="dilated_cnn", ffn_type="swiglu", quality_prefix_features=True)),)
        for name, options in variants:
            with self.subTest(name=name):
                model = DirectQualityTransformer(tiny_config(**options)).eval()
                with torch.no_grad():
                    full = model.forward_full(**tensors)
                    for cycle in range(full.shape[1]):
                        step = model.forward_step(
                            tensors["bases"], tensors["qualities"][:, :cycle],
                            tensors["lengths"], tensors["active_mask"])
                        torch.testing.assert_close(full[:, cycle], step, rtol=0, atol=0)
                        for version in (1, 2):
                            np.testing.assert_array_equal(
                                logits_to_cdfs(full[:, cycle].numpy(), version=version),
                                logits_to_cdfs(step.numpy(), version=version))

    def test_variant_checkpoints_roundtrip_through_the_real_codec(self):
        raw = b"@short\nACGT\n+\n!+5J\n@long\nTGCAACGTA\n+id\nJ50+!555J\n@empty\n\n+\n\n"
        for name, options in VARIANTS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = DirectQualityTransformer(tiny_config(**options)).eval()
                checkpoint = root / "model.pt"
                save_checkpoint(checkpoint, model)
                restored = load_training_checkpoint(checkpoint)
                self.assertEqual(restored.model.config, model.config)
                self.assertEqual(restored.payload["feature_schema"], feature_schema(model.config))
                source, archive, decoded = root / "source.fq", root / "archive.fqdc", root / "decoded.fq"
                source.write_bytes(raw)
                kwargs = dict(device=torch.device("cpu"), batch_reads=2,
                              progress=False, verify_cdf=True)
                encode_fastq(source, archive, checkpoint, quantization_version=2, **kwargs)
                decode_fastq(archive, decoded, checkpoint, **kwargs)
                self.assertEqual(decoded.read_bytes(), raw)

    def test_each_variant_trains_and_reloads_through_the_pretraining_entry_point(self):
        datasets = (DirectQualityDataset("TOY_A", "Illumina", True),
                    DirectQualityDataset("TOY_B", "MGI/BGI", True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, dataset in enumerate(datasets):
                source = root / (dataset.accession + ".fq")
                source.write_bytes(b"".join(
                    b"@r%d\nACGT\n+\n" % read
                    + bytes(33 + (read + index + position) % 42 for position in range(4))
                    + b"\n" for read in range(12)))
                create_training_cache(source, default_cache_path(source, root))
            for name, options in VARIANTS:
                with self.subTest(name=name):
                    config = tiny_config(**options)
                    torch.manual_seed(11)
                    initial = DirectQualityTransformer(config).state_dict()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        best = run_training(
                            cache_dir=root, output_dir=root / name, model_config=config,
                            epochs=1, steps_per_epoch=2, batch_reads=2,
                            train_fraction=.75, validation_max_reads_per_file=2,
                            learning_rate=.001, weight_decay=0, grad_clip=1,
                            device=torch.device("cpu"), seed=11,
                            train_datasets=datasets, unseen_datasets=(), progress=False)
                    loaded = load_training_checkpoint(best)
                    self.assertEqual(loaded.model.config, config)
                    self.assertEqual(loaded.payload["global_step"], 2)
                    self.assertEqual(loaded.payload["feature_schema"], feature_schema(config))
                    self.assertEqual(set(loaded.payload["validation_metrics"]["per_dataset"]),
                                     {"TOY_A", "TOY_B"})
                    if config.ffn_type == "swiglu":
                        parameter = "transformer.layers.0.linear_gate.weight"
                    elif config.base_encoder_type == "dilated_cnn":
                        parameter = "base_encoder.blocks.0.convolution.weight"
                    else:
                        parameter = "input_projection.0.weight"
                    self.assertFalse(torch.equal(initial[parameter],
                                                 loaded.model.state_dict()[parameter]))


if __name__ == "__main__":
    unittest.main()
