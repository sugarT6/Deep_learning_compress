import base64
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
import torch

from codec.checkpoint import save_training_checkpoint
from codec.container import (ContainerError, adaptation_metadata_bytes, read_container,
    sha256_file, _validate_metadata)
from codec.decode import decode_fastq
from codec.encode import encode_fastq
from codec.fastq_stream import BASE_PAD_ID
from codec.head_adapter import apply_head_adapter, serialize_head
from codec.model import DirectQualityModelConfig, DirectQualityTransformer
from codec.probability_quantization import logits_to_cdfs
from codec.quality_gru_adapter import (QualityGRUPredictor, feature_dim,
    import_research_artifact, serialize_quality_gru, validate_quality_gru)


class QualityGRUAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(103)
        self.base = DirectQualityTransformer(DirectQualityModelConfig(
            d_model=8, num_heads=2, num_layers=1, feedforward_dim=16,
            prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
            base_conv_channels=3, base_context_dim=4, context_length=32, dropout=0)).eval()
        self.checkpoint = self.root / 'public.pt'
        save_training_checkpoint(self.checkpoint, model=self.base, optimizer=None,
            epoch=1, global_step=1, data_split={}, sampler_config={}, sampler_statistics={},
            best_validation_bits_per_quality=1., validation_metrics={})
        self.digest = sha256_file(self.checkpoint)
        self.predictor = QualityGRUPredictor(feature_dim(self.base.config), 8, 2).eval()
        self.adapter = serialize_quality_gru(self.predictor, self.digest)
        self.artifact = self.root / 'gru.adapter.json'
        self.artifact.write_text(json.dumps(dict(format='direct-quality-head-adaptation-artifact',
            version=1, base_checkpoint_sha256=self.digest, adapter=self.adapter, report={})))

    def model(self):
        model = copy.deepcopy(self.base)
        apply_head_adapter(model, self.adapter, self.digest)
        return model

    def inputs(self):
        lengths = torch.tensor([0, 1, 5, 7])
        mask = torch.arange(7).unsqueeze(0) < lengths.unsqueeze(1)
        bases = torch.randint(0, 5, (4, 7))
        bases[~mask] = BASE_PAD_ID
        q = torch.randint(0, 42, (4, 7))
        q[~mask] = 42
        return bases, q, lengths, mask

    def packet(self):
        specs, chunks = [], []
        for name, tensor in sorted(self.predictor.state_dict().items()):
            specs.append(dict(name=name, shape=list(tensor.shape)))
            chunks.append(tensor.numpy().astype('<f4').tobytes())
        raw = b''.join(chunks)
        planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
        return dict(format='research-private-quality-gru-v1', mode='absolute', width=8,
            layers=2, feature_rounding='FP16->FP32', public_feature_checkpoint_sha256=self.digest,
            specs=specs, raw_sha256=hashlib.sha256(raw).hexdigest(),
            data_base64=base64.b64encode(zlib.compress(planes)).decode())

    def test_exact_weight_import_public_parameters_and_rng_preserved(self):
        model = copy.deepcopy(self.base)
        original = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        apply_head_adapter(model, self.adapter, self.digest)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, tensor in original.items():
            self.assertTrue(torch.equal(model.state_dict()[name], tensor), name)
        for name, tensor in self.predictor.state_dict().items():
            self.assertTrue(torch.equal(model.quality_gru.state_dict()[name], tensor), name)
        self.assertFalse(any(p.requires_grad for p in model.quality_gru.parameters()))
        # Reapplying a wire artifact never compounds changes.
        apply_head_adapter(model, self.adapter, self.digest)
        self.assertEqual(serialize_quality_gru(model.quality_gru, self.digest), self.adapter)

    def test_causal_full_cached_and_stateless_step_cdfs(self):
        model = self.model()
        bases, q, lengths, mask = self.inputs()
        with torch.inference_mode(), mock.patch.object(model.transformer, 'forward',
                side_effect=AssertionError('recurrent inference must skip Transformer')):
            full = model.forward_full(bases, q, lengths, mask)
            changed = q.clone()
            changed[:, 3:][mask[:, 3:]] = (changed[:, 3:][mask[:, 3:]] + 13) % 42
            future = model.forward_full(bases, changed, lengths, mask)
            self.assertTrue(torch.equal(full[:, :4], future[:, :4]))
            state = model.begin_quality_batch(bases, lengths, mask)
            with mock.patch.object(model.quality_gru, 'advance', wraps=model.quality_gru.advance) as advance:
                cached = torch.stack([model.forward_step(bases, q[:, :t], lengths, mask,
                    quality_state=state) for t in range(7)], 1)
                self.assertEqual(advance.call_count, 7)
            stateless = torch.stack([model.forward_step(bases, q[:, :t], lengths, mask)
                for t in range(7)], 1)
        for version in (1, 2):
            wanted = logits_to_cdfs(full[mask].numpy(), version=version)
            np.testing.assert_array_equal(logits_to_cdfs(cached[mask].numpy(), version=version), wanted)
            np.testing.assert_array_equal(logits_to_cdfs(stateless[mask].numpy(), version=version), wanted)

    def test_state_rejects_wrong_batch_prefix_owner_and_skipped_cycles(self):
        model = self.model()
        bases, q, lengths, mask = self.inputs()
        with torch.inference_mode():
            state = model.begin_quality_batch(bases, lengths, mask)
            with self.assertRaises(ValueError):
                model.forward_step(bases, q[:, :1], lengths, mask, quality_state=state)
            other = self.model()
            with self.assertRaises(ValueError):
                other.forward_step(bases, q[:, :0], lengths, mask, quality_state=state)
            changed = bases.clone(); changed[3, 0] = (changed[3, 0] + 1) % 5
            with self.assertRaises(ValueError):
                model.forward_step(changed, q[:, :0], lengths, mask, quality_state=state)
            model.forward_step(bases, q[:, :0], lengths, mask, quality_state=state)
            model.forward_step(bases, q[:, :1], lengths, mask, quality_state=state)
            changed_q = q[:, :2].clone(); changed_q[3, 0] = (changed_q[3, 0] + 1) % 42
            with self.assertRaises(ValueError):
                model.forward_step(bases, changed_q, lengths, mask, quality_state=state)
            with self.assertRaises(ValueError):
                model.forward_full(bases, q, lengths, mask, read_states=torch.zeros(4, dtype=torch.long))
            with self.assertRaises(ValueError):
                model.forward_features(bases, q, lengths, mask)

    def test_causal_prefix_public_features_remain_full_step_consistent(self):
        model = DirectQualityTransformer(replace(self.base.config, quality_prefix_features=True)).eval()
        private = QualityGRUPredictor(feature_dim(model.config), 8, 2)
        apply_head_adapter(model, serialize_quality_gru(private, self.digest), self.digest)
        b, q, n, m = self.inputs()
        with torch.inference_mode():
            full = model.forward_full(b, q, n, m)
            state = model.begin_quality_batch(b, n, m)
            step = torch.stack([model.forward_step(b, q[:, :t], n, m, quality_state=state)
                for t in range(7)], 1)
        np.testing.assert_array_equal(logits_to_cdfs(full[m].numpy(), version=2),
            logits_to_cdfs(step[m].numpy(), version=2))

    def test_empty_full_and_switch_back_to_legacy_head(self):
        model = self.model()
        bases = torch.empty((2, 0), dtype=torch.long)
        lengths = torch.zeros(2, dtype=torch.long)
        mask = bases.bool()
        self.assertEqual(tuple(model.forward_full(bases, bases, lengths, mask).shape), (2, 0, 42))
        with self.assertRaises(ValueError):
            model.begin_quality_batch(bases, lengths, mask)
        bad = dict(self.adapter, protocol='unknown')
        with self.assertRaises(ValueError):
            apply_head_adapter(model, bad, self.digest)
        self.assertTrue(hasattr(model, 'quality_gru'))
        legacy = serialize_head(self.base.output_head, self.digest)
        apply_head_adapter(model, legacy, self.digest)
        self.assertFalse(hasattr(model, 'quality_gru'))
        b, q, n, m = self.inputs()
        with torch.inference_mode():
            self.assertTrue(torch.equal(model.forward_full(b, q, n, m), self.base.forward_full(b, q, n, m)))

    def test_strict_payload_validation(self):
        compressed = base64.b64decode(self.adapter['data_base64'])
        variants = [dict(self.adapter, hidden_dim=True), dict(self.adapter, recurrent_layers=0),
            dict(self.adapter, feature_dim=4097), dict(self.adapter, parameter_bytes=1),
            dict(self.adapter, protocol='unknown'), dict(self.adapter, extra='unsupported'),
            dict(self.adapter, sha256='0' * 64), dict(self.adapter, data_base64='???')]
        for payload in (compressed[:-1], compressed + b'x', compressed + zlib.compress(b''),
                zlib.compress(b'\0' * (self.adapter['parameter_bytes'] + 4))):
            variants.append(dict(self.adapter, data_base64=base64.b64encode(payload).decode()))
        for meta in variants:
            with self.subTest(keys={k: v for k, v in meta.items() if k != 'data_base64'}):
                with self.assertRaises(ValueError):
                    validate_quality_gru(meta, self.digest)
        with self.assertRaises(ValueError):
            validate_quality_gru(self.adapter, '0' * 64)
        wrong = serialize_quality_gru(QualityGRUPredictor(18, 8, 2), self.digest)
        with self.assertRaises(ValueError):
            apply_head_adapter(copy.deepcopy(self.base), wrong, self.digest)
        with torch.no_grad():
            self.predictor.head.bias[0] = float('nan')
        with self.assertRaises(ValueError):
            serialize_quality_gru(self.predictor, self.digest)

    def test_research_import_converts_layout_without_changing_weights(self):
        packet = self.root / 'packet.json'
        packet.write_text(json.dumps(self.packet()))
        output = self.root / 'imported.adapter.json'
        report = import_research_artifact(packet, self.checkpoint, output)
        self.assertFalse(report['weights_changed'])
        self.assertEqual(json.loads(output.read_text())['adapter'], self.adapter)
        with self.assertRaises(FileExistsError):
            import_research_artifact(packet, self.checkpoint, output)

    def test_research_import_rejects_bad_specs_and_checkpoint(self):
        for key, value in [('name', []), ('shape', None), ('shape', [True])]:
            meta = self.packet(); meta['specs'][0][key] = value
            packet = self.root / 'packet.json'; packet.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                import_research_artifact(packet, self.checkpoint, self.root / 'bad.json')
        meta = self.packet(); meta['public_feature_checkpoint_sha256'] = '0' * 64
        packet.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            import_research_artifact(packet, self.checkpoint, self.root / 'bad.json')

    def test_real_container_both_quantizers_metadata_modes_and_empty_file(self):
        rows = []
        for i in range(11):
            n = 0 if i == 0 else i % 5 + 1
            ending = b'\r\n' if i % 2 else b'\n'
            rows.append(b'@read' + str(i).encode() + ending + b'ACGTN'[:n] + ending
                + b'+description' + ending + bytes(33 + (i * 7 + j * 11) % 42 for j in range(n))
                + (b'' if i == 10 else ending))
        for raw in (b''.join(rows), b''):
            source = self.root / 'sample.fastq'; source.write_bytes(raw)
            for version in (1, 2):
                for packed in (False, True):
                    with self.subTest(size=len(raw), version=version, packed=packed):
                        stem = f'test_{len(raw)}_{version}_{packed}'
                        target = self.root / (stem + '.fqdc')
                        restored = self.root / (stem + '.restored.fastq')
                        stats = encode_fastq(source, target, self.checkpoint, device=torch.device('cpu'),
                            batch_reads=8, quantization_version=version, verify_cdf=True, progress=False,
                            coder_backend='python', head_adapter_path=self.artifact, compress_metadata=packed)
                        info = read_container(target)
                        self.assertEqual(info.metadata['head_adapter']['version'], 10)
                        bad = copy.deepcopy(info.metadata)
                        bad['head_adapter'] = serialize_quality_gru(
                            QualityGRUPredictor(18, 8, 2), self.digest)
                        with self.assertRaises(ContainerError):
                            _validate_metadata(bad, physical_version=3)
                        paid, labels = adaptation_metadata_bytes(info.metadata, compressed=packed)
                        self.assertEqual(labels, 0)
                        self.assertEqual(stats.quality_and_adapter_bytes, stats.range_stream_bytes + paid)
                        decode_fastq(target, restored, self.checkpoint, device=torch.device('cpu'),
                            batch_reads=8, verify_cdf=True, progress=False)
                        self.assertEqual(restored.read_bytes(), raw)
