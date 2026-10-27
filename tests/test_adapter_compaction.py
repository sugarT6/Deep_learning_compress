import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
import torch

from codec.checkpoint import save_training_checkpoint
from codec.compact_adapter import compact_artifact, compact_suffix_adapter, round_float32_mantissa
from codec.container import (ContainerError, ContainerIntegrityError, CONTAINER_PREFIX_BYTES,
    adaptation_metadata_bytes, read_container, sha256_file)
from codec.decode import decode_fastq
from codec.encode import encode_fastq, build_parser
from codec.head_adapter import _new_head
from codec.model import DirectQualityModelConfig, DirectQualityTransformer
from codec.read_state import ReadStateConditioning, serialize_read_state
from codec.suffix_lora_adapter import install_suffix_lora, serialize_suffix_lora


PREFIX = struct.Struct('<8sHHII')


class AdapterCompactionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(96)
        self.base = DirectQualityTransformer(DirectQualityModelConfig(
            d_model=8, num_heads=2, num_layers=4, feedforward_dim=16,
            prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
            base_conv_channels=3, base_context_dim=4, context_length=32, dropout=0))
        self.checkpoint = self.root / 'base.pt'
        save_training_checkpoint(self.checkpoint, model=self.base, optimizer=None, epoch=1, global_step=1,
            data_split={}, sampler_config={}, sampler_statistics={},
            best_validation_bits_per_quality=1., validation_metrics={})
        self.digest = sha256_file(self.checkpoint)
        warm = copy.deepcopy(self.base).eval().requires_grad_(False)
        warm.output_head = _new_head(8, 16, cross_dim=4, device='cpu', dtype=torch.float32, seed=4)
        student = copy.deepcopy(warm)
        layout = install_suffix_lora(student, 4, 2, 9)
        with torch.no_grad():
            for item in layout:
                module, name = item['name'].rsplit('.', 1)
                student.get_submodule(module).parametrizations[name][0].b.normal_(std=.03)
            student.output_head.up.weight.normal_(std=.03)
            student.output_head.cross_up.weight.normal_(std=.03)
        self.adapter = serialize_suffix_lora(student, 4, 2, self.digest, base_model=warm)
        self.artifact = self.root / 'original.adapter.json'
        self.artifact.write_text(json.dumps(dict(format='direct-quality-head-adaptation-artifact',
            version=1, base_checkpoint_sha256=self.digest, adapter=self.adapter, report={})))
        state_model = copy.deepcopy(student)
        state_model.read_state_conditioning = ReadStateConditioning(8, range(4), [20000])
        state_adapter = serialize_read_state(state_model, self.adapter, self.digest)
        self.state_artifact = self.root / 'state.adapter.json'
        self.state_artifact.write_text(json.dumps(dict(format='direct-quality-head-adaptation-artifact',
            version=1, base_checkpoint_sha256=self.digest, adapter=state_adapter, report={})))
        rows = []
        for i in range(11):
            n = 0 if i == 0 else i % 4 + 1
            ending = b'\r\n' if i % 2 else b'\n'
            q = bytes(33 + (i * 7 + j * 11) % 42 for j in range(n))
            rows.append(b'@r' + str(i).encode() + ending + b'ACGT'[:n] + ending
                        + b'+description' + ending + q + (b'' if i == 10 else ending))
        self.raw = b''.join(rows)
        self.source = self.root / 'sample.fastq'
        self.source.write_bytes(self.raw)

    def encode(self, name, compressed=False, artifact=True):
        target = self.root / (name + '.fqdc')
        path = self.artifact if artifact is True else (None if artifact is False else artifact)
        stats = encode_fastq(self.source, target, self.checkpoint, device=torch.device('cpu'),
            batch_reads=8, quantization_version=2, progress=False, verify_cdf=True,
            coder_backend='python', head_adapter_path=path, compress_metadata=compressed)
        return target, stats

    def decode(self, target, name):
        restored = self.root / (name + '.restored.fastq')
        decode_fastq(target, restored, self.checkpoint, device=torch.device('cpu'),
                     batch_reads=8, progress=False)
        self.assertEqual(restored.read_bytes(), self.raw)

    def rewrite(self, source, name, payload, flags=None):
        raw = source.read_bytes()
        magic, version, original_flags, size, _ = PREFIX.unpack(raw[:PREFIX.size])
        target = self.root / name
        target.write_bytes(PREFIX.pack(magic, version, original_flags if flags is None else flags,
            len(payload), zlib.crc32(payload) & 0xffffffff) + payload + raw[PREFIX.size + size:])
        return target

    def test_mantissa_rounding_ties_sign_zero_and_input_preservation(self):
        original = np.asarray([1.125, 1.375, -1.125, -1.375, 0., -0.], dtype=np.float32)
        saved = original.view(np.uint32).copy()
        rounded = round_float32_mantissa(original, 2)
        np.testing.assert_array_equal(rounded, [1., 1.5, -1., -1.5, 0., -0.])
        self.assertEqual(int(rounded.view(np.uint32)[-1]), 0x80000000)
        np.testing.assert_array_equal(original.view(np.uint32), saved)
        np.testing.assert_array_equal(round_float32_mantissa(original, 23).view(np.uint32), saved)
        for invalid in (True, 1, 24, 8.5):
            with self.assertRaises(ValueError):
                round_float32_mantissa(original, invalid)
        for invalid in (np.nan, np.inf, np.finfo(np.float32).max):
            with self.assertRaises(ValueError):
                round_float32_mantissa([invalid], 2)

    def test_full_precision_preserves_exact_adapter_and_compaction_keeps_base(self):
        before = {k: v.detach().clone() for k, v in self.base.state_dict().items()}
        identity = compact_suffix_adapter(self.adapter, self.base, self.digest,
                                          factor_mantissa_bits=23, head_mantissa_bits=23)
        self.assertEqual(identity, self.adapter)
        compact = compact_suffix_adapter(self.adapter, self.base, self.digest,
                                         factor_mantissa_bits=4, head_mantissa_bits=8)
        self.assertEqual(compact['rank'], self.adapter['rank'])
        self.assertEqual(compact['parameter_bytes'], self.adapter['parameter_bytes'])
        self.assertNotEqual(compact['sha256'], self.adapter['sha256'])
        for name, value in self.base.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_compacted_v8_artifact_and_compressed_container_roundtrip(self):
        compact = self.root / 'compact.adapter.json'
        compact_artifact(self.artifact, self.checkpoint, compact,
                         factor_mantissa_bits=4, head_mantissa_bits=8)
        target, _ = self.encode('compact', compressed=True, artifact=compact)
        self.decode(target, 'compact')
        with self.assertRaises(FileExistsError):
            compact_artifact(self.artifact, self.checkpoint, compact)
        with self.assertRaises(ValueError):
            compact_suffix_adapter(self.adapter, self.base, '0' * 64)

    def test_transport_preserves_all_metadata_and_sections_and_charges_wire_cost(self):
        plain, plain_stats = self.encode('plain')
        packed, packed_stats = self.encode('packed', compressed=True)
        pi, zi = read_container(plain), read_container(packed)
        self.assertFalse(pi.metadata_compressed)
        self.assertTrue(zi.metadata_compressed)
        self.assertEqual(pi.metadata, zi.metadata)
        for section in pi.sections:
            with pi.open_section(section.name) as a, zi.open_section(section.name) as b:
                self.assertEqual(a.read(), b.read())
        self.assertGreater(pi.header_bytes, zi.header_bytes)
        self.assertEqual(pi.file_size - zi.file_size, pi.header_bytes - zi.header_bytes)
        base = dict(zi.metadata); base.pop('head_adapter'); base.pop('probability_profile')
        canonical = lambda m: json.dumps(m, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        expected = len(zlib.compress(canonical(zi.metadata), 9)) - len(zlib.compress(canonical(base), 9))
        self.assertEqual(packed_stats.head_adaptation['container_adapter_bytes'], expected)
        self.assertEqual(packed_stats.quality_and_adapter_bytes, packed_stats.range_stream_bytes + expected)
        self.assertEqual(packed_stats.head_adaptation['container_adapter_bytes_uncompressed'],
                         plain_stats.head_adaptation['container_adapter_bytes'])
        self.decode(packed, 'packed')

    def test_neural_only_and_read_state_containers_roundtrip(self):
        target, stats = self.encode('neural', compressed=True, artifact=False)
        self.assertEqual(stats.quality_and_adapter_bytes, stats.range_stream_bytes)
        self.decode(target, 'neural')
        target, stats = self.encode('state', compressed=True, artifact=self.state_artifact)
        info = read_container(target)
        self.assertEqual(info.metadata['format_version'], 4)
        self.assertEqual(stats.read_state_stream_bytes, 2)
        adapter, labels = adaptation_metadata_bytes(info.metadata, compressed=True)
        self.assertEqual(stats.quality_and_adapter_bytes, stats.range_stream_bytes + adapter + labels + 2)
        self.decode(target, 'state')

    def test_corrupt_compressed_metadata_crc_and_streams_are_rejected(self):
        target, _ = self.encode('packed', compressed=True, artifact=False)
        raw = bytearray(target.read_bytes()); raw[CONTAINER_PREFIX_BYTES] ^= 1
        corrupt = self.root / 'crc.fqdc'; corrupt.write_bytes(raw)
        with self.assertRaisesRegex(ContainerIntegrityError, 'checksum'):
            read_container(corrupt)
        _, _, _, size, _ = PREFIX.unpack(target.read_bytes()[:PREFIX.size])
        payload = target.read_bytes()[PREFIX.size:PREFIX.size + size]
        for name, modified in [('truncated', payload[:-1]), ('trailing', payload + b'x'),
                               ('second_stream', payload + zlib.compress(b'{}')), ('invalid', b'no zlib stream')]:
            with self.subTest(name=name), self.assertRaises(ContainerIntegrityError):
                read_container(self.rewrite(target, name + '.fqdc', modified))
        with self.assertRaisesRegex(ContainerError, 'flags'):
            read_container(self.rewrite(target, 'flags.fqdc', payload, flags=2))

    def test_decompression_limit_and_canonical_json_are_enforced(self):
        target, _ = self.encode('packed', compressed=True, artifact=False)
        bomb = self.rewrite(target, 'oversize.fqdc', zlib.compress(b' ' * 4097))
        with mock.patch('codec.container.MAX_METADATA_BYTES', 4096):
            with self.assertRaisesRegex(ContainerError, 'decompressed.*size limit'):
                read_container(bomb)
        raw = json.dumps(read_container(target).metadata, indent=2).encode()
        with self.assertRaisesRegex(ContainerError, 'canonical'):
            read_container(self.rewrite(target, 'noncanonical.fqdc', zlib.compress(raw)))

    def test_cli_flag_and_invalid_boolean(self):
        args = build_parser().parse_args(['a.fq', 'b.fqdc', 'c.pt', '--compress-metadata'])
        self.assertTrue(args.compress_metadata)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.encode('invalid', compressed='yes')


if __name__ == '__main__':
    unittest.main()
