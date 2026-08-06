import gzip
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from prepare_base_sidecars import create_base_sidecar
from sequence_residual_transformer_model import (
    BASE_A_TOKEN,
    BASE_C_TOKEN,
    BASE_G_TOKEN,
    BASE_N_TOKEN,
    BASE_PAD_TOKEN,
    BASE_T_TOKEN,
    QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
    RESIDUAL_CLASSES,
    ResidualTransformer,
    base_sidecar_path_for_h5,
    batch_to_torch,
    inspect_base_sidecar,
    read_h5_read_range,
)


class BaseFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir_context.name)
        self.h5_path = self.root / "SRRTEST.block.fq.gz.qual_model.h5"
        self.fastq_path = self.root / "SRRTEST.block.fq.gz"
        self.sidecar_dir = self.root / "base_sidecars"
        self.sidecar_path = base_sidecar_path_for_h5(self.h5_path, self.sidecar_dir)

        # Body lengths are 3, 0, and 2. The complete raw reads remain longer,
        # so the model can use bases corresponding to the trailing Q2 run.
        observed = np.asarray([32, 33, 34, 35, 36], dtype=np.uint8)
        offsets = np.asarray([0, 3, 3, 5], dtype=np.int64)
        freqs = np.ones((observed.size, 95), dtype=np.uint32)
        freqs[np.arange(observed.size), observed] = 20
        with h5py.File(self.h5_path, "w") as handle:
            handle.create_dataset("/observed", data=observed)
            handle.create_dataset("/freqs", data=freqs)
            handle.create_dataset("/read_offsets", data=offsets)

        with gzip.open(self.fastq_path, "wb") as handle:
            handle.write(
                b"@read0\nACGT\n+\nABC#\n"
                b"@read1\nTNA\n+\n###\n"
                b"@read2\nGGCCT\n+\nDE###\n"
            )

        create_base_sidecar(
            h5_path=self.h5_path,
            fastq_path=self.fastq_path,
            output_path=self.sidecar_path,
        )

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    def test_sidecar_uses_flat_full_read_layout(self) -> None:
        info = inspect_base_sidecar(self.h5_path, self.sidecar_path)
        self.assertEqual(info.read_count, 3)
        self.assertEqual(info.base_count, 12)
        with h5py.File(self.sidecar_path, "r") as handle:
            np.testing.assert_array_equal(
                handle["/base_read_offsets"][:],
                np.asarray([0, 4, 7, 12], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                handle["/body_lengths"][:],
                np.asarray([3, 0, 2], dtype=np.int64),
            )
            np.testing.assert_array_equal(
                handle["/base_ids"][:],
                np.asarray(
                    [
                        BASE_A_TOKEN,
                        BASE_C_TOKEN,
                        BASE_G_TOKEN,
                        BASE_T_TOKEN,
                        BASE_T_TOKEN,
                        BASE_N_TOKEN,
                        BASE_A_TOKEN,
                        BASE_G_TOKEN,
                        BASE_G_TOKEN,
                        BASE_C_TOKEN,
                        BASE_C_TOKEN,
                        BASE_T_TOKEN,
                    ],
                    dtype=np.uint8,
                ),
            )

    def test_batch_keeps_full_bases_and_skips_empty_quality_read(self) -> None:
        batch = read_h5_read_range(
            path=self.h5_path,
            read_start=0,
            read_stop=3,
            base_sidecar_path=self.sidecar_path,
        )
        np.testing.assert_array_equal(batch.lengths, np.asarray([3, 2]))
        np.testing.assert_array_equal(batch.base_lengths, np.asarray([4, 5]))
        self.assertEqual(batch.base_ids.shape, (2, 5))
        self.assertEqual(int(batch.base_ids[0, 3]), BASE_T_TOKEN)
        self.assertEqual(int(batch.base_ids[0, 4]), BASE_PAD_TOKEN)
        self.assertEqual(int(batch.base_ids[1, 4]), BASE_T_TOKEN)
        self.assertEqual(batch.exact_q_lags.shape, (2, 3, 0))
        self.assertEqual(batch.exact_r_lags.shape, (2, 3, 0))
        self.assertEqual(batch.continuous.shape[-1], QHAT_ONLY_CONTINUOUS_FEATURE_DIM)
        np.testing.assert_array_equal(batch.zero_residual_run[0], np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(batch.same_quality_run[0], np.asarray([0, 1, 1]))

    def test_base_conv_model_forward_and_checkpoint_shapes(self) -> None:
        batch = read_h5_read_range(
            path=self.h5_path,
            read_start=0,
            read_stop=3,
            base_sidecar_path=self.sidecar_path,
        )
        tensors = batch_to_torch(batch, torch.device("cpu"))
        model = ResidualTransformer(
            continuous_dim=QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
            qmer_ks=(2,),
            rmer_ks=(2,),
            base_embed_dim=4,
            base_conv_kernels=(3, 5, 7),
            base_conv_channels=4,
            base_context_dim=8,
            d_model=16,
            num_heads=4,
            num_layers=1,
            feedforward_dim=32,
            context_length=8,
            dropout=0.0,
        )
        logits = model(
            continuous=tensors["continuous"],
            q_hat=tensors["q_hat"],
            prev_q=tensors["prev_q"],
            prev_r=tensors["prev_r"],
            exact_q_lags=tensors["exact_q_lags"],
            exact_r_lags=tensors["exact_r_lags"],
            zero_residual_run=tensors["zero_residual_run"],
            same_quality_run=tensors["same_quality_run"],
            qmer_tokens=tensors["qmer_tokens"],
            rmer_tokens=tensors["rmer_tokens"],
            base_ids=tensors["base_ids"],
            lengths=tensors["lengths"],
        )
        self.assertEqual(tuple(logits.shape), (2, 3, RESIDUAL_CLASSES))

        with self.assertRaisesRegex(ValueError, "base_ids are required"):
            model(
                continuous=tensors["continuous"],
                q_hat=tensors["q_hat"],
                prev_q=tensors["prev_q"],
                prev_r=tensors["prev_r"],
                exact_q_lags=tensors["exact_q_lags"],
                exact_r_lags=tensors["exact_r_lags"],
                zero_residual_run=tensors["zero_residual_run"],
                same_quality_run=tensors["same_quality_run"],
                qmer_tokens=tensors["qmer_tokens"],
                rmer_tokens=tensors["rmer_tokens"],
                lengths=tensors["lengths"],
            )


if __name__ == "__main__":
    unittest.main()
