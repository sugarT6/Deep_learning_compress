import unittest
from pathlib import Path

import torch

from predict_sequence_residual_transformer import platform_id_for_file
from sequence_residual_transformer_model import (
    ALPHABET_SIZE,
    DIRECT_LOGITS,
    PLATFORM_TO_ID,
    QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
    Q_BOS_TOKEN,
    R_BOS_TOKEN,
    ResidualTransformer,
)
from train_sequence_residual_transformer import resolve_platform_map


class PlatformFeatureTest(unittest.TestCase):
    def test_explicit_platform_map_resolves_all_six_files(self) -> None:
        files = [
            Path("h5/ERR2755197_1.block.fq.gz.qual_model.h5"),
            Path("h5/SRR1238539.block.fq.gz.qual_model.h5"),
            Path("h5/SRR3066199_1.block.fq.gz.qual_model.h5"),
            Path("h5/SRR5867380.block.fq.gz.qual_model.h5"),
            Path("h5/SRR622457_1.block.fq.gz.qual_model.h5"),
            Path("h5/SRR6691666_1.block.fq.gz.qual_model.h5"),
        ]
        mapping = resolve_platform_map(
            files,
            "ERR2755197=BGISEQ,SRR1238539=IonTorrent,"
            "SRR3066199=Illumina,SRR5867380=IonTorrent,"
            "SRR622457=Illumina,SRR6691666=Illumina",
        )
        self.assertEqual(mapping[files[0].name], "BGISEQ")
        self.assertEqual(mapping[files[1].name], "IonTorrent")
        self.assertEqual(mapping[files[2].name], "Illumina")

    def test_platform_map_rejects_missing_or_ambiguous_match(self) -> None:
        file = Path("h5/SRR1238539.block.fq.gz.qual_model.h5")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_platform_map([file], "ERR2755197=BGISEQ")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_platform_map(
                [file],
                "SRR1238539=IonTorrent,SRR=Illumina",
            )

    def test_platform_embedding_is_required_and_preserves_output_shape(self) -> None:
        model = ResidualTransformer(
            continuous_dim=QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
            qmer_ks=(),
            rmer_ks=(),
            platform_embed_dim=4,
            d_model=16,
            num_heads=4,
            num_layers=1,
            feedforward_dim=32,
            context_length=8,
            dropout=0.0,
            output_dim=ALPHABET_SIZE,
            output_parameterization=DIRECT_LOGITS,
        )
        continuous = torch.zeros(2, 3, QHAT_ONLY_CONTINUOUS_FEATURE_DIM)
        q_hat = torch.zeros(2, 3, dtype=torch.long)
        prev_q = torch.full((2, 3), Q_BOS_TOKEN, dtype=torch.long)
        prev_r = torch.full((2, 3), R_BOS_TOKEN, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "platform_id is required"):
            model(continuous, q_hat, prev_q, prev_r)

        logits = model(
            continuous,
            q_hat,
            prev_q,
            prev_r,
            platform_id=torch.tensor(
                [PLATFORM_TO_ID["BGISEQ"], PLATFORM_TO_ID["Illumina"]]
            ),
        )
        self.assertEqual(tuple(logits.shape), (2, 3, ALPHABET_SIZE))
        self.assertFalse(torch.allclose(logits[0], logits[1]))

    def test_prediction_restores_stable_platform_id(self) -> None:
        path = Path("other/location/ERR2755197_1.block.fq.gz.qual_model.h5")
        config = {
            "platform_embed_dim": 8,
            "platform_id_by_file": {path.name: PLATFORM_TO_ID["BGISEQ"]},
        }
        self.assertEqual(
            platform_id_for_file(config, path),
            PLATFORM_TO_ID["BGISEQ"],
        )


if __name__ == "__main__":
    unittest.main()
