import copy
import unittest

import numpy as np
import torch

from codec._legacy_quality_history import (
    history_feature_schema, quality_history_features, validate_history_feature_schema,
)


class QualityHistoryFeaturesTest(unittest.TestCase):
    def test_matches_scalar_formula_and_missing_history(self):
        q = torch.tensor([[0, 41, 20, 5, 40, 30, 10, 35, 15, 3, 41, 0],
                          [41, 0, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42],
                          [42] * 12])
        mask = q < 42
        actual = quality_history_features(q, mask).numpy()
        expected = np.zeros_like(actual)
        for row in range(q.shape[0]):
            for t in range(int(mask[row].sum())):
                past = q[row, :t].numpy().astype(np.float64)
                mean = past.mean() / 41 if t else 0
                window_mean = past[-8:].mean() / 41 if t else 0
                expected[row, t] = [past[-2] / 41 if t >= 2 else -1,
                                    past[-3] / 41 if t >= 3 else -1,
                                    (past[-1] - past[-2]) / 41 if t >= 2 else 0,
                                    mean, past.std() / 41 if t else 0, window_mean,
                                    window_mean - mean, min(t, 8) / 8]
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(actual[0, 0], [-1, -1, 0, 0, 0, 0, 0, 0])

    def test_every_prefix_exactly_matches_full_and_ignores_future(self):
        generator = torch.Generator().manual_seed(21)
        q = torch.randint(42, (4, 150), generator=generator)
        lengths = torch.tensor([150, 9, 1, 0])
        mask = torch.arange(150)[None, :] < lengths[:, None]
        q[~mask] = 42
        full = quality_history_features(q, mask)
        for t in range(150):
            partial = q.clone()
            partial[:, t:] = 42
            step = quality_history_features(partial, mask)
            self.assertTrue(torch.equal(full[:, :t + 1], step[:, :t + 1]), t)
            partial[:, t:] = 17
            self.assertTrue(torch.equal(full[:, :t + 1], quality_history_features(partial, mask)[:, :t + 1]), t)
        self.assertEqual(quality_history_features(q[:, :0], mask[:, :0]).shape, (4, 0, 8))

    def test_wire_schema_is_strict(self):
        validate_history_feature_schema(history_feature_schema())
        for key, value in (("version", True), ("version", 1.0), ("window", 16),
                           ("quality_scale", 42), ("std", "sample"), ("scope", "whole_read")):
            bad = copy.deepcopy(history_feature_schema())
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_history_feature_schema(bad)
        for bad in (None, {}, {**history_feature_schema(), "extra": 1}):
            with self.assertRaises(ValueError):
                validate_history_feature_schema(bad)
