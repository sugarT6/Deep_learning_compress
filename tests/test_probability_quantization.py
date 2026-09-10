import math
import unittest

import numpy as np

from codec.probability_quantization import (
    QUALITY_ALPHABET_SIZE,
    TOTAL,
    ProbabilityQuantizationError,
    frequencies_to_cdf,
    logits_to_cdf,
    logits_to_frequencies,
    probabilities_to_cdf,
    probabilities_to_frequencies,
    quantized_symbol_bits,
)


class ProbabilityQuantizationTest(unittest.TestCase):
    def test_tied_probabilities_use_quality_id_as_fixed_tie_break(self):
        probabilities = np.ones(QUALITY_ALPHABET_SIZE, dtype=np.float64)
        frequencies = probabilities_to_frequencies(probabilities)

        remaining = TOTAL - QUALITY_ALPHABET_SIZE
        quotient, remainder = divmod(remaining, QUALITY_ALPHABET_SIZE)
        expected = tuple(
            1 + quotient + (quality_id < remainder)
            for quality_id in range(QUALITY_ALPHABET_SIZE)
        )
        self.assertEqual(frequencies, expected)
        self.assertEqual(logits_to_frequencies(np.zeros(42)), expected)

    def test_q0_q41_minimum_frequency_and_cdf_contract(self):
        probabilities = np.zeros(QUALITY_ALPHABET_SIZE, dtype=np.float64)
        probabilities[0] = 1.0
        probabilities[41] = 1e-300
        frequencies = probabilities_to_frequencies(probabilities)
        cdf = frequencies_to_cdf(frequencies, total=TOTAL)

        self.assertEqual(frequencies[41], 1)
        self.assertGreater(frequencies[0], frequencies[41])
        self.assertEqual(min(frequencies), 1)
        self.assertEqual(sum(frequencies), TOTAL)
        self.assertEqual(len(cdf), 43)
        self.assertEqual(cdf[0], 0)
        self.assertEqual(cdf[-1], TOTAL)
        self.assertTrue(all(left < right for left, right in zip(cdf, cdf[1:])))
        self.assertAlmostEqual(
            quantized_symbol_bits(41, cdf), math.log2(TOTAL), places=12
        )

    def test_probability_and_logit_quantization_is_repeatable(self):
        rng = np.random.default_rng(20260910)
        probabilities = rng.random(QUALITY_ALPHABET_SIZE)
        logits = rng.normal(size=QUALITY_ALPHABET_SIZE)

        expected_probabilities = probabilities_to_cdf(probabilities)
        expected_logits = logits_to_cdf(logits)
        for _ in range(20):
            self.assertEqual(probabilities_to_cdf(probabilities), expected_probabilities)
            self.assertEqual(logits_to_cdf(logits), expected_logits)

    def test_total_equal_to_alphabet_assigns_every_class_one(self):
        frequencies = probabilities_to_frequencies(np.arange(1, 43), total=42)
        self.assertEqual(frequencies, (1,) * 42)
        self.assertEqual(frequencies_to_cdf(frequencies, total=42), tuple(range(43)))

    def test_rejects_invalid_probabilities_logits_dimensions_and_total(self):
        valid = np.ones(QUALITY_ALPHABET_SIZE)
        invalid_vectors = (
            np.ones(41),
            np.ones(43),
            np.ones((1, 42)),
            np.full(42, np.nan),
            np.full(42, np.inf),
        )
        for vector in invalid_vectors:
            with self.subTest(shape=vector.shape, first=vector.flat[0]):
                with self.assertRaises(ProbabilityQuantizationError):
                    probabilities_to_frequencies(vector)
                with self.assertRaises(ProbabilityQuantizationError):
                    logits_to_frequencies(vector)

        negative = valid.copy()
        negative[4] = -0.01
        with self.assertRaises(ProbabilityQuantizationError):
            probabilities_to_frequencies(negative)
        with self.assertRaises(ProbabilityQuantizationError):
            probabilities_to_frequencies(np.zeros(42))
        for total in (0, 41, 41.5, True):
            with self.subTest(total=total):
                with self.assertRaises(ProbabilityQuantizationError):
                    probabilities_to_frequencies(valid, total=total)

    def test_rejects_invalid_frequency_and_quality_cdf_inputs(self):
        with self.assertRaises(ProbabilityQuantizationError):
            frequencies_to_cdf([1] * 41)
        with self.assertRaises(ProbabilityQuantizationError):
            frequencies_to_cdf([1] * 41 + [0])
        with self.assertRaises(ProbabilityQuantizationError):
            frequencies_to_cdf([1] * 42, total=43)
        with self.assertRaises(ProbabilityQuantizationError):
            quantized_symbol_bits(42, tuple(range(43)))
        with self.assertRaises(ProbabilityQuantizationError):
            quantized_symbol_bits(0, (1,) + tuple(range(1, 43)))


if __name__ == "__main__":
    unittest.main()
