"""Check source weighting and optimizer-batch microbatch gradient equivalence."""

import unittest

import torch
from torch.nn import functional as F

from codec.mixed_read_loss import source_weighted_cross_entropy


class MixedObjectiveTest(unittest.TestCase):
    def test_long_reads_do_not_change_source_weight_and_padding_is_ignored(self):
        logits = torch.tensor([[[3., 0.], [1., 2.], [5., 3.]],
                               [[.5, 1.], [1., 4.], [2., 2.]]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 42, 42], [1, 0, 1]])
        mask = targets.ne(42)
        probabilities = torch.tensor([.5, .5], dtype=torch.float64)
        counts = torch.tensor([1., 3.], dtype=torch.float64)
        loss, nll = source_weighted_cross_entropy(
            logits, targets, mask, torch.tensor([0, 1]), probabilities, counts)
        expected = (.5 * F.cross_entropy(logits[0, :1], targets[0, :1])
                    + .5 * F.cross_entropy(logits[1], targets[1]))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-12))
        full_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, logits)
        self.assertTrue(torch.allclose(full_grad, expected_grad, atol=1e-12))
        self.assertEqual(float(full_grad[0, 1:].abs().sum()), 0.)
        self.assertTrue(torch.allclose(loss, (nll / counts * probabilities).sum()))
        pieces = logits.detach().clone().requires_grad_(True)
        for row in range(2):
            part, _ = source_weighted_cross_entropy(
                pieces[row:row+1], targets[row:row+1], mask[row:row+1],
                torch.tensor([row]), probabilities, counts)
            part.backward()
        self.assertTrue(torch.allclose(pieces.grad, full_grad, atol=1e-12))

    def test_interleaved_sources_with_unequal_weights_accumulate_exact_micro_gradients(self):
        generator = torch.Generator().manual_seed(753)
        logits = torch.randn(7, 5, 4, generator=generator, dtype=torch.float64,
                             requires_grad=True)
        lengths = torch.tensor([1, 5, 2, 4, 3, 1, 5])
        mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
        targets = torch.randint(0, 4, (7, 5), generator=generator).masked_fill(~mask, 42)
        sources = torch.tensor([2, 0, 1, 0, 2, 1, 2])
        probabilities = torch.tensor([.3, .2, .5], dtype=torch.float64)
        counts = torch.tensor([9., 3., 9.], dtype=torch.float64)
        loss, _ = source_weighted_cross_entropy(
            logits, targets, mask, sources, probabilities, counts)
        reference = sum(
            probabilities[source] * F.cross_entropy(
                logits[sources == source][mask[sources == source]],
                targets[sources == source][mask[sources == source]])
            for source in range(3))
        self.assertTrue(torch.allclose(loss, reference, atol=1e-12))
        gradient, = torch.autograd.grad(loss, logits)
        pieces = logits.detach().clone().requires_grad_(True)
        accumulated = 0.
        # Some pieces omit a source; their denominator still covers the full batch.
        for start, stop in ((0, 2), (2, 5), (5, 7)):
            part, _ = source_weighted_cross_entropy(
                pieces[start:stop], targets[start:stop], mask[start:stop],
                sources[start:stop], probabilities, counts)
            accumulated += part.detach()
            part.backward()
        self.assertTrue(torch.allclose(accumulated, reference.detach(), atol=1e-12))
        self.assertTrue(torch.allclose(pieces.grad, gradient, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
