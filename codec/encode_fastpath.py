"""Encoder-only fast paths for internally validated batches.

No persistent prior cache: every call observes the frozen pre-batch state.
The reference prior/quantizer and decoder remain unchanged.
"""

import numpy as np

from .fastq_stream import QUALITY_ALPHABET_SIZE
from .online_prior import PREV_Q_CONTEXTS, OnlinePriorState


def fuse_batch_logits(prior: OnlinePriorState, logits, previous, cycles):
    """Compute each used context adjustment once, preserving float64 order."""
    keys = (cycles // prior.config.cycle_bin_width) * PREV_Q_CONTEXTS + previous
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # Representative original cycles preserve the BOS validation contract.
    probabilities = prior.probabilities(previous[first], cycles[first])
    adjustment = prior.config.prior_weight * np.log(probabilities * QUALITY_ALPHABET_SIZE)
    return np.asarray(logits, dtype=np.float64) + adjustment[inverse]


def selected_quantized_bits(symbols, cdfs, total):
    """Exact bits from a quantizer-owned CDF; do not rescan all 42 classes."""
    rows = np.arange(symbols.size)
    frequencies = cdfs[rows, symbols + 1] - cdfs[rows, symbols]
    return float(-np.log2(frequencies.astype(np.float64) / total).sum(dtype=np.float64))
