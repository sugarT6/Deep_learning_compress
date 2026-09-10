"""Data and codec components for direct, lossless FASTQ quality coding.

The package is intentionally independent of the historical SeqArc/HDF5
probability-matrix pipeline.  It contains the direct FASTQ/cache contract,
minimal causal model, deterministic probability quantizer, and integer range
coder.  The entropy coder remains independent of the neural model until the
container integration stage.
"""

from .fastq_stream import (
    DEFAULT_BATCH_READS,
    FastqBatch,
    RawFastqRecord,
    iter_fastq_batches,
    iter_fastq_records,
)
from ._range_common import (
    InvalidCDFError,
    InvalidRangeStreamError,
    RANGE_CODER_VERSION,
    RANGE_STREAM_HEADER_BYTES,
    RangeCodingError,
    TruncatedRangeStreamError,
)
from .probability_quantization import (
    QUANTIZATION_VERSION,
    TOTAL,
    logits_to_cdf,
    probabilities_to_cdf,
)
from .range_decoder import RangeDecoder
from .range_encoder import RangeEncoder

__all__ = [
    "DEFAULT_BATCH_READS",
    "FastqBatch",
    "InvalidCDFError",
    "InvalidRangeStreamError",
    "QUANTIZATION_VERSION",
    "RANGE_CODER_VERSION",
    "RANGE_STREAM_HEADER_BYTES",
    "RawFastqRecord",
    "RangeCodingError",
    "RangeDecoder",
    "RangeEncoder",
    "TOTAL",
    "TruncatedRangeStreamError",
    "iter_fastq_batches",
    "iter_fastq_records",
    "logits_to_cdf",
    "probabilities_to_cdf",
]
