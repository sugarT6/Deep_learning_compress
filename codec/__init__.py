"""Data and codec components for direct, lossless FASTQ quality coding.

The package is intentionally independent of the historical SeqArc/HDF5
probability-matrix pipeline.  It contains the direct FASTQ/cache contract,
minimal causal model, deterministic probability quantizer, integer range
coder, and byte-exact FASTQ container contract.
"""

from .fastq_stream import (
    DEFAULT_BATCH_READS,
    MAX_BATCH_READS,
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
from .container import (
    CONTAINER_FORMAT,
    CONTAINER_VERSION,
    ContainerError,
    ContainerIntegrityError,
    TruncatedContainerError,
    read_container,
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
    "CONTAINER_FORMAT",
    "CONTAINER_VERSION",
    "ContainerError",
    "ContainerIntegrityError",
    "DEFAULT_BATCH_READS",
    "MAX_BATCH_READS",
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
    "TruncatedContainerError",
    "TruncatedRangeStreamError",
    "iter_fastq_batches",
    "iter_fastq_records",
    "logits_to_cdf",
    "probabilities_to_cdf",
    "read_container",
]
