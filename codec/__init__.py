"""Data and codec components for direct, lossless FASTQ quality coding.

The package is intentionally independent of the historical SeqArc/HDF5
probability-matrix pipeline.  Stage A provides the shared batch contract,
strict FASTQ streaming parser, and training-only direct-quality cache.
"""

from .fastq_stream import (
    DEFAULT_BATCH_READS,
    FastqBatch,
    RawFastqRecord,
    iter_fastq_batches,
    iter_fastq_records,
)

__all__ = [
    "DEFAULT_BATCH_READS",
    "FastqBatch",
    "RawFastqRecord",
    "iter_fastq_batches",
    "iter_fastq_records",
]
