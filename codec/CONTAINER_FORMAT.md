# Direct-quality container format draft

Stage A does not write the final codec container. The following contract is
reserved now so later model and entropy-coder work does not silently redefine
the lossless boundary.

- All fixed-width integers use little-endian byte order.
- The fixed header will contain magic bytes, format version, flags, Phred
  offset, quality alphabet size, batch size, total read count, and the number
  of entropy substreams.
- A versioned section table will store byte offsets, byte lengths, uncompressed
  lengths, and checksums for the header, base, plus, and quality streams.
- Header, base, and plus streams use gzip in version 1. Those streams plus
  framing metadata retain the original non-quality field bytes, all four
  lines' LF/CRLF/empty terminators, and the final newline state required to
  reproduce the decompressed FASTQ byte for byte. Quality contents themselves
  are represented by the neural probability model and entropy stream.
- The quality section will record probability-quantization version and total
  frequency, range-coder version, stream framing, and checksums.
- An external model is identified by architecture version plus a cryptographic
  checkpoint hash. A decoder must reject a missing or mismatched model.
- Read lengths or block-level offsets must be available before quality decode.
  Whether version 1 derives them from the base stream or stores a compact index
  remains a Stage D decision and must include its exact overhead in reports.

The initial implementation will use one quality range stream. Multi-stream
framing is reserved but will not be selected until the single-stream codec has
passed byte-exact round-trip tests.
