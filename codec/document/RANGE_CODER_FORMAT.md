# Deterministic probability and range-stream format, version 1

Stage C is deliberately independent of the neural model and the future FASTQ
container. It defines one sequential entropy stream. Every symbol may use a
different synchronized integer CDF.

## Probability quantization

The quality support is Q0 through Q41. Quantization version 1 uses:

```text
QUALITY_ALPHABET_SIZE = 42
TOTAL = 65536 = 2^16
frequency[q] >= 1
cdf[0] = 0
cdf[42] = TOTAL
```

Probabilities and logits are converted to finite CPU float64 values. Logits
use stable softmax after subtracting their maximum. Probability weights are
scaled by their maximum before normalization, avoiding overflow without
changing their ratios.

Every quality id first receives frequency one. The remaining `TOTAL - 42`
units are apportioned as follows:

1. compute each normalized probability's quota;
2. assign the floor of every quota;
3. distribute leftover units by descending fractional remainder;
4. break exact remainder ties by ascending quality id.

NaN, Inf, negative probability weights, zero total probability, a shape other
than exactly 42, noninteger `TOTAL`, and `TOTAL < 42` are errors. Version 1
also limits `TOTAL` to `2^30`, matching the safe total for the range coder's
32-bit state. The canonical default is `2^16`; other totals must be recorded
explicitly by a future container.

## Integer range state

The coder uses an inclusive 32-bit interval:

```text
low  = 0
high = 2^32 - 1
range = high - low + 1
```

For CDF interval `[cdf[s], cdf[s+1])` and `total = cdf[-1]`:

```text
new_low  = low + floor(range * cdf[s]     / total)
new_high = low + floor(range * cdf[s + 1] / total) - 1
```

All interval calculations use integers. A CDF must start at zero, contain only
integers, increase strictly, and have total no greater than `2^30`.

Renormalization follows the standard E1/E2/E3 cases with half=`2^31`,
quarter=`2^30`, and three-quarter=`3*2^30`:

- interval wholly below half: emit 0 and the complements of deferred bits;
- interval wholly above half: emit 1 and deferred complements, then subtract
  half;
- interval inside the middle half: defer one underflow bit and subtract
  quarter;
- otherwise stop renormalizing.

After any subtraction, `low` and `high` are shifted left once and the low bit
of `high` is filled with one. Output bits are packed most-significant first.

For a nonempty sequence, finalization increments the deferred-bit count and
emits 0 when `low < quarter`, otherwise 1, followed by deferred complements.
Only the last output byte is zero-padded. An empty sequence has zero payload
bits.

## Standalone range-stream frame

All integers are little-endian. The 24-byte header is:

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 4 | magic/version `QRC1` |
| 4 | 8 | decoded symbol count, unsigned 64-bit |
| 12 | 8 | meaningful arithmetic payload bits, unsigned 64-bit |
| 20 | 4 | CRC32 of `magic + symbol_count + bit_count + payload` |
| 24 | variable | arithmetic payload bytes |

The physical payload length must equal `ceil(payload_bits / 8)`, unused low
bits in the last byte must be zero, and trailing bytes are forbidden. CRC32 is
for accidental corruption detection, not authentication.

The decoder initializes its 32-bit code value from the first 32 meaningful
payload bits. A finalized short stream is extended with logical zero guard
bits; those zeros are not stored. Successful termination means decoding
exactly the header-declared symbol count and calling `finish()`. Asking for an
additional symbol or finishing early is an error. A truncated header/payload,
bad checksum, invalid padding, or malformed CDF is rejected.

The actual standalone byte size includes the 24-byte frame plus final byte
padding. Reports must keep quantized theoretical bits, meaningful arithmetic
payload bits, and full serialized stream bits as separate quantities.
