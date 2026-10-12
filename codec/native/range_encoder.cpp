// Single-threaded QRC1 batch kernel. Framing and finalization stay in Python.
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {
constexpr std::uint64_t half = 1ULL << 31;
constexpr std::uint64_t quarter = 1ULL << 30;
constexpr std::uint64_t three_quarters = 3 * quarter;

// All fields are uint64_t to keep the ctypes ABI independent of C padding.
struct State {
    std::uint64_t low, high, pending, byte, used, bit_count;
};

struct Writer {
    State& state;
    std::uint8_t* output;
    std::size_t capacity, size = 0;

    bool write(std::uint64_t bits, unsigned count) {
        state.byte = (state.byte << count) | bits;
        state.used += count;
        state.bit_count += count;
        if (state.used == 8) {
            if (size == capacity) return false;
            output[size++] = static_cast<std::uint8_t>(state.byte);
            state.byte = state.used = 0;
        }
        return true;
    }

    bool emit(unsigned bit) {
        if (!write(bit, 1)) return false;
        while (state.pending) {
            const unsigned take = static_cast<unsigned>(
                std::min(state.pending, 8 - state.used));
            if (!write(bit ? 0 : ((1ULL << take) - 1), take)) return false;
            state.pending -= take;
        }
        return true;
    }
};

struct Intervals {
    const std::int64_t *lower, *upper, *totals;
    bool get(std::size_t i, std::int64_t& lo, std::int64_t& hi,
             std::int64_t& total) const {
        lo = lower[i]; hi = upper[i]; total = totals[i];
        return true;
    }
};

struct CDFs {
    const std::int64_t *symbols, *cdfs;
    std::size_t columns;
    std::int64_t total;
    bool get(std::size_t i, std::int64_t& lo, std::int64_t& hi,
             std::int64_t& denominator) const {
        const auto symbol = symbols[i];
        if (symbol < 0 || static_cast<std::uint64_t>(symbol) >= columns - 1)
            return false;
        const auto row = cdfs + i * columns;
        lo = row[symbol]; hi = row[symbol + 1]; denominator = total;
        return true;
    }
};

// Status: 0 success, 1 invalid/collapsed interval, 2 output capacity exceeded,
// 3 invalid symbol. Commit the state only after the entire batch succeeds.
template<class Source>
int encode(State* state, const Source& source, std::size_t count,
           std::uint8_t* output, std::size_t capacity, std::size_t* written) {
    State current = *state;
    Writer writer{current, output, capacity};
    for (std::size_t i = 0; i < count; ++i) {
        std::int64_t lo, hi, total;
        if (!source.get(i, lo, hi, total)) return 3;
        if (lo < 0 || hi <= lo || total < hi ||
            total > static_cast<std::int64_t>(quarter)) return 1;
        const auto width = current.high - current.low + 1;
        const auto new_low = current.low + width * lo / total;
        const auto high_end = current.low + width * hi / total;
        if (high_end <= new_low) return 1;
        current.low = new_low;
        current.high = high_end - 1;
        while (true) {
            unsigned bit;
            if (current.high < half) {
                bit = 0;
            } else if (current.low >= half) {
                bit = 1;
                current.low -= half; current.high -= half;
            } else if (current.low >= quarter && current.high < three_quarters) {
                ++current.pending;
                current.low = (current.low - quarter) << 1;
                current.high = ((current.high - quarter) << 1) | 1;
                continue;
            } else {
                break;
            }
            if (!writer.emit(bit)) return 2;
            current.low <<= 1;
            current.high = (current.high << 1) | 1;
        }
    }
    *state = current;
    *written = writer.size;
    return 0;
}
}  // namespace

extern "C" {
unsigned fqrc_abi_version() { return 1; }

int fqrc_encode_intervals(State* state, const std::int64_t* lower,
                          const std::int64_t* upper, const std::int64_t* totals,
                          std::size_t count, std::uint8_t* output,
                          std::size_t capacity, std::size_t* written) {
    return encode(state, Intervals{lower, upper, totals}, count, output, capacity, written);
}

int fqrc_encode_cdfs(State* state, const std::int64_t* symbols,
                     const std::int64_t* cdfs, std::size_t count,
                     std::size_t columns, std::int64_t total,
                     std::uint8_t* output, std::size_t capacity, std::size_t* written) {
    return encode(state, CDFs{symbols, cdfs, columns, total}, count, output, capacity, written);
}
}
