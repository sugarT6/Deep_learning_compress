# Single-threaded C++ arithmetic encoding

Build once in the Python environment used for compression:

```bash
python -m codec.build_native
```

This needs a Linux C++14 compiler (`c++`, or `CXX`); no Python development
headers, pybind11, Cython or extra pip packages are needed. The shared library
is local and ignored by Git. Its filename includes the CPU architecture and
C++ source hash; rebuild after changing architecture/source. `--force`
rebuilds an existing library.

`--coder-backend native` requires C++ and fails with the build command if
unavailable. `--coder-backend python` selects the reference. The default `auto`
uses native when built, otherwise Python. The log and JSON `coder_backend`
identify the actual backend. Loading happens before model loading/head fitting.

The batch arithmetic loop moves to C++: interval updates, E1/E2/E3
renormalization, pending underflow bits and byte emission. Native int64 NumPy
arrays pass through ctypes without `.tolist()` or per-symbol Python calls.
Floor variable-total intervals and original fixed-total CDF batches both work.
Scalar Python calls remain a reference and may be mixed with native batches.
Framing, CRC32 and finalization remain shared Python code. No threads are
created. 32-bit state uses 64-bit intermediate products (totals <= 2^30).

Quantization, inference, head fitting, gzip, container metadata and protocol
versions are unchanged. Decoding needs no native library. Changing only the
backend produces byte-identical containers for the same input/model/adapter
and runtime.

## Validation and focused timing

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
python -m unittest discover -s tests
```

Native tests skip if not built; build first to exercise both backends. Tests
cover maximum/variable totals, mixed scalar/batch state, long E3 underflow
chains across batches, 1/2^30 intervals, empty streams, read-only/strided/
byte-swapped arrays, invalid inputs, container equality and existing-decoder
FASTQ restoration.

Reuse cached real CNR logits from the chunk benchmark (new output directory):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
python -m codec.benchmark_range_coder \
  codec/runs/floor_workspace_20261012_v1/chunks_v1/real_logits.npz \
  codec/runs/NEW_native_range_benchmark --repeats 8 --copies 4
```

Quantization is outside timers. Both backends warm up; eight rounds alternate
their order and compare complete streams. `--copies 4` replays four 256-read
batches four times into one stream (614400 qualities): this is a coder timing,
not 4096 distinct reads or an end-to-end compression measurement. Timers include
array bridging/output copying; finalization is separate. Reports record
cache/library SHA-256, environment, all trials and stream equality.

On 2026-10-12, on `cyclone001-agent-214` (aarch64), the eight-round medians
for this replay were:

| Backend | Arithmetic encoding seconds |
| --- | ---: |
| Python | 1.714126 |
| C++ | 0.018299 |

The coder stage was 93.7x faster and every stream was byte-identical
(171356 bytes). This does not measure full-file speed. Local report:
`codec/runs/native_coder_20261012_v1/benchmark/report.json`.
All 215 repository tests passed in 27.830 seconds with native built. An
explicit `--coder-backend native` CLI run also encoded nine real 150-base CNR
reads with the saved 2000-step adapter; its complete container was identical
to the previously roundtrip-verified Python reference (`real_cli/report.json`
under the same run directory).

## Full-file timing with the previous training configuration

Run from the repository. Choose an available GPU (`3` matches the previous
command). Use `time_baseline_nodsub`, with 8000 adaptation reads, 2000 updates
and the same 20-second cooperative budget:

```bash
python -m codec.build_native &&
run_dir=$(mktemp -d "$PWD/codec/output/time_baseline_nodsub/CNR_floor_cpp_XXXXXX") &&
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
/usr/bin/time -v -o "$run_dir/walltime.txt" \
python -u -c 'import runpy, torch; torch.set_num_threads(1); torch.set_num_interop_threads(1); runpy.run_module("codec.encode", run_name="__main__")' \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$run_dir/CNR0847462.fqdc" \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --quantizer floor --quantization-total 65536 --coder-backend native \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 \
  --head-steps 2000 --head-max-seconds 20 \
  --save-head-adapter "$run_dir/CNR0847462.adapter.json" --no-progress \
  > "$run_dir/metrics.json" 2> "$run_dir/encode.log"
echo "Results: $run_dir"
```

Compare `encode_seconds`, `encoding_stage_seconds.range_encode`, `coder_backend`
and `head_adaptation.steps_completed`. For a strict backend-only pair, replace
all training/export options in both runs with the same
`--head-adapter PREVIOUS_RUN/CNR0847462.adapter.json`; change only
`--coder-backend python` vs `native`, using fresh outputs. Both omit training;
do not attribute that saved training time to C++.
