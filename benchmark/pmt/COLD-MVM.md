# Four-MVM cold-path benchmark

The target workload has four FP32 dense MVMs with output-by-input dimensions
30,000×30,000, 30,000×5,000, 5,000×30,000, and 30,000×20,000. The latency budget
is approximately **500 µs for all four operations**, not per operation. Matrix
storage alone is 7.2 GB per invocation; initialization and upload of fixed weights
are excluded. "Cold path" identifies this workload, not a cold CUDA context or
first-use cache measurement.

`benchmark-cold-pipeline.cu` measures an **all-inputs-ready bundle**, not a claimed
AO dependency graph. The four independent outputs are all ready at the end of
each sample. An application with dependencies between these MVMs still needs an
integrated measurement using its actual equations. Do not sum individual P99s
and label that a pipeline P99.

## Build and run

Use CUDA 12.4 or newer and cuBLAS. The build needs no StarPU or Chameleon.

```bash
bash build-cold-pipeline.sh ./benchmark-cold-pipeline sm_80  # A100
# Use sm_90 on H100 / GH200.
# If necessary set NVCC, CUDA_HOST_CXX, and CUBLAS_LIB_DIR explicitly.

# Four GPUs, 1,000 measured frames per configuration, all three distributions.
MVM_GRAPH_MODES=1 bash run-cold-pipeline.sh ./benchmark-cold-pipeline results 1000 '1 2 4'
```

Use a single Slurm task with four GPUs visible. On Perlmutter, select A100 80 GB
nodes and record CPU affinity. On DeltaAI, use the native aarch64 build and its
cuBLAS library directory. Each platform must produce its own hardware record.

Direct executable arguments:

```text
GPUs samples split graph host [raw.csv] [column_groups=1]
```

- `split=0`: cuBLAS FP32 GEMV in pedantic math mode.
- `split=1,2,4,8,16`: fused dispatch of four custom GEMVs, aligned vector loads,
  warp reduction, and optionally a deterministic local split-K reduction.
- `graph=1`: one pre-instantiated graph per device and input frame.
- `host=0`: vectors already on the GPUs; includes host dispatch, all GPU work,
  distribution-specific reduction/communication, and completion synchronization.
- `host=1`: also includes pinned-host input copies and output copies. Fixed
  matrix weights remain resident on the GPUs in both scopes.
- `column_groups=1`: four row partitions, no cross-GPU arithmetic reduction.
- `column_groups=2`: a 2×2 row/column grid. Two partial outputs per row partition
  are combined on its owner GPU before completion.
- `column_groups=4`: four column partitions. All four partial outputs are combined
  on one owner GPU. Peer copies, event waits, and reduction are timed.

`MVM_HOST_SCOPES`, `MVM_GRAPH_MODES`, and `MVM_SPLITS` restrict a sweep. All
variants retain FP32 weight storage, input/output, and accumulation. No tensor
core reduced-precision mode, low-rank approximation, or compression is implied.

Set `MVM_PEER_DIRECT=1` to replace peer-copy staging with a fused reduction
kernel that reads peers' partial vectors through CUDA peer access. Event waits
still enforce completion before the remote reads. Compare this mode with the
default staged copies on the same allocation; its benefit depends on topology.

## Validation and timing

Eight changing synthetic input vectors avoid validating only one repeated input.
Before timing, frames 0 and 7 check every local output against cuBLAS and 17 rows
per matrix/shard against independent CPU FP64 dots reconstructed from global
indices. After timing, the actual final output (including graph replay and peer
reduction) is checked against a fresh full cuBLAS reference and sampled global
CPU FP64 dots. The gates are relative L2 ≤5e-6 and scaled absolute error ≤1e-4,
where the scaling denominator is `max(1, abs(reference))`. These are implementation
checks on synthetic data, not AO control-quality acceptance criteria.

Each case performs 300 unmeasured warmups. Wall-clock timing starts before any
device is submitted and stops after every participating stream completes.
The CSV reports P50/P95/P99, maximum, RMS deviation from the mean, and count
above 500 µs. Raw frame samples are retained. Setup, graph instantiation,
validation, and writing the raw CSV are outside the measured interval.

Select configurations with a screening sweep, then repeat the baseline and
selected candidate in alternating order on the same four GPUs. Keep failed
validations and outliers visible; never report an unvalidated configuration as
a successful optimization. A selection-sweep minimum is not a repeatability claim.

## Existing evidence

The earlier independent-MVM A100 measurements are in
`log/pmt/mvm/perlmutter-cold-61023602/results.csv` under the workspace root.
Their row-major host-roundtrip P50s sum to 1,266.574 µs; their GPU event P50s
sum to 1,094.880 µs. These are sums of separate experiments, not joint samples.
The joint benchmark's `host=0` scope is wall-clock time and must not be equated
with the older GPU-event-only scope.

## A100 optimization results, 2026-11-11

Perlmutter job `61126138`, node `nid008253`, four A100 SXM4 80 GB GPUs:
screening selected **row partitioning, split-K=2, CUDA Graph** for both timing
scopes. GPU peer access is irrelevant to this selected row-only configuration.
The baseline is cuBLAS with CUDA Graph and identical row partitioning.

Each mode was then measured three times with 6,000 frames per run, alternating
baseline/selected, selected/baseline, baseline/selected on the same allocation.
The table uses the median of the three per-run P50s, not the minimum sample.

| Scope | Baseline P50 | Selected P50 | Reduction | Selected per-run P99 range |
| --- | ---: | ---: | ---: | ---: |
| GPU-resident inputs, wall clock | 1,090.813 µs | 1,010.649 µs | 7.35% | 1,016.790–1,017.701 µs |
| Pinned-host input/output roundtrip | 1,156.449 µs | 1,077.227 µs | 6.85% | 1,088.198–1,089.951 µs |

All samples still exceed 500 µs. There are occasional millisecond-scale outliers:
selected GPU-resident run RMS values are 20.656, 21.985, and 14.016 µs; host
roundtrip RMS values are 23.316, 16.682, and 21.594 µs. Do not describe this as a
hard real-time deadline guarantee. These are cold-path measurements; they do not
replace the separate hot-path 150 µs / jitter <20 µs acceptance test.

The selected implementation's full-output relative L2 difference from cuBLAS is
6.09e-7, and sampled CPU FP64 scaled error is 6.82e-6. Actual graph replay and
host-returned results passed the same checks. This validates the synthetic
benchmark implementation, not controller accuracy on real matrices.

Other distributions were tested with both staged peer copies and direct peer
reads. Direct reads improved communication overhead, but row partitioning still
won the screening comparison. Earlier screening jobs `61125845` and `61126028`
used different nodes; use the same-allocation repeats above for speedup claims.

The devices report a 1,593,000 kHz memory clock and 5,120-bit memory bus. The
nominal four-device transfer rate from `2 × clock × bus_width / 8` is 8.156 TB/s.
Reading 7.2 GB once at that rate takes about **883 µs**, before dispatch,
reductions, and vector transfers (and ignoring the small cache contribution).
The selected GPU-resident result corresponds to about 7.124 TB/s of effective
matrix bandwidth. Reaching 500 µs on these A100s requires changing the amount of
matrix data read, precision, or mathematical structure, not just repartitioning
the same dense FP32 matrices.

Evidence under the workspace root:

- `log/mvm-opt/pmt/peer-61126138/`: screening, device topology, binary hash.
- `log/mvm-opt/pmt/repeat-61126138/`: alternating repeats and raw frame samples.
- `log/mvm-opt/pmt/benchmark-cold-pipeline-v3.cu`: exact source snapshot.

Summarize repeats with:

```bash
python3 summarize-cold-pipeline.py /path/to/log/mvm-opt/pmt/repeat-61126138
```

The GH200/H100 investigation now uses the source-backed dependent controller
benchmark rather than another independent bundle sweep. See
`log/mvm-opt/status.json` for its live job handle. No H100 performance result is
established by these A100 measurements.

### Audit correction: column-partition host-transfer scope

Source snapshots v2 and v3 mistakenly conditioned the input copy on the
partial-output copy flag. For `host=1,column_groups>1`, input copies were omitted
and unnecessary partial-output copies were included. **Those screening rows are
invalid host-roundtrip measurements and must not be used.** The row-only
configuration (`column_groups=1`) and all `host=0` measurements are unaffected,
including every alternating-repeat result in the table above. The comparison
between row and column partitions in host-roundtrip scope requires remeasurement.

The corrected source always copies host inputs when `host=1` and copies only
final outputs. Device input buffers are zeroed after eager validation, so
omitted input transfers produce an incorrect result and fail replay validation.

`test-cold-transfers.cu` directly tests the suppressed-partial-output path on a
small single-GPU problem, with both eager and graph execution, cuBLAS and custom
kernels, and two input frames. It fails against archived v3 and passes against
the corrected source. Results are in `log/mvm-opt/local/transfer-regression-v*.txt`.

Corrected full-size job `61126372` completed successfully. It again selected
row partitioning and split-K=2. Same-node alternating repeats gave GPU-resident
P50 1,090.922 → 1,010.659 µs and host-roundtrip P50 1,153.144 → 1,076.075 µs.
Use `log/mvm-opt/pmt/repeat-61126372/comparison.json` and
`log/mvm-opt/pmt/corrected-61126372/` for corrected comparisons across distributions.

For the source-backed dependent ATA update and hybrid A/G/L-row, K-column
distribution, see [COLD-CONTROLLER.md](COLD-CONTROLLER.md). Its timings include
state/command redistribution and are a different scope from this independent
bundle benchmark.
