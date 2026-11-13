# Dependent four-MVM controller benchmark

The local `ata` checkout on branch `DARE_opt`, commit
`5cbad07e83fceadc86e093ec6ee65e918a22ba16`, implements this update in
`ataSupervisor.py:856`:

```text
x_next = A x + L s - G u_prev
u_next = K x_next
```

`u_prev` corresponds to the supplied `um1` history input. The benchmark does not
change ATA's command-history convention or simulate the optical feedback loop.
The nominal matrices match the user's four requested dimensions:

| Matrix | Output × input |
| --- | --- |
| A | 30,000 × 30,000 |
| G | 30,000 × 5,000 |
| K | 5,000 × 30,000 |
| L | 30,000 × 20,000 |

The [requirements email](https://mail.google.com/mail/u/?authuser=yuxi.hong.research%40gmail.com#all/1a1bdd4914a2d1b8)
sets approximately 500 µs for the four cold-path MVMs. The
[follow-up](https://mail.google.com/mail/u/?authuser=yuxi.hong.research%40gmail.com#all/1a1bebe9c0ee2068)
confirms FP32 is acceptable and says FP16 needs validation. This experiment keeps
FP32. The email does not specify the inter-frame hot/cold schedule; the ATA
dependency graph is source-backed, but must not be presented as a verified
implementation of that separate scheduling protocol.

## Distributions

`benchmark-cold-controller.cu` compares three complete, dependent steps:

- **All rows (`layout=0`)**: A, G, and L produce disjoint new-state slices. Each
  GPU gathers the complete new state, then computes its assigned rows of K.
  The command slices are gathered onto every GPU.
- **Hybrid (`layout=1`)**: A, G, and L use the same row distribution. K is split
  by columns matching those state slices. Each GPU immediately computes its
  partial full command using its local new state. Afterward, GPUs gather state
  slices and sum command partials through peer access.
- **Packed hybrid (`layout=2`)**: prepack `[A,L,-G]` once, and concatenate inputs
  `[x,s,u_prev]`. One dense MVM directly produces each local new-state slice,
  followed by the same column-distributed K. The matrix byte count is unchanged;
  packing removes intermediate state products and reduces input-copy and kernel
  counts. It changes FP32 reduction order, so its numerical checks are explicit.

All layouts finish with **complete new state and command vectors on every GPU**.
Thus a faster result cannot come from omitting the redistribution needed for
the next update. Cross-device event waits, peer reads, vector sums, and all
completion synchronization are timed. The hybrid mode moves the state-gather
dependency off the path before K; it currently performs final redistribution
after local K completion rather than overlapping it on an extra stream.

`host=0` starts with all three external inputs (x, s, u_prev) resident. `host=1`
additionally copies them from pinned host buffers and returns the complete
state and command to pinned host buffers. This deliberately broad roundtrip
scope is not the narrower deployment boundary where only s arrives from the
CPU. Fixed weights stay GPU resident.

Each sample uses one of eight independently specified input triples. Outputs
are not fed back through random, potentially unstable synthetic matrices. This
measures one step of the correct dependency graph, not closed-loop stability.

## Build and run

Keep `benchmark-cold-controller.cu` and `benchmark-cold-pipeline.cu` together;
the controller reuses the matrix initializer and FP32 GEMV kernels. Build with
CUDA and cuBLAS, using `sm_80` for A100 or `sm_90` for H100/GH200:

```bash
nvcc -O3 -std=c++17 -arch=sm_80 benchmark-cold-controller.cu -lcublas -o benchmark-cold-controller
bash run-cold-controller.sh ./benchmark-cold-controller results 1000 6000
```

The runner screens cuBLAS and custom split-K=1/2/4, all distributions and both
timing scopes. It then rotates row-cuBLAS, hybrid-cuBLAS, and the selected
configuration through three 6,000-frame repeats on the same allocation. The
two cuBLAS distributions isolate the distribution change from custom-kernel
improvements. Each run has 300 unmeasured warmups.

Direct executable arguments:

```text
GPUs samples split graph host layout [raw.csv] [states measurements actuators]
```

Optional dimensions support small correctness tests. Production-size timing
claims require the default 30,000 / 20,000 / 5,000 dimensions and four GPUs.

The custom kernel also supports 4, 8 (default), or 16 rows per CUDA block via
`-DMVM_ROWS_PER_BLOCK`. Each row still uses one warp and the same FP32 reduction;
only the block grouping changes. To screen this parameter without changing a
queued executable, build separate binaries from one source snapshot:

```bash
for rows in 4 8 16; do
    bash build-cold-controller.sh controller-r$rows sm_90 $rows
done
bash run-cold-block-tuning.sh ./controller block-results 1000 6000
python3 summarize-cold-controller.py block-results
```

The tuning runner screens both hybrid layouts and split-K=1/2/4, then repeats
the selected combination against row and hybrid cuBLAS in a rotated order.
Its CSV records `rows_per_block`; older results implicitly used 8. Small local
checks of all three block sizes, three layouts, both host scopes, and three
split factors passed at dimensions 301/203/53 (54 cases). Those checks verify
numerics and partial blocks only, not target-size throughput.

## Numerical checks

Before timing, frames 0 and 7 check complete state and command vectors on
**every GPU** against a fresh cuBLAS execution of the dependent graph. Sampled
state rows are independently reconstructed on the CPU in FP64; sampled command
rows use CPU FP64 dots against the produced full state. After timing, the actual
final graph output receives the same checks. Host-returned vectors must match
device output exactly. Host-input device buffers are cleared after eager
validation to detect omitted timed H2D copies.

Synthetic implementation gates are relative L2 ≤1e-5 for state and command,
and sampled CPU scaled absolute error ≤1e-3 (`max(1,abs(reference))` denominator).
Two dependent reductions amplify absolute rounding differences, so these are
explicitly distinct from the independent-GEMV checks. They are not AO scientific
acceptance thresholds. Real-matrix control quality remains a separate test.

## Four-A100 evidence

Job `61126770` completed on `nid008233` with four A100 SXM4 80 GB GPUs. All
three layouts and cuBLAS/custom kernels passed validation. The same-allocation
rotated repeats contain 18,000 measured frames per configuration and scope.

| Scope | All-row cuBLAS P50 | Hybrid cuBLAS P50 | Selected P50 | Selected pooled P99 | Selected layout |
| --- | ---: | ---: | ---: | ---: | --- |
| GPU-resident inputs | 1,128.291 µs | 1,115.636 µs | 1,040.270 µs | 1,049.178 µs | Hybrid, split-K=2 |
| Full host roundtrip | 1,163.458 µs | 1,151.014 µs | 1,069.948 µs | 1,082.131 µs | Packed hybrid, split-K=2 |

P50 values are medians of three per-run P50s. Changing only the cuBLAS data
distribution reduces P50 by about 1.1%; the selected implementations reduce it
by 7.80% / 8.04% relative to all-row cuBLAS. Packing did not improve the
GPU-resident winner, but did improve the host-roundtrip result. All samples
still exceed the **500 µs** target. Maximum selected latencies reach 2,986.563 /
3,303.814 µs; these measurements do not establish a hard real-time guarantee.

Selected full-output relative L2 errors are at most 5.325e-7 / 6.181e-7 for
state/command in the hybrid case, and 7.932e-7 / 8.565e-7 for packed hybrid.
The earlier dependent run `61126624` independently selected the same unpacked
hybrid configuration before packing was added.

The block-size sweep `61127542` also completed successfully, screening 40
configurations and then taking three 6,000-frame repeats per mode and scope.
It again selected **8 rows/block, split-K=2**: unpacked hybrid for GPU-resident
inputs and packed hybrid for host roundtrip. The selected P50s were
1,040.301 / 1,070.128 µs and pooled P99s were 1,049.058 / 1,082.593 µs.
All selected samples exceeded 500 µs. Testing 4 and 16 rows per block did not
produce a better screened configuration; the repeat results corroborate the
previous A100 result rather than establish another speedup. This run's exact
sources, binaries, hardware record, numerical checks and raw timings are in
`log/mvm-opt/pmt/controller-v3-blocks/`; its comparison is
`results-61127542/comparison.json`.

Evidence under the workspace root:

- `log/mvm-opt/pmt/controller-v2/results-61126770/comparison.json`
- `log/mvm-opt/pmt/controller-v2/results-61126770/repeat/` (raw frame samples)
- `log/mvm-opt/pmt/controller-v2/` (exact source snapshots and executable)

Regenerate the comparison with:

```bash
python3 summarize-cold-controller.py /path/to/results-61126770
```

The user reaffirmed the 500 µs requirement and requested DAI testing. The
same complete sweep and repeated comparison are submitted on four GH200s;
track the current job in `log/mvm-opt/status.json`. Do not infer GH200 results
from these A100 measurements.

A separate two-minute DAI job screens the three block sizes with 500 measured
frames per configuration (`MVM_SCREEN_ONLY=1` in `run-cold-block-tuning.sh`).
It keeps the original queued experiment intact and can potentially fit a shorter
scheduler gap. Screening results alone do not replace rotated repeats.
