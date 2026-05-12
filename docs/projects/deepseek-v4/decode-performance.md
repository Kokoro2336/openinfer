# DeepSeek V4 Decode Performance

**Created**: 2026-05-12
**Status**: active

## TL;DR

This document consolidates the DeepSeek V4 decode work that moved fixed long decode from the `~108-113ms/token` band to the current `31.99-34.22ms/token` W13 runtime branch, with shared-quant-only validation at `33.33-34.29ms/token` and prior same-code fast runs at `32.75-33.90ms/token`. The retained changes are grouped MoE pointer caching, rank-worker placement, removal of hot temporary zero-fill, rank-owned decode scratch, caller-owned grouped FP4 workspace, shared W1/W3 activation quantization, W13 grouped FP4 runtime launch, and benchmark/counter instrumentation. The active MoE goal is stable sub-`30ms/token` decode by reducing local expert launch/intermediate work without bs=1 specialization. Exact E2E remains `20/20`, and the fixed bench token hash remains `6346f03343d75a65`.

The retained team lessons are more important than the discarded attempt logs: compare identical token traces, separate NCCL wait from transfer, treat capacity and logical length separately, keep MoE semantic zero on device, and prove allocation cleanup with application-visible CUDA API counters rather than nsys attribution alone.

## Baseline And Result

Use this fixed decode bench for comparable DeepSeek V4 direct-runtime work:

```bash
target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  request \
  --prompt-len 1 \
  --output-len 160 \
  --warmup 2 \
  --iters 3 \
  --seed 42
```

| Milestone | Fixed long decode | Key change |
| --- | ---: | --- |
| AG/RS grouped MoE baseline | `107.92-112.61ms/token` | GPU AG/RS path existed, but repeated pointer-array setup and per-token allocations remained. |
| Grouped MoE pointer cache | `83.37-89.65ms/token` | Cache grouped FP4 expert weight/scale pointer arrays per rank worker. |
| Rank-worker affinity | `72.88-73.60ms/token` | Reduce rank-arrival skew before f32 collectives. |
| Remove hot temporary zero-fill | `63.35-64.51ms/token` | Fully overwritten hot temporaries allocate uninitialized storage. |
| Rank-owned decode scratch | `34.34-35.36ms/token` forced NUMA, `32.75-33.90ms/token` same-code fast band | Move hot intermediate storage to per-rank scratch and remove grouped FP4 C-side growth-cache workspace. |
| Final PR validation | `35.253ms/token` | After review fixes: dynamic NUMA topology, buffer-derived capacity checks, and `_ptsz` counter separation. |
| Shared W1/W3 act quant | `33.330ms`, repeated `34.289ms` | Decode scratch W1 and W3 reuse one TileLang `act_quant_k4096`; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| W13 grouped runtime launch | text `34.22ms`, JSON `31.986ms` | W1 and W3 share one TileLang grouped FP4 launch after shared activation quant; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |

Final PR validation on 5090:

| Metric | Value |
| --- | ---: |
| steady TPOT avg | `35.253ms` |
| steady TPOT p50 | `34.800ms` |
| steady TPOT p95 | `37.335ms` |
| first decode avg | `33.743ms` |
| generated-token hash | `6346f03343d75a65` |
| exact E2E | `20/20` |

## Retained Design

### Grouped MoE pointer cache

Each persistent rank worker builds a `MoeGroupedPtrCache` once after context binding. The cache stores per-layer GPU arrays for local expert weight pointers and scale pointers for W1/W2/W3 grouped FP4 linears. Decode and prefill MoE paths pass this cache to grouped FP4 local expert execution.

This removed repeated host vector construction and H2D pointer-array copies from every grouped FP4 call. The grouped FP4 kernels did not become materially faster in nsys; the improvement showed up as a shorter MoE reduce-scatter synchronization window, which points to lower rank-arrival skew.

### Rank-worker placement

Rank workers are pinned before CUDA work begins. The final PR path resolves topology dynamically:

1. CUDA driver `cuDeviceGetPCIBusId` maps CUDA ordinal to PCI bus id.
2. `/sys/bus/pci/devices/<pci>/numa_node` maps PCI to NUMA node.
3. `/sys/devices/system/node/node<numa>/cpulist` supplies target CPUs.
4. The target list is intersected with the process's allowed cpuset.
5. Missing topology, empty intersection, or `pthread_setaffinity_np` failure panics.

Do not encode ordinal assumptions such as `GPU0..3 -> NUMA0`. A review caught that earlier draft; it matched 5090 but was still a machine-specific fact in runtime logic. Also avoid CUDA runtime topology calls here: `cudaDeviceGetPCIBusId` loaded an incompatible `libcudart` on 5090, while the CUDA driver API path worked.

5090 final pin evidence:

| GPU ordinal | PCI bus | NUMA | pinned CPU |
| --- | --- | ---: | ---: |
| `0` | `0000:16:00.0` | `0` | `0` |
| `1` | `0000:27:00.0` | `0` | `1` |
| `2` | `0000:38:00.0` | `0` | `2` |
| `3` | `0000:5a:00.0` | `0` | `3` |
| `4` | `0000:98:00.0` | `1` | `36` |
| `5` | `0000:a8:00.0` | `1` | `37` |
| `6` | `0000:c8:00.0` | `1` | `38` |
| `7` | `0000:d8:00.0` | `1` | `39` |

### Rank-owned decode scratch

`RankDecodeScratch` is created once per rank worker and reused by decode commands. The current direct scheduler still sends one token per rank command, but the scratch design is capacity-based and should not assume batch size one in API contracts.

| Area | Scratch owner | Note |
| --- | --- | --- |
| Token upload | `RankDecodeScratch::token_ids` | Replaces per-token `clone_htod(&[token_id])` with H2D copy into rank-owned storage. |
| Entry hidden | `DecodeEntryScratch` | Embedding and HC expand outputs are fully overwritten. |
| HC pre/post | `HcPreNormScratch`, `HcPostScratch` | HC pre-state and layer outputs reuse rank-owned buffers; HC post layer output uses ping-pong slots to avoid adjacent-layer aliasing. |
| Attention | `AttentionProjectionScratch`, `AttentionIndexScratch`, `AttentionAuxScratch`, `AttentionOutputScratch` | Active ratio `0` and ratio `4` decode paths use capacity buffers with logical lengths passed separately. |
| Shared expert | `SharedExpertScratch` | Fixed-shape gate/up/activation/out storage. |
| MoE AG/RS | `MoeAgRsScratch` | Hidden/token all-gather, route buffers, compact maps, expert intermediates, partial routed output, local reduce-scatter output, routed+shared output. |
| Grouped FP4 workspace | `MoeAgRsScratch::{fp4_act_workspace,fp4_act_scale_workspace}` | Caller-owned workspace avoids the C-side grouped FP4 growth-cache/mutex path. |
| Final logits | `FinalLogitsScratch` | HC head, final norm, local logits, and gathered logits are reusable. |

### Capacity and logical length

Reusable scratch must not use mutable `seq_len` as allocation capacity. The final code exposes buffer-derived `seq_capacity()` helpers for `Bf16HiddenStates`, `F32HiddenStates`, and `HcHiddenStates`. Scratch-backed `*_into` operators check capacity from storage length, then set `seq_len` to the logical length for this decode step.

NCCL calls must use logical prefix slices, not whole-capacity buffers:

- BF16 hidden all-gather sends `hidden_dim * local.seq_len` and receives `hidden_dim * gathered_seq_len`.
- F32 reduce-scatter sends `hidden_dim * global.seq_len` and receives `hidden_dim * local_seq_len`.
- U32 token all-gather and ratio-4 indexer score all-reduce slice to logical prefixes.

### MoE dynamic content

MoE route values remain dynamic. Static storage does not mean static route content:

- route weights and indices change per token/layer.
- compact maps and `expert_indptr` depend on the route.
- local expert counters/cursors need semantic initialization.

Storage is capacity-based. Semantic clears remain inside `deepseek_moe_local_mapping_cuda` for counters/cursors/indptr and mapping sentinels.

## Active MoE Sub-30 Work

### Goal

Drive decode MoE from roughly `17-19ms/token` toward `10-12ms/token`, enough to move overall fixed long decode from roughly `35ms/token` to stable `28-30ms/token`. Optimizations must remain batch-general, keep route/tile scheduling on GPU, and preserve exact E2E `20/20`.

### Attempt: fuse SwiGLU with W2 activation quant

The local expert decode path originally did:

```text
W1 grouped FP4 GEMM -> gate BF16
W3 grouped FP4 GEMM -> up BF16
SwiGLU clamp -> activated BF16
TileLang act_quant(activated) -> FP8 activation + E8M0 scales
W2 grouped FP4 GEMM
```

The attempted branch replaced the decode scratch W2 input path with:

```text
W1 grouped FP4 GEMM -> gate BF16
W3 grouped FP4 GEMM -> up BF16
fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
W2 grouped FP4 GEMM
```

The fused quant keeps the old semantic order by rounding the SwiGLU result to BF16 before FP8 quantization. That matters for exact output: skipping the BF16 intermediate rounding would not be the same operator.

The first implementation used one CTA per `(row, 128-column group)`. That was the wrong GPU shape for this workload: with about `64` routed rows and `16` scale groups, it launched about `1024` CTAs per W2 quant, while the TileLang act_quant shape is `ceil(rows / 32) * 16`, about `32` CTAs. This explains why launch-count fusion did not translate into stable TPOT.

Evidence from the row-per-CTA version:

| Run | Result |
| --- | --- |
| Exact E2E on 5090 | `All 20 DeepSeek V4 exact cases passed` |
| Fixed bench run 1 | steady TPOT avg `31.922ms`, p50 `31.417ms`, p95 `33.812ms`, hash `6346f03343d75a65` |
| Fixed bench run 2 | steady TPOT avg `34.939ms`, p50 `34.388ms`, p95 `37.047ms`, hash `6346f03343d75a65` |
| Fixed bench run 3 | steady TPOT avg `34.572ms`, p50 `34.022ms`, p95 `36.767ms`, hash `6346f03343d75a65` |

Keep/drop decision: do not treat the `31.922ms` run as evidence. The repeated runs show the row-per-CTA version is not a stable win. The useful retained lesson is that a fused kernel must preserve the original row-block quantization shape, not merely reduce kernel launches.

Follow-up repair used a 4-warp row-block kernel and restored exact E2E `20/20`, but the fixed long bench token hash changed from `6346f03343d75a65` to `00fd2d772e4b8886` and TPOT regressed to `35.554ms`. Drop decision: do not retain this path. It changes long decode behavior even though the short exact suite passed.

### Retained: share W1/W3 activation quant

The retained local-expert change is deliberately narrower:

```text
Before:
W1: act_quant(expanded_input) + grouped FP4 GEMM
W3: act_quant(expanded_input) + grouped FP4 GEMM

After:
act_quant(expanded_input) once
W1 grouped FP4 GEMM reuses act/scale
W3 grouped FP4 GEMM reuses act/scale
```

This preserves the original TileLang activation quantization bit path and only removes duplicate work on the identical W1/W3 input. W2, SwiGLU, combine, route planning, and collectives are unchanged.

Validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | steady TPOT avg `33.330ms`, p50 `32.858ms`, p95 `35.274ms`, hash `6346f03343d75a65` |
| fixed bench run 2 | steady TPOT avg `34.289ms`, p50 `33.979ms`, p95 `36.852ms`, hash `6346f03343d75a65` |

Keep decision: retain. The gain is modest and still noisy, but the method is sound: fixed token trace, exact-safe, and it removes one W1/W3 activation quant launch per layer without introducing new quant math. The next MoE step should be W13 grouped GEMM or GPU active tile list, not more W2 quant fusion.

### Retained: W13 grouped FP4 GEMM runtime launch

W13 was first evaluated as a pure operator change before touching runtime. The TileLang generator now emits:

```text
deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096
```

It is generated from the existing `N=2048,K=4096` FP4 grouped GEMM, but launches `grid.x=32` blocks:

```text
blockIdx.x 0..15   -> W1 pointer arrays -> gate output
blockIdx.x 16..31  -> W3 pointer arrays -> up output
```

The C++ tool `pegainfer-kernels/tools/deepseek_v4/w13_grouped_fp4_bench.cu` links the generated TileLang object directly and compares:

```text
baseline: grouped_gemm(W1) + grouped_gemm(W3)
candidate: grouped_w13_gemm(W1, W3)
```

Fuzz uses BF16 random input, TileLang `act_quant_k4096`, random FP4 bytes and bounded E8M0-like scale bytes, expert-major `expert_indptr` with empty experts, and bitwise BF16 output comparison for `gate` and `up`.

Verified compile command shape:

```bash
OUT_DIR=$(find target/release/build/pegainfer-kernels-* -maxdepth 1 -type d -name out -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120 \
  -I/usr/local/cuda/include \
  pegainfer-kernels/tools/deepseek_v4/w13_grouped_fp4_bench.cu \
  "$OUT_DIR/libkernels_cuda.a" \
  -lcudart \
  -o /tmp/w13_grouped_fp4_bench
```

5090 microbench results:

| Rows | Experts | Fuzz | Baseline two GEMMs | W13 one GEMM | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| `64` | `4` | PASS | `0.126931ms` | `0.063485ms` | `1.999x` |
| `64` | `8` | PASS | `0.126988ms` | `0.122769ms` | `1.034x` |
| `64` | `16` | PASS | `0.300209ms` | `0.236817ms` | `1.268x` |
| `96` | `4` | PASS | `0.126970ms` | `0.063513ms` | `1.999x` |
| `96` | `8` | PASS | `0.126975ms` | `0.122862ms` | `1.033x` |
| `96` | `16` | PASS | `0.300129ms` | `0.238110ms` | `1.260x` |
| `160` | `4` | PASS | `0.127066ms` | `0.122937ms` | `1.034x` |
| `160` | `8` | PASS | `0.128706ms` | `0.124902ms` | `1.030x` |
| `160` | `16` | PASS | `0.303603ms` | `0.239735ms` | `1.266x` |
| `256` | `4` | PASS | `0.127242ms` | `0.124970ms` | `1.018x` |
| `256` | `8` | PASS | `0.246201ms` | `0.184480ms` | `1.335x` |
| `256` | `16` | PASS | `0.314086ms` | `0.251675ms` | `1.248x` |

Runtime integration replaces the two W1/W3 grouped GEMM launches after shared activation quant with the W13 launcher:

```text
act_quant(expanded_input) once
W13 grouped FP4 GEMM writes gate and up outputs
W2 path unchanged
```

The first runtime attempt failed exact E2E with `CUDA_ERROR_NOT_SUPPORTED` at decode layer 0. The cause was launch-wrapper setup inside CUDA graph capture: the new W13 kernel had not been used before capture, so its first `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, 98304)` could return `cudaErrorNotSupported`. The W13 wrapper now treats `cudaErrorNotSupported` like the existing `cudaErrorInvalidValue` tolerance and lets the actual launch result decide correctness.

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench text run | steady TPOT avg `34.22ms`, p50 `33.77ms`, p95 `36.53ms`, first decode avg `32.94ms` |
| fixed bench JSON run | steady TPOT avg `31.986ms`, p50 `31.458ms`, p95 `34.052ms`, first decode avg `30.544ms`, hash `6346f03343d75a65` |

Interpretation: W13 is exact at the operator level and in runtime, but the speedup depends heavily on routed-row distribution, local expert count, and run-to-run system noise. It is not automatically a `2x` W1/W3 win; some shapes mainly save launch overhead, while others are dominated by the expanded grid and grouped scheduling. Keep the runtime change because it removes one launch per layer and preserves the fixed token trace, but do not count the `31.986ms` run as stable sub-32 evidence until repeated long benches confirm it.

## Rejected Patterns

These are worth remembering because they looked plausible:

| Attempt | Result | Lesson |
| --- | --- | --- |
| Fuse final HC head plus RMSNorm | Exact-safe but regressed TPOT | Saving small launches can lose to worse reduction/kernel shape. |
| Reuse deterministic window top-k across layers | Exact-safe, no stable long-bench win | Launch-count reduction alone is weak evidence. |
| Fuse KV RoPE plus no-PE quant | Exact-safe, regressed short decode | Combining tiny kernels can hurt scheduling/occupancy. |
| Hand-written decode HC mixes kernel | Exact-safe, slower than cuBLAS path | cuBLAS small GEMV remained better on this shape. |
| Isolated final logits scratch | Correct but noisy/regressive in repeated runs | Isolated storage movement near sampling boundary did not address the dominant per-layer allocation/skew structure. |
| Host-sized active-tile count for grouped MoE | Not used | Pulling active counts D2H would reintroduce hot-path synchronization. |

## Profiling And Benchmark Rules

### Token trace first

Always compare generated-token hashes before comparing TPOT. DeepSeek V4 routing and expert balance depend on token sequence. The bench JSON now records per-iteration timing and generated-token trace.

### NCCL wall is wait-inclusive

Nsight Systems NCCL kernel wall time includes rank-arrival waiting. Treat NCCL rows as synchronization-window evidence unless rank-arrival skew and post-arrival tail have been separated. The rank-affinity work was selected because corrected f32 all-reduce grouping showed attention hidden all-reduce dominated by arrival skew, not post-arrival NCCL tail.

### Allocation proof

Full-process nsys attribution was not reliable enough for allocation proof:

- nsys-only `cuMemAllocAsync` attribution did not reconcile with application-visible symbols.
- CUDA event tracing can distort API counts.
- NCCL wall can dominate profile views while reflecting upstream skew.

The retained allocation evidence combines source-level inventory with `tools/cuda_api_counter.c`, an `LD_PRELOAD` counter that covers directly linked runtime/driver symbols and CUDA driver function-table lookup via `cuGetProcAddress`.

| API group | Baseline | Current |
| --- | ---: | ---: |
| `cudaMalloc` calls | `12944` | `136` |
| `cudaFree` calls | `12848` | `32` |
| `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` | noisy nsys-only attribution | `0/0/0` in counter |
| `cudaMallocAsync/cudaFreeAsync/cudaMemsetAsync` | not used | `0/0/0` |
| `cuGetProcAddress` replacements | not covered | `0` |

The counter exports base and `_ptsz` wrappers separately for `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async`. Do not share one stored real function pointer across base and `_ptsz` variants.

## Remote Workflow Notes

Remote test syncs should use touched-file `rsync -azR`. A full repository rsync with delete/excludes stalled for about 10 minutes during this work. Also, `cargo check` does not rebuild already-built release binaries; rebuild `deepseek_v4_e2e` and `bench_serving` before trusting remote validation.

Verified command set for this PR:

```bash
cargo fmt --check
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
cargo check --release -p pegainfer-server --features deepseek-v4
gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl
```

## Validation

Local:

- `cargo fmt --check`
- `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`
- `cargo check --release -p pegainfer-server --features deepseek-v4`
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers
- `git diff --check`
- pre-commit hooks on commit, including clippy

5090:

- `cargo fmt --check`
- `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`
- `cargo check --release -p pegainfer-server --features deepseek-v4`
- release `deepseek_v4_e2e`: `All 20 DeepSeek V4 exact cases passed`
- release fixed bench log `/tmp/dsv4_pr_driver_numa_bench.log`: steady TPOT avg `35.253ms`, p50 `34.800ms`, p95 `37.335ms`, first decode avg `33.743ms`, hash `6346f03343d75a65`
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers

The benchmark process still prints the existing NCCL communicator abort panic during shutdown after JSON output and scheduler exit. Track that as shutdown cleanup, not decode TPOT evidence.

## Follow-ups

- Fix NCCL communicator shutdown.
- Move DeepSeek V4 off the temporary direct runtime into the scheduler/executor shape used by the rest of the engine.
- Revisit CUDA graph capture after pointer stability is broad enough.
- Keep MoE active-expert/tile-list work separate from allocation scratch; the next MoE win is likely reducing empty CTA/kernel work, not more host allocation cleanup.
