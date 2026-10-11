# RX 6800 / gfx1030 — llama.cpp MMVQ Benchmarks and Profiling

**Report date:** 2026-10-11  
**Coverage:** Beginning of this conversation through the recovered October 11 performance-counter analysis. Earlier scheduling results are included from the project continuation roadmap referenced in this conversation.  
**Excluded:** All discussion and implementation of browser/JavaScript conversation export, per request.  
**Status:** Benchmark and profiling measurements preserved; next step is *offline per-dispatch-shape analysis and true GPU ISA extraction*, not another broad profiling run.

## 1. Executive summary

The project optimizes llama.cpp's ROCm/HIP MMVQ (quantized matrix–vector multiplication) kernels for **AMD Radeon RX 6800 (`gfx1030`, RDNA2)** running a **Qwen3.8 27B IQ4_XS GGUF** with an embedded MTP speculative head. The tuned path is particularly relevant to **three output columns (`ncols_dst == 3`)** with MTP depth 2.

**What has been established:**

- The original **1 warp × 1 row** schedule averaged **42.433 tokens/s** in the prior controlled benchmark. **2 warps × 4 rows** averaged **45.245 tokens/s**, a **6.63%** end-to-end improvement, with **1,301/1,301** ROCm0 `MUL_MAT` backend tests passed.
- Subsequent optimized V1 work (Q4_K/Q5_K packed-header handling and IQ3_S sign-multiplication changes, in addition to the 2×4 schedule) was reported around **45.79 tokens/s**, with a **46.005 tokens/s best observed run**. Those values came from previous reported measurements, not a new clean A/B series in this conversation. The later 3,200-token uninstrumented October 11 campaign recorded **45.5297 before** and **45.3609 after**.
- A new Q5_K tile sweep (eight shapes) **failed to beat** the preserved V1 microbenchmark baseline **53.36 / 125.97 µs**; the nominal best 2×4 candidate was **53.55 / 126.23 µs**.
- A 900-token `rocprofv3` runtime trace measured **MMVQ at 75.62% of summed recorded GPU kernel execution time**. **IQ4_XS / 3-column** is the largest individual hotspot (**23.05%** of GPU kernel time), followed by Q4_K/3 (**16.82%**), Q5_K/3 (**8.72%**), IQ3_S/3 (**7.70%**), and a large separate **Q5_K/1-column** workload (~**6.5%**).
- A nine-group hardware-counter campaign successfully wrote **2,165,481 per-metric rows** covering **27 requested metric slots**, recovered from disk despite abnormal profiler shutdown (`-9`). These indicate high memory-unit activity and wave dependency waits for IQ4_XS; high memory activity and low L2 hit rate for the large Q5_K/1 workload; zero recorded scratch allocation for target variants.
- The current evidence supports prioritizing **IQ4_XS load organization / latency hiding / VGPR pressure**, then Q4_K instruction-issue pressure and Q5_K/1 traffic, **rather than speculative Q5_K bit-shift micro-optimizations**.
- No genuine instruction-level ATT trace was collected. An initial purported ISA extract contained **x86 HIP launch stubs**, not device instructions; actual GPU code objects are available locally for offline disassembly. The final offline deep-dive script was provided, but its *output has not yet been supplied in this conversation*.

**Scope caution:** Kernel-time shares are **not equivalent** to fractions of wall-clock time or inference latency. Cross-run trace comparisons are observational, not controlled head-to-head benchmarks.

## 2. Hardware, toolchain and controlled workload

| Item | Configuration / observed state |
|---|---|
| GPU | AMD Radeon RX 6800, Navi 21 / RDNA2, `gfx1030`, 16 GiB |
| Operating system | Native Ubuntu, Linux `7.0.0-38-generic` in October 11 preflight |
| ROCm/HIP | ROCm 10.0.0 |
| Profiling utilities | `rocprofv3` 1.3.5 and `rocprofv3-avail` |
| LLVM toolchain | AMD LLVM 23 |
| llama.cpp checkout | Commit `139997d` |
| Build mode | HIP Release `-O2`, GPU target `gfx1030`, forced MMQ; previous `-O3`/high-parallel builds encountered compiler issues |
| Model | `Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf` (~14.4 GB), embedded MTP |
| MTP | `draft-mtp`, 2 maximum draft tokens; observed `ncols_dst == 3` |
| Controlled benchmark | 3,200 generated tokens, `seed=1234`, baseline profile `T8_TB4_F16`, two-round comparisons when available |
| Inference flags | `-c 8192`, `-ngl 99`, ROCm0, `--fit off`, flash attention enabled, F16 KV, threads 8 / batch threads 4, batch 1024 / microbatch 128, 1 parallel slot |
| Isolation | Bench harness used port 8091; profiling used port 8092, leaving user's normal 8090 undisturbed |

The controlled test prompt was long and complex, with output checks based on full generated text and MTP draft acceptance. A **matching seed does not alone guarantee matching generated output**: inspect hashes and draft counts to detect path changes. The earlier two-round sweep also used per-round seed changes, so within-group hash equivalence is more informative than nominal seeds across all variants.

### 2.1 Historical controlled scheduling matrix

The beginning-of-conversation work inherited a prior scheduling campaign (summarized in `rdna2_mmvq_optimization_roadmap_2026-10-10.md`). The table records its two independent 3,200-token trial means; the 4×8 row was *unverified in the historical summary*, not an established measurement.

| Warps × rows | Trial 1 (t/s) | Trial 2 (t/s) | Mean (t/s) | vs. 1×1 | Backend-test status |
|---|---:|---:|---:|---:|---|
| 1×1 | 42.177 | 42.689 | **42.433** | Reference | Reference |
| 2×1 | 41.686 | 41.881 | 41.784 | −1.53% | Earlier comparison |
| 2×2 | 44.263 | 44.556 | 44.410 | +4.66% | Pass |
| 1×2 | 44.527 | 45.082 | 44.805 | +5.59% | Pass |
| 1×3 | 44.292 | 44.831 | 44.562 | +5.02% | Pass |
| 1×4 | 44.361 | 44.912 | 44.637 | +5.19% | Pass |
| 2×3 | 44.897 | 45.098 | 44.998 | +6.04% | Pass |
| **2×4** | **45.158** | **45.332** | **45.245** | **+6.63%** | **1,301/1,301 passed** |
| 4×2 | 41.906 | 42.613 | 42.260 | −0.41% | Pass |
| 4×4 | 42.593 | 43.353 | 42.973 | +1.27% | Pass |
| 2×8 | 42.327 | 42.398 | 42.363 | −0.17% | Not confirmed in reported output |
| 1×8 | 41.564 | 42.202 | 41.883 | −1.30% | Pass |
| 4×8 | — | — | — | — | Not established by the historical report |

**Scheduling conclusion:** More rows were beneficial initially, but **eight rows** strongly regressed; four warps also underperformed two at four rows. The **2×4** schedule was the *best measured candidate*, not a proof of global optimality. Its specialized modification was confined to the RDNA2 three-column case in `ggml/src/ggml-cuda/mmvq.cu`:

```cpp
// calc_nwarps(...)
if (table_id == MMVQ_PARAMETERS_RDNA2 && ncols_dst == 3) return 2;

// calc_rows_per_block(...)
if (table_id == MMVQ_PARAMETERS_RDNA2 && ncols_dst == 3) return 4;
```

The project explicitly retained separate binary snapshots because the mutable checkout folder name (`llama-rocm-mmvq-2x1`) **does not describe the current compiled scheduling configuration**.

### 2.2 Reported later optimization experiments

| Experiment / modification | Measurement reported in conversation | Decision / qualification |
|---|---|---|
| Original schedule | 42.433 t/s | Baseline |
| 2×2 schedule | 44.410 t/s | Improved |
| 2×4 schedule | 45.245 t/s | Preserved reference |
| Optimized V1: packed Q4_K/Q5_K header handling + IQ3_S sign multiplication | ~45.79 t/s, best observed 46.005 t/s | Retained, but no new controlled paired run available here |
| IQ4_XS ordered lookup | 45.134 t/s | Rejected |
| Q4_K header shuffle | 54.24 → 58.91 µs | Regression, rejected |
| Q4_K DPP reduction | 49.74 → 50.92 µs | Regression, rejected |
| Q4_K hybrid DPP | 50.18 / 116.15 µs | Rejected |
| Q5_K alternative tile shapes | All nominally slower | Rejected |
| Q5_K high-bit (`qh`) decoding | Source inspection only | No patch or improvement established |

Historical values above are retained as **reported results**, not recomputed from the unsupplied raw A/B logs.

## 3. Latest Q5_K tile sweep: direct starting point of this chat

Command executed by user:

```bash
python3 ~/Downloads/run_q5k_tile_sweep.py
```

The microbenchmark explicitly retained the original optimized V1 timing and compared an experimental default plus eight candidate configurations:

| Candidate | Short kernel (µs) | Long kernel (µs) | Mean speedup vs. V1 |
|---|---:|---:|---:|
| **Preserved V1** | **53.36** | **125.97** | **Baseline** |
| Candidate default (existing 2×4) | 53.57 | 126.24 | — |
| 1×2 | 56.36 | 137.12 | −6.73% |
| 1×4 | 53.52 | 127.35 | −0.69% |
| 2×2 | 55.99 | 134.48 | −5.51% |
| **2×4** | **53.55** | **126.23** | **−0.28%** |
| 4×2 | 57.70 | 141.07 | −9.11% |
| 4×4 | 54.16 | 130.20 | −2.36% |
| 8×2 | 59.95 | 142.84 | −11.40% |
| 8×4 | 56.90 | 131.55 | −5.23% |

**Output artifact:** `/home/xu/mmvq-q5k-sweep-results.csv`.

No candidate beat V1. The −0.28% difference for 2×4 is plausibly benchmarking noise and does not establish a genuine regression. Larger warp counts and 2-row tiles tended to regress strongly. Decision: **stop blind Q5_K tile searches**; move to hardware profiling.

### 3.1 Source inspection of Q5_K

Files/regions reviewed in the mutable worktree:

- `ggml/src/ggml-cuda/vecdotq.cuh`, lines ~1021–1120: `vec_dot_q5_K_q8_1`.
- Lines ~564–595: `vec_dot_q5_K_q8_1_impl_vmmq`.
- Low-bit `qs` and high-bit `qh` are loaded in two packed words each; `vh[0] = qh[0] >> bq8_offset` and `vh[1] = qh[4] >> bq8_offset`.
- Under `GGML_USE_HIP && __gfx1030__`, optimized V1 uses a packed 16-byte `q5k_header_words` struct, conditional alignment assumption and `__builtin_memcpy` to read packed `dm`/scale words, then branchless bit-mask operations to select scale and minimum pieces.
- The inner loop unpacks 4 low bits, reconstructs the fifth bit, calculates two `ggml_cuda_dp4a` chains: one weighted quantized dot product, one sum of activation bytes, and accumulates scaled floats before combining the `half2 dm5` terms.

Representative loop from the inspected source:

```cpp
#pragma unroll
for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 = ggml_cuda_dp4a(v0i, u[2*i], ggml_cuda_dp4a(v1i, u[2*i+1], 0));
    const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
}
```

**Interpretation:** A potential optimization is reuse of the activation-only sum across rows, subject to lane mapping, load location, synchronization, and register costs. Merely removing `qh` shifts is **not** justified: they are required for the bit layout unless an equivalent encoding is established. ISA must be examined before attributing cost to them.

## 4. October 11 comprehensive profiling campaign

### 4.1 Rationale and coverage plan

The team chose an evidence-driven campaign instead of modifying more decoding instructions. The planned data included kernel timelines, occupancy, registers, global/L2 traffic, stalls, compute issue, ISA and, if supported, targeted advanced thread tracing (ATT). The aim was to save the raw data in one organized folder for repeatable **offline** examination.

Preflight:

```bash
bash ~/Downloads/gfx1030_profile_preflight.sh
# Output: ~/Downloads/gfx1030-preflight-20261011-142635.tar.gz
# Available-counter inventory lines: 874; archive ~72 KB
```

The initial `/sys/class/drm/card1` clock/utilization errors were for the **secondary GPU**. RX 6800 telemetry was found at **`/sys/class/drm/card2/device`**. The 874 inventory lines should not be read as 874 unique supported metrics.

A preflight inventory and built code-object extraction established tool availability: ROCm 10.0, `rocprofv3` 1.3.5, AMD LLVM 23, `gfx1030`, and two earlier kernel-summary files.

### 4.2 Counter-planning validation fix

The first campaign preflight found **0 accepted counter groups** because of incorrect ordering of the `rocprofv3-avail` global device argument:

```text
rocprofv3-avail: error: unrecognized arguments: -d
```

The argument was moved before `pmc-check`. The corrected dry run accepted **9 groups / 27 requested slots**:

| Group | Requested counters |
|---|---|
| `compute` | `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `SQ_INSTS_SMEM`, `SQ_WAVES` |
| `wave_stalls` | `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES` |
| `memory_issue` | `SQ_INST_CYCLES_VMEM`, `SQ_INSTS_FLAT`, `SQ_INSTS_LDS`, `SQ_INSTS_SMEM` |
| `l2_transactions` | `GL2C_HIT`, `GL2C_MISS`, `GL2C_MC_RDREQ`, `GL2C_MC_WRREQ` |
| `memory_amountsa` | `FETCH_SIZE` |
| `memory_amountsb` | `WRITE_SIZE`, `L2CacheHit` |
| `occupancy` | `MeanOccupancyPerCU`, `OccupancyPercent`, `Wavefronts` |
| `unit_pressure` | `MemUnitBusy`, `WAVE_DEP_WAIT`, `WAVE_ISSUE_WAIT` |
| `lds` | `LDSBankConflict`, `ALUStalledByLDS` |

`SQ_INSTS_SMEM` appears in two groups, so **27 metric slots does not mean 27 distinct metric names**. Passing compatibility validation does not itself prove the counters contain usable measurements; the subsequent recovery verified CSV rows.

### 4.3 Preserved library identity and linker issue

Saved optimized V1 directory:

`/home/xu/mmvq-rdna2-q4q5-iq3s-signmul-bin/`

Its shared-library layout was not initially uniform:

```text
libggml-hip.so           regular 54 MB file, dated Oct 11 13:12
libggml-hip.so.0         symlink -> libggml-hip.so.0.25.3
libggml-hip.so.0.25.3    regular 54 MB file, dated Oct 10 20:34
```

The first two regular files had **different hashes**; merely setting `LD_LIBRARY_PATH` without checking the resolved SONAME would be insufficient proof. The runner recorded the **loaded versioned** file with SHA-256:

`494d7cd83c10dd3aebd899d7ca367c0e631547dfc06d8da80ce0d610bafb7100`

It matched the identified optimized V1 runtime library. Neither saved file was modified in this conversation. Various `*.hipv4-amdgcn-amd-amdhsa--gfx1030` code objects were extracted during preflight. **For future trials, resolve the *actual loaded library* and record its hash rather than inferring it from the directory name.**

### 4.4 Actual campaign run

Executed on Ubuntu:

```bash
python3 ~/Downloads/gfx1030_profile_campaign_v2.py --execute --include-att
```

The runner saved its data under `~/mmvq-profile-campaign/`. The plan attempted uninstrumented baselines before/after, a 900-token kernel trace, nine 400-token counter groups, static code-object extraction and optional ATT. Counter passes altered throughput appreciably; those throughput numbers are **profiling overhead**, not product performance.

| Pass | Requested tokens | Generated | Tokens/s | Elapsed (s) | Profiler exit |
|---|---:|---:|---:|---:|---:|
| Baseline before | 3,200 | 3,200 | **45.5297** | 78.25 | 0 |
| Runtime trace | 900 | 900 | 40.9582 | 120.13 | −9 |
| PMC 01 compute | 400 | 400 | 24.2174 | 115.16 | −9 |
| PMC 02 wave stalls | 400 | 400 | 24.1236 | 115.22 | −9 |
| PMC 03 memory issue | 400 | 400 | 24.2133 | 115.15 | −9 |
| PMC 04 L2 transactions | 400 | 400 | 19.7335 | 118.92 | −9 |
| PMC 05 fetch | 400 | 400 | 19.6506 | 119.00 | −9 |
| PMC 06 write/L2 hit | 400 | 400 | 19.7656 | 118.85 | −9 |
| PMC 07 occupancy | 400 | 400 | 24.2664 | 115.12 | −9 |
| PMC 08 unit pressure | 400 | 400 | 22.5426 | 116.39 | −9 |
| PMC 09 LDS | 400 | 400 | 24.1987 | 115.15 | −9 |
| Baseline after | 3,200 | 3,200 | **45.3609** | 78.41 | 0 |

**Baseline drift:** about **−0.37%** across the campaign, with identical 3,200-token generated-text SHA-256 (`e7668d0b119c1705...`). Both baseline runs recorded **1,915 accepted of 2,565 drafted MTP tokens = 74.66%**. The 400-token PMC passes recorded identical output hashes to each other and **247 accepted of 303 draft tokens**, confirming that the workload stayed aligned for those instrumented runs. The 900-token trace had its own output trajectory/length.

**Abnormal profiling shutdown:** Every profiler-wrapped pass recorded `profile_rc=-9` after the request returned. It was subsequently discovered that counter CSVs had in fact been written before the profiler was forcibly terminated. A successful inference request did not imply clean profiler process termination, but recoverable output was present.

**ATT:** No ATT passes ran: the kernel-name filter sought readable `IQ4_XS`/`Q4_K` strings, whereas reported kernels encoded their quantization using numeric `(ggml_type)` template values. The saved detected-kernel list was empty. Therefore **ATT instruction timing is not available** from this campaign.

**Thermals:** One reported final-baseline peak was ~71 °C. Stable baseline throughput gives no direct evidence of thermal throttling in these tests. Temperature data should not be extrapolated to a guarantee about long continuous runs.

## 5. Kernel timing results: 900-token runtime trace

The trace analyzed in `gfx1030_profile_review_2026-10-11.md` contains approximately **887,693 kernel calls** in **119 kernel variants**, with summed GPU kernel durations of **19,476.68 ms**.

| Kernel / output columns | Calls | Summed GPU time (ms) | Mean duration (µs) | Share of recorded GPU time |
|---|---:|---:|---:|---:|
| **IQ4_XS / 3** | **56,110** | **4,489.97** | **80.02** | **23.05%** |
| **Q4_K / 3** | **42,716** | **3,275.42** | **76.68** | **16.82%** |
| Q5_K / 3 | 17,376 | 1,697.63 | 97.70 | 8.72% |
| IQ3_S / 3 | 15,566 | 1,498.82 | 96.29 | 7.70% |
| Q5_K / 1 — grouped | 771 | 1,274.62 | 1,653.20 | 6.54% |
| IQ3_XXS / 3 | 5,792 | 596.00 | 102.90 | 3.06% |

All MMVQ variants account for about **75.62%** of summed recorded GPU kernel time. The five leading entries account for about **62.83%**. Q5_K/1 above groups more than one similar kernel shape; the most significant single shape has **759 calls, ~1,676.79 µs mean and 6.53%** of the measured GPU kernel time. Avoid conflating the grouped and dominant-shape measurements.

### 5.1 Earlier kernel-trace comparison (observational)

| Kernel | Prior mean (µs) | October 11 mean (µs) | Observed difference |
|---|---:|---:|---:|
| IQ4_XS / 3 | 80.35 | 80.02 | −0.41% |
| Q4_K / 3 | 93.17 | 76.68 | −17.70% |
| Q5_K / 3 | 107.81 | 97.70 | −9.38% |
| IQ3_S / 3 | 118.36 | 96.29 | −18.65% |
| IQ3_XXS / 3 | 101.98 | 102.90 | +0.90% |

These values hint that previous changes benefited Q4_K/Q5_K/IQ3_S more than IQ4_XS. They are **not controlled paired runs**; differing dispatch shapes, model state, or tracing conditions can explain some of the changes. Do not treat these percentages as independently verified speedups attributable to a single patch.

## 6. Counter recovery: 9 passes, 2,165,481 records

The original `review-upload.tar.gz` was size-filtered: files larger than 16 MiB were omitted. That hid the actual `*_counter_collection.csv` outputs in the first upload. The recovered on-disk CSVs range from approximately **42 to 169 MB** each. An offline script collected summaries without running the GPU again:

```bash
python3 ~/Downloads/gfx1030_extract_existing_counters.py
# Output: ~/mmvq-profile-campaign/counter-recovery/counter-recovery-upload.zip
```

The archive confirmed **9/9 readable PMC CSVs**, **2,165,481 per-metric rows**, and 27 requested metric slots. Representative breakdown: groups 01–04 contributed 320,812 rows each, the one-metric `FETCH_SIZE` group 80,203 rows, and the two-metric group 06 160,406 rows. **A counter's row count is not the number of unique inference requests**: rows repeat for each metric and dispatch.

### 6.1 Major hardware metrics by quantization format

The following values are **arithmetic averages over covered dispatches and kernel shapes** in the recovered counters. This is not a time-weighted utilization model; Q5_K/3 particularly spans large grid-size variation.

| Metric | IQ4_XS / 3 | Q4_K / 3 | Q5_K / 3 | Q5_K / 1* | IQ3_S / 3 | IQ3_XXS / 3 |
|---|---:|---:|---:|---:|---:|---:|
| VGPR count | **120** | 88 | 96 | 40 | 112 | 104 |
| SGPR count | 128 | 128 | 128 | 128 | 128 | 128 |
| Workgroup threads | 64 | 64 | 64 | 32 | 64 | 64 |
| Scratch allocation | **0** | 0 | 0 | 0 | 0 | 0 |
| LDS block size (B) | 1,536 | 1,536 | 1,536 | 0 | 1,536 | 1,536 |
| Mean `OccupancyPercent` | **53.35%** | 58.54% | 43.48% | **102.80%†** | 60.98% | 60.95% |
| `MemUnitBusy` | **89.42%** | 84.81% | 73.79% | **99.53%** | 90.55% | 90.01% |
| `WAVE_DEP_WAIT` | **86.27%** | 52.93% | 59.43% | 87.33% | 69.41% | 63.06% |
| `WAVE_ISSUE_WAIT` | 9.73% | **39.93%** | 31.29% | 7.97% | 22.27% | 28.06% |
| `L2CacheHit` | 37.65% | 25.98% | 23.51% | **9.59%** | 27.90% | 30.29% |
| `FETCH_SIZE` (KB per dispatch, mean) | 33,749 | 32,168 | 43,072 | **861,880** | 34,327 | 31,635 |
| `LDSBankConflict` (derived %) | 0.103% | 0 | 0 | 0 | 0 | 0 |

\* Q5_K/1 counter row corresponds to a dominant large workgroup/grid variant and must not be treated as all Q5_K single-column shapes.  
† **The derived 102.80% occupancy value is not a physically literal occupancy.** Its normalization is unreliable for this configuration and should not be used as evidence of >100% active GPU occupancy. Compute true architectural limits from VGPR/LDS/wave allocation and inspect the metric definition before interpreting it.

These counters were acquired in **separate 400-token profiled runs** from the **900-token trace**. Combining their timings and values to infer an exact per-kernel effective bandwidth is not justified without same-pass dispatch matching and unit verification.

### 6.2 Interpretation: IQ4_XS / 3 columns

- Largest quantization hotspot: **23.05%** of measured GPU kernel time.
- **120 VGPRs** vs Q4_K's 88; same reported scratch allocation (**0**), both workgroup size 64 and LDS block size 1,536 B.
- **89.42% memory-unit busy**, **86.27% dependency wait**, and **9.73% issue wait** in the reported profiler categories.
- **37.65% L2 hits** in the averaged data.

**Supported hypothesis:** memory latency/dependencies and register live ranges are worth investigating. High dependency waiting may include memory effects, but is measured at wavefront level; it does *not* mean 86% of device time was completely idle. High memory-unit busy likewise is not proof of saturated DRAM bandwidth. Need the generated AMDGPU ISA and grid-size stratification to distinguish load bottlenecks, cache behavior, decode chains and scheduling limits.

### 6.3 Interpretation: Q4_K / 3 columns

- Second-largest hotspot: **16.82%** of measured GPU kernel time.
- **88 VGPRs**, 0 scratch, 64-thread workgroup.
- **39.93% issue wait** vs IQ4_XS's **9.73%**, with **52.93% dependency wait**.
- Likely fruitful inspection: pipeline issue pressure, decode and dot-product sequence, scheduling between integer ops and loads, and exact machine-instruction footprint.

Do **not** assume wavefront issue waits are all caused by DP4A; the ISA and pipeline counters must substantiate the mechanism.

### 6.4 Interpretation: Q5_K / 1 column

- Separate unusually large kernel with **~1.65–1.68 ms** average calls, ~**6.5%** of GPU kernel time.
- Dominant shape grid size **7,946,240**, workgroup size **32**, VGPR count **40**, LDS block size **0**, 0 scratch.
- **99.53% memory-unit busy**, **9.59% L2 hit**, **~861,880 KB fetched per dispatch** as reported by `FETCH_SIZE`.

These features are **consistent with streaming a very large weight matrix**, making bandwidth, traffic volume, cache line use, and avoidable passes more promising than a single shift or mask optimization. `FETCH_SIZE` is a profiler-derived metric in KB; it should not automatically be interpreted as bytes arriving from VRAM without checking its definition. This workload's performance does not necessarily improve through the 3-column 2×4 schedule.

### 6.5 Other formats, LDS and spills

- IQ3_S/3: 112 VGPRs, ~90.55% memory-unit busy, ~69.41% dependency wait; moderate hotspot at 7.70%.
- IQ3_XXS/3: 104 VGPRs, ~90.01% memory-unit busy; 3.06% hotspot.
- No recorded scratch allocation for principal variants, hence no resource-metadata evidence of spilling; this does not exhaustively rule out every compiler/codegen issue.
- Recorded LDS conflicts and ALU-stalled-by-LDS values were near zero; LDS-bank tuning is not a leading priority.
- Occupancy should not be maximized indiscriminately; higher occupancy only helps when it relieves the actual latency/resource constraint.

### 6.6 Quantitative upside bounds

Given IQ4_XS/3's **23.05%** share of summed GPU kernel durations, **speeding up just that kernel by 20%** gives a theoretical aggregate GPU-kernel-time reduction of:

`0.2305 × (1 − 1/1.20) = 0.0384`, or **~3.84%**.

That is not a prediction of end-to-end server tokens/s; launches, overlapping execution, CPU work, speculation and other kernels can reduce or otherwise complicate the realized gain. This calculation illustrates why prioritization by kernel share is important.

## 7. Static ISA: what was and wasn't recovered

An offline disassembly artifact was reportedly generated (`static/amdgpu-isa-full.txt.gz`, ~178 MB uncompressed) and a first filtered `selected-mmvq-isa.txt.gz` extract was uploaded. **Important correction:** that extract matched host-side `__device_stub__` functions with x86 instructions (`movq`, `callq hipLaunchKernel`) rather than gfx1030 instructions. The presence of matched names does **not** establish device disassembly.

The local code objects extracted from `libggml-hip.so` include files with suffix `hipv4-amdgcn-amd-amdhsa--gfx1030`. Before reasoning about dot-product opcodes and instruction scheduling, confirm **AMDGPU ELF type** (e.g. with `file`/`llvm-readelf`) and inspect with LLVM `llvm-objdump` from the ROCm toolchain. Look for:

- MMVQ template specialization by `ggml_type`, column count, workgroup / grid size.
- Actual `v_dot4*`/packed-dot instructions and their dependency chains.
- VMEM/SMEM loads, load width, coalescing signals from source-to-lane mapping, and `s_waitcnt` placements.
- VGPR/SGPR allocation in code-object metadata, scratch usage, LDS resource requirements.
- Redundant unpack operations, scale/minimum decode lifetime, issue/latency overlap, reuse of activation sums.

The final helper `gfx1030_offline_deep_dive.py` was supplied to perform per-grid-size grouping and true device-code selection, but **no resulting `offline-deep-dive-upload.zip` appears in this conversation**. Thus the deep-dive findings remain **pending**, not completed.

## 8. Artifact inventory and preservation

The following file paths were reported on the user's Ubuntu machine; they are not downloadable through this Markdown document unless manually copied from that machine.

| Path | Role |
|---|---|
| `~/llama-rocm-mmvq-2x1/` | Mutable llama.cpp experimental worktree; folder name not an active variant label |
| `~/llama-rocm-mmvq-2x1/ggml/src/ggml-cuda/mmvq.cu` | RDNA2 MMVQ scheduling specialization |
| `~/llama-rocm-mmvq-2x1/ggml/src/ggml-cuda/vecdotq.cuh` | Q5_K, Q4_K, IQ variants, packed decoding and DP4A |
| `~/mmvq-rdna2-2x4.patch` | Preserved winning schedule patch |
| `~/mmvq-rdna2-2x4-bin/` | Saved reference executable/GPU library |
| `~/mmvq-rdna2-optimized-v1-bin/` | Reported optimized V1 snapshot |
| `~/mmvq-rdna2-q4q5-iq3s-signmul-bin/` | Saved optimized Q4/Q5/IQ3S variant, campaign library resolution |
| `~/mmvq-q5k-sweep-bin/` | Q5_K experimental builds |
| `~/mmvq-q5k-sweep-results.csv` | Full microbenchmark sweep output |
| `~/Downloads/run_q5k_tile_sweep.py` | Q5_K sweep harness |
| `~/Downloads/bench-qwen-rocm-linux.py` | End-to-end controlled inference/hashes harness |
| `~/Downloads/gfx1030_profile_preflight.sh` | ROCm device/counter inventory |
| `~/Downloads/gfx1030-preflight-20261011-142635.tar.gz` | Preflight archive |
| `~/Downloads/gfx1030_profile_campaign_v2.py` | Profiling runner used with `--execute --include-att` |
| `~/mmvq-profile-campaign/` | Raw campaign results, telemetry, code objects, output logs |
| `~/mmvq-profile-campaign/review-upload.tar.gz` | Size-limited profiler review subset |
| `~/Downloads/gfx1030_extract_existing_counters.py` | Offline recovery, no GPU use |
| `~/mmvq-profile-campaign/counter-recovery/counter-recovery-upload.zip` | Recovered metric summaries |
| `~/Downloads/gfx1030_offline_deep_dive.py` | Pending offline per-grid and actual ISA extractor |
| `~/mmvq-profile-campaign/offline-deep-dive/offline-deep-dive-upload.zip` | **Expected output, not yet supplied** |

Companion analyses produced in this conversation:

- `gfx1030_profile_review_2026-10-11.md`: trace, profiler errors, timing and recoverability.
- `gfx1030_counter_analysis_2026-10-11.md`: recovered PMC counters, limits and decisions.
- `gfx1030_hotspot_comparison.csv`: per-quant aggregate with timing and hardware metrics.

### Important data hygiene

- **Retain full raw CSVs on Ubuntu:** uploaded compact reports are enough to guide first-pass decisions but cannot replace all dispatch-level data.
- Preserve the exact runtime `libggml-hip.so.0.25.3` (SHA-256 above), launch executable and build flags together.
- Keep timing traces, counter CSVs, disassembly and output hash records distinguishable by pass type and generated-token length.
- Treat a bad profiler process exit as **collection-integrity warning**, not automatic counter invalidation or success.
- Do not confuse host ISA stubs, toolchain metadata and actual gfx1030 kernel ISA.

## 9. Decision log and proposed next optimizations

| Priority | Action | Reason | Required validation |
|---|---|---|---|
| **P0** | Finish offline deep-dive on existing CSV/code objects | Already paid for broad profiling; avoid another GPU session | Per-kernel × `Grid_Size` × workgroup counter table and real `elf64-amdgpu` ISA |
| **P1** | Investigate IQ4_XS/3 memory loads, waits, register lifetimes | Top hotspot (23.05%), 120 VGPR, high memory activity and dependency waiting | One isolated source change, correct codegen, short kernel microbench and same-output 3,200-token A/B |
| **P1** | Analyze Q4_K/3 issue pressure | 16.82% hotspot and 39.93% wave issue wait | ISA-based hypothesis followed by controlled test |
| **P2** | Investigate Q5_K/1 streaming traffic | ~6.5% hotspot; large dispatch, very low L2 hit rate | Focus on traffic volume and shape, not 3c tile settings |
| **P2** | Activation-only `dot2` reuse / cross-row common subexpression | Potential redundant calculation in Q5_K inner loop | Check lane mapping, value sharing, extra registers and result equivalence |
| **P3** | Targeted ATT only if needed | Broad ATT never ran, but PMC + ISA may already answer key questions | Specific unresolved question that requires instruction-level time correlation |
| **Defer** | New blind tile sweeps, arbitrary `qh` shift patches, LDS bank tuning | Prior regressions or weak bottleneck evidence | Reopen only on compelling new data |

### Repeatable candidate acceptance contract

1. **Freeze inputs:** same model, exact GGUF, context 8192, MTP depth 2, CPU threads, attention, KV cache, sampler and seed.
2. **Freeze binary identity:** compile flags and git diff, complete `bin/` snapshot, `LD_LIBRARY_PATH` and the actual resolved HIP SONAME/hash.
3. **Correctness:** run ROCm0 `test-backend-ops test -b ROCm0 -o MUL_MAT`; record pass/fail and any unavailable tests.
4. **Performance:** compare identical short kernel benchmark shapes, then controlled 3,200-token application runs; use per-run tokens/s, MTP draft counts and full output hashes. Interleave order or use repeated rounds when differences are ~1% or below.
5. **Promotion:** require evidence of a material end-to-end benefit without unexplained output or correctness regressions. A one-off best-run result is not sufficient to replace the preserved baseline.
6. **Avoid unnecessary profiling:** reuse saved raw metrics and disassembly; add a new GPU profiling pass only when a specific open question cannot be resolved offline.

## 10. Outstanding uncertainties and explicit exclusions

- We do **not** yet have the user's final offline deep-dive results; ISA-level conclusions remain tentative.
- No successful ATT data was obtained during the campaign.
- Mean counters aggregate different grid sizes and separate runs; they do not prove exact per-shape DRAM bandwidth saturation or dependency root cause.
- Occupancy >100% is an artifact/normalization issue, not a physical state.
- Scheduling sweep's best 2×4 mean is from an earlier controlled series, and optimized V1's ~45.79 / 46.005 values are prior reported outcomes. Do not silently combine them into a statistically clean single tuning sweep.
- The Q5_K tile microbenchmark times (53.36 / 125.97 µs) are **different workloads** from the traced Q5_K/3 mean (~97.70 µs). They are not inconsistent or interchangeable.
- The recovered `FETCH_SIZE` per-dispatch values are metric averages and need careful normalization before interpreting achieved DRAM bandwidth.
- The reported 4×8 historical pending status is not evidence that it never ran; no subsequently verified 4×8 benchmark was included in the source material used for this report.
- There is no evidence in this conversation that the broad profiling effort caused GPU damage or thermal throttling.

## 11. References and source lineage

**Primary evidence for this report**

1. This conversation's user-supplied Q5_K sweep terminal output, Q5_K source excerpts, rocprof preflight/dry-run outputs, library listing and campaign console excerpt.
2. `rdna2_mmvq_optimization_roadmap_2026-10-10.md` (prior work summary, available in the user's file library): full scheduling sweep, validation notes, build/benchmark protocol.
3. Uploaded `gfx1030-preflight-20261011-142635.tar.gz`, `review-upload.tar.gz`, and `counter-recovery-upload.zip` with recovered CSV-derived tables.
4. Generated `gfx1030_profile_review_2026-10-11.md`, `gfx1030_counter_analysis_2026-10-11.md`, `gfx1030_hotspot_comparison.csv` and counter recovery manifests.

**Interpretation references (not independent verification of these user's numbers)**

- AMD ROCm profiler documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- AMD GPU performance model: https://rocm.github.io/rocprofiler-compute/performance_model.html
- AMD GPUOpen occupancy explanation: https://gpuopen.com/learn/occupancy-explained/
- LLVM AMDGPU disassembly: https://llvm.org/docs/CommandGuide/llvm-objdump.html

---

**Bottom line as of 2026-10-11:** Preserve optimized V1, do **not** resume Q5_K tile/bit-shift guesswork, and exploit the recovered profiling dataset. The highest-leverage next move is an **offline analysis of IQ4_XS/3 actual gfx1030 ISA and per-shape memory/dependency counters**, then one targeted optimization candidate with controlled correctness and throughput tests.
