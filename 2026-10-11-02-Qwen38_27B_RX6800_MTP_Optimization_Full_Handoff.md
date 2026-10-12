# Qwen3.8 27B / Radeon RX 6800 — MTP synchronization, transfers, and ROCm optimization: research handoff

**Research date:** 2026-10-11  
**Scope:** Linux ROCm llama.cpp MTP2 on AMD Radeon RX 6800 (`gfx1030`), with prior Windows/Linux optimization context.  
**Document status:** Detailed synthesis of saved benchmark results, *older* ROCm trace/counter captures, GPU ISA/resource audits, and the user's latest `rg` output from their current local source tree. **No code modification or new profiling was performed for this review.**  
**Research collaborators:** GPT, Gemini, DeepSeek, occasionally Claude.  

> **Request to the next model:** Treat this as an experimental handoff, not a request to start over. Prior kernel sweeps, HIP Graphs checks, ISA load-width checks, and register-spill checks are already complete. Determine the smallest *new* experiment that can conclusively attribute MTP logits/hidden-state transfers and CPU waits in the exact local llama.cpp build. Preserve the known-correct decoder, acceptance/rejection probabilities, token positions, and KV state. Do not call summed HIP synchronization time “wasted time”; do not claim redundant Q8 quantization without tensor identity evidence. Distinguish archived profiling data from the **current official 45.723 t/s benchmark**.

## 1. Executive summary

The **latest live Linux development build** generates **45.723 tokens/s mean** over three successful 1,000-token MTP2 runs; the pooled acceptance count is **1,790 accepted / 2,411 drafted = 74.24%**. Its source is modified relative to Git commit `139997d`, so that commit identifier alone is **not** a complete description of the build.

The **older, separate ROCm tracing campaign** recorded **36,905 `hipStreamSynchronize` calls**, but only **371 exceeded 20 ms**. Those 371 calls contributed **19.320 of 21.609 seconds** of cumulative synchronization API duration. **370/371 immediately followed a `hipMemcpyAsync` whose correlated GPU operation was device-to-host (D2H).** Nearly all long waits overlap substantial GPU kernel execution; deleting a CPU wait would not delete the GPU work or satisfy sampling dependencies.

A second pattern comprises **725 D2H copies on stream 3**, followed by approximately **2.8-ms draft-like synchronization cycles** containing typically **43 GPU kernels** and ~**2.56 ms** of kernel execution. Its count is almost twice the ~370 long target-like cycles, **consistent with MTP2**, but still not definitive phase attribution because the archived trace lacks C++ call stacks and transfer sizes.

The current local source excerpt supplied by the user identifies **asynchronous retrieval of `t_logits` and `t_h_nextn`** from `src/llama-context.cpp`. This is a plausible explanation for paired D2H transfers during target verification; **buffer identity for any particular trace record remains unconfirmed**. The next step is **readback call-site/byte-size attribution**, not another nine-counter profiling sweep or an untested removal of synchronization.

## 2. Reproducibility and provenance — two different builds

| Property | Current authoritative live development benchmark | Archived profiling campaign (diagnostic only) |
|---|---|---|
| Platform | Linux; ROCm 10.0; Radeon RX 6800 (`gfx1030`, 16 GiB) | Linux; ROCm profiler; same GPU family |
| llama-server | `/home/xu/ROCm-LLM/development/build/bin/llama-server` | `/home/xu/mmvq-rdna2-q4q5-iq3s-signmul-bin/llama-server` per `campaign-parameters.json` |
| Version | `0.5.0-dev`, build 1, Git base `139997d` **plus local changes** | Earlier optimized binary, **not identical** |
| Model | `/home/xu/ROCm-LLM/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf` | Same GGUF basename on Windows NTFS-mounted path |
| Model quantization | UD-IQ4_XS | UD-IQ4_XS |
| Speculation | `draft-mtp`, maximum 2 draft tokens, draft KV f16/f16 | MTP-enabled; same broad research target |
| Context / GPU offload | 8,192 / `ROCm0`, `-ngl 99`, `--fit off` | Campaign-specific; consult saved command JSON |
| Context threads / batch | `-t 8 -tb 4 -b 1024 -ub 128` | Campaign-specific |
| Target KV | f16/f16 | Campaign-specific |
| Other flags | `-fa on`, `-np 1`, `--jinja` | Campaign-specific |
| Definitive benchmark | Three **unprofiled**, 1,000-token runs | Separate 3,200-token baseline and instrumented runs |
| Interpretation | **Authoritative throughput reference** | **Diagnostic kernel / API / memory attribution only** |

**Current result directory on Ubuntu:** `/home/xu/ROCm-LLM/benchmarks/results/speculate-20261011-202008/`; CSV `results.csv`.  
**Development source:** `/home/xu/ROCm-LLM/development/source/`.  
**Current shared object:** `/home/xu/ROCm-LLM/development/build/bin/libggml-hip.so.0.25.3` (do not substitute archived baseline libraries).

### Current official MTP2 throughput

| Run | Seed | Generated tokens | Tokens/s | Accepted / drafted | Draft acceptance rate | Completion hash (prefix) | Status |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 1234 | 1,000 | 45.346 | 593 / 811 | 73.12% | `edd11c79537494c2` | OK |
| 2 | 1235 | 1,000 | 45.284 | 593 / 811 | 73.12% | `9ea431712f68d984` | OK |
| 3 | 1236 | 1,000 | 46.539 | 604 / 789 | 76.55% | `7c18a1c1808f2ed0` | OK |
| **Aggregate** | — | **3,000** | **45.723 arithmetic mean** | **1,790 / 2,411** | **74.24% pooled** | — | **3/3 OK** |

- Sample standard deviation of throughput: **~0.707 t/s**; observed range **45.284–46.539 t/s**.
- Benchmark script: `~/ROCm-LLM/benchmarks/scripts/bench-qwen-rocm-linux.py`; phase `speculate`, profile `MTP2`, `--rounds 3 --max-tokens 1000 --port 8091` with explicit `--server` and `--model` paths.
- Runs use **different seeds**; hash differences are expected and are not automatically evidence of a correctness defect.
- The script also sets its own reasoning/sampling options; preserve them for subsequent comparable A/B measurements.
- **No direct controlled MTP2-vs-MTP0 speedup is established by these three MTP2-only trials.**

### Previous campaign integrity check — archived profiling **only**

| Diagnostic run | Output tokens | Integrity / profiler status | Role in this report |
|---|---:|---|---|
| `baseline-before` | 3,200 | Output hash `e7668d0b119c1705…` | Campaign integrity check, **not** a performance reference |
| `baseline-after` | 3,200 | Same output hash | Campaign integrity check, **not** a performance reference |
| Hardware-counter collection, 9 passes | 400 each | Same 400-token output hash; each profiler process reported exit `-9` | Kernel/memory counter attribution only |

All nine counter CSVs were reported present and parseable in the previous offline recovery (**2,165,481 records**, 27 requested metric slots). Abnormal profiler termination means these are diagnostic captures, not clean performance trials. **Historical tokens-per-second values are deliberately omitted: the sole authoritative inference throughput number is the current development build's 45.723 t/s.**

## 3. Capture scope and limits

| Dataset | File | Coverage | Important limitation |
|---|---|---|---|
| HIP runtime API timeline | `mtp-traces.tar.gz` → `trace_hip_api_trace.csv` | **199,454 API calls**, ~29.47-s timestamp span | Functions, timestamps, correlation IDs; **no C++ call stacks** |
| GPU kernel timeline | `mtp-traces.tar.gz` → `trace_kernel_trace.csv` | **887,693 kernel launches**, **119** distinct kernel names | GPU kernels; summed durations may overlap across streams |
| GPU copies | `mtp-copy-traces.tar.gz` → `trace_memory_copy_trace.csv` | **5,234 GPU copy events** | Direction, stream, correlation, timestamps; **not payload size or tensor name** |
| Copy summary | `mtp-copy-traces.tar.gz` → `trace_memory_copy_stats.csv` | H2D/D2H aggregate durations | Measures GPU copy duration, not host waiting |
| General campaign summary | `review-upload.tar.gz` | Timeline, telemetry, kernel/API summary, campaign parameters | Large per-dispatch counter CSVs excluded from lightweight review archive |
| Hardware counter recovery | `counter-recovery-upload.zip`, prior counter reports | Nine counter groups and quantization-type aggregates | Counter runs differ from 900-token GPU timeline; profiler perturbs execution |
| Current source excerpt | User's `rg` result in chat | Exact **local** `llama-context.cpp` line numbers, shown below | Does **not** show full `common/speculative.cpp` consumer and sync call stacks |

**Timestamp units:** trace CSV timestamps and durations are in nanoseconds; tables use µs/ms/s as labeled. API duration is time within a host API call, GPU kernel duration is device work, and GPU copy duration is device transfer. **Do not add these categories to estimate wall time.**

## 4. GPU kernel budget — archived profiled trace

Total **887,693** GPU kernel launches with **19.476678 s** of *summed* device kernel durations.

| Kernel group | GPU launches | Summed GPU runtime (s) | Fraction of summed kernel time |
|---|---:|---:|---:|
| `mul_mat_vec_q` / MMVQ | **190,081** | **14.884365** | **76.42%** |
| `mul_mat_q` | 3,024 | 1.917184 | 9.84% |
| `concat_non_cont` | 17,760 | 0.460523 | 2.36% |
| `gated_delta_net_cuda` | 17,808 | 0.348800 | 1.79% |
| `rms_norm_f32` | 120,820 | 0.336580 | 1.73% |
| `quantize_q8_1` | **190,081** | 0.276683 | 1.42% |
| `flash_attn_tile` | 7,031 | 0.256733 | 1.32% |
| `cpy_scalar` | 60,455 | 0.194408 | 1.00% |

**Interpretation:** MMVQ dominates *recorded GPU kernel time*, not necessarily 76.42% of application wall time. A previously documented **75.62%** share came from a **different capture**; do not blend those two percentages.

### Largest quantized MMVQ contributors

| Kernel shape | Launches | GPU time (ms) | Mean per launch (µs) | Fraction of all kernel time |
|---|---:|---:|---:|---:|
| IQ4_XS / 3 columns (`ggml_type 23`) | 56,110 | 4,489.97 | 80.02 | 23.05% |
| Q4_K / 3 columns (`ggml_type 12`) | 42,716 | 3,275.42 | 76.68 | 16.82% |
| Q5_K / 3 columns (`ggml_type 13`) | 17,376 | 1,697.63 | 97.70 | 8.72% |
| IQ3_S / 3 columns (`ggml_type 21`) | 15,566 | 1,498.82 | 96.29 | 7.70% |
| Q5_K / 1 column (aggregated templates) | 771 | 1,274.62 | 1,653.20 | 6.54% |
| IQ3_XXS / 3 columns (`ggml_type 18`) | 5,792 | ~596.00 | 102.90 | 3.06% |

**Research status:** All the leading formats have been explored in previous experiments. This report does **not** recommend repeating IQ4_XS ordered lookup, Q4_K scheduling, Q5_K tile shape, or IQ3_S sign-mask sweeps. Q5_K 1-column contains heterogeneous grids and is not one uniform performance case.

## 5. HIP runtime API summary

| HIP API | Calls | Sum of API time (ms) | Median API time (ms) | Interpretation |
|---|---:|---:|---:|---|
| `hipStreamSynchronize` | **36,905** | **21,609.298** | 0.0002 | Dominated by waiting for prior GPU work/results |
| `hipGraphLaunch` | 1,434 | 1,032.909 | 0.0744 | May include HIP runtime and profiling overhead |
| `hipMemcpyAsync` | 13,955 | 1,030.721 | 0.0037 | API calls not equal to physical copy events |
| `hipGraphExecUpdate` | 16 | 536.839 | 0.1833 | Includes expensive outlier updates; no proven steady-state speedup |
| `hipLaunchKernel` | 43,693 | 465.219 | 0.0034 | Only explicit runtime launches; graph executions contain many kernels |
| `hipHostMalloc` | 8 | 105.291 | 4.1714 | Initialization/host allocation, not assumed per-token cost |
| `hipGraphInstantiate` | 3 | 22.528 | 7.6022 | Setup/warmup |

**Crucial:** `hipStreamSynchronize` cumulative time is **not** 21.61 seconds of overhead we can remove. Its host thread waits while actual kernels are executing. `hipGraphLaunch` being called 1,434 times while 887,693 kernels execute is compatible with graph execution; launch counts and kernel counts have different instrumentation scope. Separately, current debug logs showed HIP Graph reuse (**144 reuses**)—graphs are working and should not be re-investigated from scratch.

### Synchronization duration distribution

| Subset | Calls | Cumulative API time / observation |
|---|---:|---|
| All `hipStreamSynchronize` | 36,905 | **21.609 s** |
| At most 20 µs | 35,184 | ~0.170 s total |
| Between 20 µs and 20 ms | 1,350 | Remaining mid-range waits |
| More than 20 ms | **371** | **19.320 s**, **89.4%** of all synchronization API time |
| Steady long waits, 45–52 ms | **356** | Mean wait **47.818 ms**; see timing decomposition |
| Maximum wait (startup/transient) | 1 | **361.849 ms** (do not mix into steady state) |

**Correlation:** **370/371** long waits are immediately preceded by a `hipMemcpyAsync` in their host-thread API sequence; the copy's correlation ID maps to a D2H GPU operation. One exceptional wait does not follow that pattern.

## 6. Long verification-like cycles: GPU kernels occupy most of the wait

For **356** steady 45–52-ms waits in the archived trace:

| Component | Value per long wait | Meaning |
|---|---:|---|
| Host `hipStreamSynchronize` duration | **47.818 ms mean** | Host waited for stream completion |
| Sum of intersecting GPU kernel durations | **~39.33 ms mean** | Useful GPU work during wait; potentially overlapping streams |
| Inter-kernel timeline gaps | **~8.35 ms mean** | Some host/dispatch/graph/profiler overhead; **not automatically removable** |
| Kernel launches in interval | ~2,131 mean; typically 2,134 | Many kernels per MTP target-like compute cycle |
| Median positive inter-kernel gap | **~3.88 µs** | Timing is profiler-dependent |
| Initial and final wait margins | ~0.05 ms and ~0.14 ms | Not the dominant interval |
| `hipGraphLaunch` within five prior HIP calls | **355/356** | Strong temporal association with GPU batch launch |

**Conclusion:** The large synchronization periods are *primarily waits for real GPU computation*, not expensive synchronization primitives themselves. Do not estimate throughput gain by removing 39.33 ms of useful computation. The ~8.35-ms gap total is a **profiling-time lead**, not measured recoverable idle time or an instruction on its own to retry HIP Graphs work.

## 7. Device memory copies and readback correlation

| GPU copy direction | Events | Summed GPU transfer time | Approx. median GPU copy duration |
|---|---:|---:|---:|
| Host → Device (H2D) | **3,480** | **519.279 ms** | ~8.44 µs |
| Device → Host (D2H) | **1,754** | **66.702 ms** | ~7.52 µs |
| **Total** | **5,234** | **585.981 ms** | — |

The above GPU copy events are **not** identical to the 13,955 `hipMemcpyAsync` API calls. HIP runtime API calls and GPU transfer records use different scopes and some calls may not generate one distinct traced transfer.

### Long target-like synchronization windows

| Property | Observation |
|---|---|
| Synchronization waits >20 ms | **371** |
| Preceded immediately by D2H-related async HIP API | **370/371** |
| Long waits containing exactly **two D2H copies** | **364/371** |
| Long waits containing exactly **one D2H copy** | **7/371** |
| First copy duration in typical paired wait | ~**0.110 ms** |
| Second copy duration in typical paired wait | ~**0.0075 ms** |
| Last copy completion to host wait return (median) | ~**4.1–4.2 µs** |
| Long waits followed by **two H2D copies** within 1 ms | **367/371** |
| D2H copy events on principal GPU stream 2 | **735** |

The final D2H copy ends almost immediately before the CPU wait returns. **The readback is a completion dependency** rather than a 47-ms PCIe transfer. The two D2H copies may represent target logits and MTP hidden states, but **the archived trace contains no byte sizes, pointer identities, or tensor names**, so this remains a strong hypothesis, **not a proven mapping**. Similarly the two immediately following H2D copies are not proven redundant; they may be new token/embedding inputs.

### Separate short, draft-like readback cycles

| Property | Value |
|---|---:|
| D2H copies on GPU stream 3 | **725** |
| Paired subsequent short synchronization cycles | **725** identified |
| Short sync duration | **~2.805 ms median** |
| Typical GPU kernel execution within cycle | **~2.557 ms** |
| GPU kernel launches per cycle | **43 typical** |
| D2H GPU copy duration | **~0.006 ms typical** |
| Short syncs in 1–5 ms range | **724/725** |
| Ratio short readback cycles / long waits | **725/371 ≈ 1.95** |

**Interpretation:** One to two draft forward passes per target-like step fits MTP2; the 725:371 ratio is additional support for this interpretation. But the GPU timestamps **cannot directly label a phase as draft sampling, target verify, acceptance, or catch-up**. Most of the ~2.8-ms wait is running 43 GPU kernels—not the last ~6-µs result transfer. Do not propose deleting these waits without changing the dependency graph and preserving identical sampling semantics.

### D2H stream inventory

| Stream ID | D2H copy events | Current phase inference |
|---|---:|---|
| 1 | 294 | Generally short/microsecond synchronization; may include setup/other operations |
| 2 | **735** | Associated with long target-like compute/readback waits |
| 3 | **725** | Associated with short draft-like compute/readback waits |
| **Total** | **1,754** | **Stream-phase interpretation unverified at C++ call-site level** |

## 8. Source-level readback evidence — exact *local* excerpt supplied by user

The user executed:

```text
rg -n -C 2 'ggml_backend_tensor_get(_async)?\(|ggml_backend_synchronize\(' \
  ~/ROCm-LLM/development/source/src \
  ~/ROCm-LLM/development/source/common | head -100
```

In the **current local source** `src/llama-context.cpp`, lines printed include:

| Source location | Observed API and requested bytes | Potential runtime role | Confidence |
|---|---|---|---|
| `llama-context.cpp:1561` | `ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float))` | Target logits readback | **Function/payload expression directly observed**, exact trace association not shown |
| `llama-context.cpp:1626` | `ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float))` | MTP hidden-state readback | **Function/payload expression directly observed**, exact trace association not shown |
| `llama-context.cpp:1945` | `ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float))` | Another decode/output logits path | Directly observed |
| `llama-context.cpp:2026` | `ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float))` | Another decode/output hidden-state path | Directly observed |
| `llama-context.cpp:1577`, `1594`, `1610`, `1965`, `1984`, `1999` | Async readback of assorted embedding outputs | General embedding/pooling output paths; may be inactive for MTP2 | Directly observed; MTP use unproven |
| `llama-context.cpp:1684` | Async get of a tensor into a row pointer | Unspecified tensor readback | Directly observed; active phase unknown |
| `llama-context.cpp:2290` | Async tensor-get of input-layer embeddings | Separate layer-input path | Directly observed; active phase unknown |
| `llama-context.cpp:2665`, `2777` | **Synchronous** `ggml_backend_tensor_get` calls | Other tensor serialization/state-related paths | Should **not** be assumed part of steady MTP inference |
| `common/debug.cpp:181`, `src/llama-model.cpp:3410,3415` | Synchronous `ggml_backend_tensor_get` | Debug/model-access paths | Should **not** be assumed active in ordinary decoding |

**Key observation:** The current source visibly requests **two distinct types of data from GPU to host**: logits and `h_nextn` (next-token hidden/pre-normalization state). This plausibly explains repeated paired D2H operations. It does **not** by itself prove that both transfers occur in each observed long window, nor that they are followed by the source's particular `hipStreamSynchronize` calls.

**What this grep did not reveal:** precise stream synchronization call sites, C++ consumers of copied logits and hidden-state rows, device-to-host byte sizes measured at runtime, and the draft/target context identity of each transfer. The command searched `ggml_backend_synchronize` but a context may synchronize through other abstractions such as a scheduler or a sampler. A `head -100` truncation is not a complete call graph.

### Why both readbacks may be necessary

1. The CPU **must see the relevant target distribution** for CPU-side rejection/acceptance and corrected sampling, unless those operations are performed on the GPU.
2. The MTP draft may require the target model's **`h_nextn` hidden state** to predict subsequent tokens. Removing that transfer without providing an equivalent GPU-resident handoff breaks model inputs.
3. CPU draft sampling needs completed draft logits or a GPU-based sampling result to select the next token. A GPU stream/event can enforce *device-side* order, but if CPU decision logic consumes results, that boundary is inherently blocking once the CPU reaches it.
4. When the next forward pass depends on acceptance/rejection and token-position correction, it cannot always run concurrently with the verification that decides its inputs. Overlap is possible only for *independent* work or through speculative work performed in advance and potentially discarded.

### Strongest *new* architecture hypothesis

**Investigate whether target `h_nextn` leaves GPU VRAM only to be copied back to the same RX 6800 for the MTP draft model.** The source's GPU→host get of `t_h_nextn`, paired D2H and subsequent paired H2D activity, and two-context MTP architecture make this plausible. If proven, a **GPU-to-GPU transfer or shared tensor/buffer** could reduce host round trips; however, it requires a safe cross-context/backend lifetime design and correct stream dependencies. Small copy durations mean any speedup is likely from reducing **synchronization/CPU orchestration**, not PCIe bandwidth alone. **This optimization is not implemented, verified, or benchmarked.**

## 9. Repeated activation quantization and tensor copies

| Candidate kernel | Calls | Total GPU time (s) | Classification | Optimization caution |
|---|---:|---:|---|---|
| `quantize_q8_1` | **190,081** | **0.276683** | Exactly one prior to each MMVQ kernel launch in the captured execution order | May be **required conversion of new activation**, not duplicate conversion |
| `mul_mat_vec_q` | **190,081** | **14.884365** | Companion quantized matvec compute | Many workgroups use same quantized activation; fusion could replicate conversion work |
| `concat_non_cont` | 17,760 | **0.460523** | Tensor-layout/concatenation work | May be needed for recurrent states or layout correctness |
| `cpy_scalar` | 60,455 | **0.194408** | Scalar copy kernel | Small per-kernel cost, but many launches; must identify tensors and graph lifetime |

**Safe inference:** Paired quantization and MMVQ launch counts, and their immediate adjacency, prove a *producer/consumer execution pattern*. They do **not** prove that an unchanged activation is re-quantized repeatedly. Total Q8 quantization runtime is only **1.42% of summed GPU kernel runtime**; eliminating its arithmetic alone cannot explain a large application speedup. Fusion might save launch/intermediate-buffer costs but may duplicate quantization across workgroups or lose favorable scheduling. Similarly the sum of `quantize_q8_1 + concat_non_cont + cpy_scalar` execution (~**0.932 s**) is not automatically removable, and a reduction in their summed GPU time is not equivalent to end-to-end wall time.

## 10. GPU ISA and register audit — completed; no need to restart

Data from **current development** `mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1030` disassembly/metadata:

| Instruction/resource check | Observed | Interpretation |
|---|---:|---|
| Static 128-bit global/buffer/flat `dwordx4` loads | **1,010** | Vectorized loads already extensively compiled |
| Static 64-bit `dwordx2` loads | **1,188** | Vectorized loads already compiled |
| Static 32-bit `dword` loads | **3,581** | Scalar reads may be structurally necessary; count alone does not diagnose coalescing |
| Native `v_dot8.*i4` instructions | **0** | Existing path instead uses packed INT8 dot products (`v_dot4c_i32_i8`) |
| Kernels covered by metadata | **325** | All MMVQ variants, not only those used by this model |
| Nonzero reported VGPR spill count | **0 kernels** | No compiler-reported VGPR spills in this code object |
| Nonzero reported SGPR spill count | **0 kernels** | No compiler-reported SGPR spills |
| Some kernels' `private_segment_fixed_size` | **16 bytes/thread** | Not evidence of spilling when `.vgpr_spill_count`/`.sgpr_spill_count` are 0 |
| High register-count outlier | **146 VGPR** | Reported for a `ggml_type 41` MoE template, not necessarily part of this dense Qwen workload |

Previously completed tests: RDNA2 **2 warps × 4 rows** MMVQ scheduling retained, Q4_K and Q5_K packed 128-bit header loads retained, IQ3_S sign multiplication retained, IQ3_XXS gfx1030-only sign multiplication mathematically reviewed and ISA inspected (isolated performance benefit not proven). Rejected/negative results include Q4_K scheduling changes, Q5_K tile sweep, Q4_K wave shuffle/DPP, Q5_K fifth-bit hoist, IQ3_S sign addition/LDS staging, IQ4_XS ordered lookup. HIP Graph capture/replay confirmed operative. **Do not repeat those experiments without new evidence.**

## 11. Counter-derived memory context, with strict comparability warning

These **archived, separate 400-token counter passes** were summarized previously (`gfx1030_counter_analysis_2026-10-11.md`). **They are not paired per-dispatch with this 900-token trace**; do not use them to compute an end-to-end percentage improvement. Ratios and counter names require architecture-specific definitions.

| Metric | IQ4_XS 3c | Q4_K 3c | Q5_K 3c | Q5_K 1c | IQ3_S 3c | IQ3_XXS 3c |
|---|---:|---:|---:|---:|---:|---:|
| `VGPR_Count` | 120 | 88 | 96 | 40 | 112 | 104 |
| `Scratch_Size` | 0 | 0 | 0 | 0 | 0 | 0 |
| `Workgroup_Size` | 64 | 64 | 64 | 32 | 64 | 64 |
| `LDS_Block_Size` (bytes) | 1,536 | 1,536 | 1,536 | 0 | 1,536 | 1,536 |
| `OccupancyPercent` | 53.35% | 58.54% | 43.48% | **102.80%*** | 60.98% | 60.95% |
| `MemUnitBusy` | 89.42% | 84.81% | 73.79% | 99.53% | 90.55% | 90.01% |
| `WAVE_DEP_WAIT` | 86.27% | 52.93% | 59.43% | 87.33% | 69.41% | 63.06% |
| `WAVE_ISSUE_WAIT` | 9.73% | 39.93% | 31.29% | 7.97% | 22.27% | 28.06% |
| `L2CacheHit` | 37.65% | 25.98% | 23.51% | 9.59% | 27.90% | 30.29% |
| Avg `FETCH_SIZE` (KB/dispatch) | 33,749 | 32,168 | 43,072 | 861,880 | 34,327 | 31,635 |
| Derived `LDSBankConflict` | 0.103% | 0 | 0 | 0 | 0 | 0 |

\* A derived occupancy counter greater than 100% cannot represent physical occupancy; treat this as a normalization/reporting quirk. High `WAVE_DEP_WAIT` doesn't automatically imply removable idle wall time.

One previously printed, **single `(ggml_type)23, 2-column`** dispatch (not the central 3-column template) had `GL2C_HIT=55,637`, `GL2C_MISS=220,389`, `GL2C_MC_RDREQ=439,475`, `GL2C_MC_WRREQ=1,048`. This is a *representative raw record*, **not a cache-hit percentage or proof of uncoalesced accesses**. L2 hit/miss/read-request counters alone cannot determine per-wave memory transaction efficiency without sector utilization, access footprints, and normalization.

## 12. Direct answers to the three research questions

### Q1. Which synchronization is necessary to accept/reject draft tokens?

**Necessary dependency:** A host-side accept/reject algorithm must have completed target logits/probabilities before comparing against draft predictions and choosing the corrected token; similarly a CPU-side draft sampler needs completed draft logits. The MTP hidden-state handoff must also complete before the draft consumes that hidden state. **`hipStreamSynchronize` is one way to enforce these dependencies; it is not categorically the only way.** GPU events, stream ordering, or GPU-resident sampling could enforce some dependencies without blocking CPU, provided dataflow is redesigned correctly.

**Measured:** 370 long target-like waits are associated with D2H transfers; 725 short draft-like cycles are associated with D2H transfers. **Not measured:** the exact C++ accept/reject caller for each sync.

### Q2. Which synchronizations prevent CPU sampling / GPU overlap?

**Observed candidates:** The target-like ~47.8-ms wait after graph execution and D2H, and draft-like ~2.8-ms readback/sync per short draft pass. These are natural critical-path barriers for CPU decisions. Nearly all host sync time overlaps required GPU work; it is not a free optimization. Independent preprocessing, prompt work, logging, and any GPU-resident draft processing that does not depend on target acceptance could potentially overlap; the actual available work must be established.

**Specific new question:** Are full target logits or a full hidden-state row being downloaded to the CPU even when a much smaller GPU-produced decision or a device-to-device hidden-state transfer would suffice? Runtime bytes and active sampler path are missing.

### Q3. Can activation conversion or copying be removed without duplication?

**Not established:** The quantize-Q8→MMVQ one-to-one pattern may be essential. To justify reuse, show identical activation tensor identity, generation/position, shape, and contents across multiple MMVQs before overwriting buffers. To justify fuse/reorder, show that quantization is not reproduced in each workgroup and that launch reductions outweigh occupancy/register/graph regressions. For `concat_non_cont` and `cpy_scalar`, first map tensor producer/consumer and role in recurrent state, token embeddings, or KV handling.

**Potentially higher-leverage path:** Avoid a **GPU→CPU→same GPU** `h_nextn` bounce if proven by source call graph and byte-size instrumentation. This targets orchestration/dependency rather than ~0.277-s quantization arithmetic.

## 13. Ranked *new* hypotheses, with decisions and risk

| Hypothesis | Supporting evidence | Unresolved question | Minimal test | Risk/status |
|---|---|---|---|---|
| **A. Target hidden-state CPU bounce** | Local source requests async `t_h_nextn` readback; paired D2H then paired H2D around target-like waits | Does same hidden-state payload return to RX 6800 unchanged? Which buffer/bytes? | Instrument `tensor_get_async`/`tensor_set_async` with tensor label, pointer, nbytes and phase ID; inspect same-context GPU buffer eligibility | **Priority 1; hypothetical** |
| **B. Full logits copied for CPU accept/reject** | Local async `t_logits` gets; D2H precedes long waits | Whole vocabulary every verification row? GPU sampler available? | Log `n_outputs*n_vocab*sizeof(float)` and active sampler type; correlate with HIP copy ID | **Priority 2; correctness-sensitive** |
| **C. Draft iteration CPU round trips** | 725 short readback cycles, typically ~2.8 ms and 43 kernels | Is GPU→CPU sample→GPU step serialized twice per MTP2 cycle? | Source phase timers around draft decode, sampler and hidden readback; correlate IDs | **Priority 3; required dependencies** |
| **D. Kernel dispatch gap reductions** | ~8.35-ms summed gaps in profiled 47.8-ms windows | How much survives profiling removed? Are graph updates/copy scheduling involved? | Lightweight per-phase timing on current build; only then A/B | **Priority 4; profiler confounded** |
| E. Q8 activation conversion fusion/reuse | 190,081 conversion→MMVQ adjacent pairs | Same activation multiple times or one necessary conversion per op? | Log tensor hashes/pointers/generation without materially perturbing timing; review graph | **Low priority / not proven redundant** |
| F. `concat_non_cont`, `cpy_scalar` removal | 17,760 + 60,455 launches, ~0.655-s summed kernel duration | Are copies semantically necessary recurrent-state/layout ops? | Map precise tensors and lifetime | **Low priority / correctness risk** |

## 14. What profiling or source data is actually missing

**No new broad hardware-counter capture is justified.** To close the analysis conclusively, obtain **the exact local source** (`common/speculative.cpp`, `common/sampling.cpp`, `src/llama-context.cpp`, and the active server decoding/verification call sites in `tools/server/server-context.cpp`) and locate:

1. **MTP source route:** calls to `llama_get_embeddings_nextn_ith`, `llama_get_logits_ith`, `common_sampler_sample`, target verify/accept logic, `llama_process(ctx_dft)`, and `batch.set_embd`/equivalent.
2. **Synchronization path:** `llama_context::synchronize`, `ggml_backend_sched_synchronize`, `ggml_backend_synchronize`, and the concrete caller that runs between `ggml_backend_tensor_get_async` and CPU consumption.
3. **Transfer identities:** backend, tensor name, tensor shape, `nbytes`, src/dst pointer, host-pinned status, stream, `ctx_tgt` vs `ctx_dft`, draft index (0 or 1), position and phase, and whether the copy is logits or hidden-state.
4. **Sampling location:** CPU or GPU sampler, exact bytes transferred, whether full logits are necessary to implement the same acceptance law (including rejection correction).
5. **State correctness:** full target/draft KV positions, hidden-state row selected after partial acceptance/rejection, graph capture/replay safety, and buffer lifetime when moving tensor between contexts.

If these source paths remain ambiguous, use **one narrowly targeted trace/profile on the current development build** with phase marks and transfer byte tags. Do not run nine PMC groups again. The prior data already establishes the large/small wait patterns and producer/consumer kernel timeline. Keep diagnostic runs separate from performance measurements.

### Proposed measurement protocol for any actual patch

| Stage | Measurement | Pass condition |
|---|---|---|
| Before edit | Record exact binary sha256, GGUF sha256/name, source `git diff`, ROCm version, flags | Reproducible current baseline |
| Correctness A/B | Same prompt, seed, sampling, reasoning settings; compare draft/verify token sequences, accept decisions, KV and hidden states | **No unintended semantic divergence**; if floating-point differs, characterize tolerance and resulting sampling differences |
| Instrumented path check | Source phase spans, actual payload sizes and call sites; distinguish CPU wait from GPU work | Hypothesis supported/refuted; measured reduction at intended boundary |
| Uninstrumented A/B | Interleaved repeated comparable runs, controlled GPU clocks/thermals, **at least 3** 1,000-token trials per condition (prefer more if 1% gains) | End-to-end t/s beats current control **on identical workload** outside noise |
| Regression | Prefill and single-token decode, partial MTP acceptance/rejection, boundary/KV cache rollbacks, reasoning behavior, other GGUF formats | No correctness regressions or unacceptable slowdowns |

**Do not claim that the historical profiler's ~8.35 ms gaps imply a 17.5% throughput gain.** The 45.723 t/s reference remains valid until replaced by a controlled *current-build* A/B.

## 15. Preserved completed work — do not reopen by default

| Workstream | Result |
|---|---|
| RDNA2 MMVQ three-column scheduling | **2 warps × 4 rows** retained as previous best V1 |
| Q4_K / Q5_K | Packed 128-bit header implemented, other sweeps rejected/uncertain |
| IQ3_S | Sign-mask multiplication retained; addition and LDS staging rejected |
| IQ4_XS | Extensively investigated, ordered-lookup attempt unsuccessful; not the default next branch |
| IQ3_XXS | gfx1030-only multiplication sign-mask implemented; mathematically checked, ISA inspected; isolated win unconfirmed |
| HIP Graphs | Operational; replay/reuse verified; no established worthwhile graph toggle speedup |
| ISA/vector-load audit | 1,010 x4; 1,188 x2; 3,581 scalar loads in compiled MMVQ variants |
| INT4 dot | Zero native `v_dot8.*i4` used; no demonstrated advantage for Q4 weight × Q8 activation |
| Register spilling | 325 MMVQ kernels, zero compiler-reported SGPR/VGPR spills |
| New authoritative MTP2 benchmark | **45.723 tokens/s** across three current-development trials |

## 16. Trace/archive file inventory and paths

**User-provided inputs for this analysis (attach to another model if it can process files):**

```text
review-upload.tar.gz
mtp-traces.tar.gz                 # trace_hip_api_trace.csv + trace_kernel_trace.csv
mtp-copy-traces.tar.gz            # trace_memory_copy_trace.csv + trace_memory_copy_stats.csv
counter-recovery-upload.zip       # prior summarized raw PMC coverage
rocm-mmvq-changes.diff            # current development source modification patch
```

On the user's Ubuntu machine:

```text
~/ROCm-LLM/development/source/src/llama-context.cpp
~/ROCm-LLM/development/source/common/speculative.cpp
~/ROCm-LLM/development/source/common/sampling.cpp
~/ROCm-LLM/development/source/tools/server/server-context.cpp
~/ROCm-LLM/development/build/bin/llama-server
~/ROCm-LLM/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf
~/ROCm-LLM/profiling/current/mmvq-profile-campaign/
~/ROCm-LLM/benchmarks/results/speculate-20261011-202008/results.csv
```

**Research ask for another model:** Explain precisely the local MTP2 data dependency DAG: target forward → logits / `h_nextn` readback → host accept/reject and sample → draft context input → draft forward/readback → subsequent target verification, including where blocking synchronizations originate. Identify and prove any GPU→CPU→GPU hidden-state bounce, quantify transferable bytes and necessary waits, and propose the smallest **opt-in, testable** improvement (prefer buffer-sharing/device-side handoff over source-level quantization sweeps). Provide a correctness test for mixed acceptance, rejection, KV rollback, graph capture, and multi-token output. Distinguish evidence, hypothesis, and unverified assumptions. Do not edit source or assert speedup before a controlled A/B.

---

**Bottom line:** Existing traces show two **real** GPU-to-CPU decision boundaries, one long target-like and one short draft-like. The waits mostly contain necessary GPU computation. The local source shows plausible logits and `h_nextn` readbacks, but archived traces lack transfer byte sizes/call stacks to map individual copies to individual tensors. **The actionable next task is narrow call-site and payload tracing, not kernel micro-optimization recycling.** The unchanged current benchmark is **45.723 t/s**.
