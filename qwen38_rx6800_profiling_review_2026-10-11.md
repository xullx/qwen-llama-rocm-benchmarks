# Qwen3.8 27B / RX 6800 — existing ROCm profiling review

Date: 2026-10-11. Source: user-provided `review-upload.tar(1).gz`, extracted for offline inspection. **No new GPU tests were executed.**

## Version separation

- **Current authoritative inference performance:** 45.723 generated tokens/s, three 1000-token MTP2 trials (45.346, 45.284, 46.539 t/s) on `~/ROCm-LLM/development/build/bin/llama-server`, commit `139997d` plus local changes, Linux ROCm, IQ4_XS, 8192 context.
- **Archive being analyzed:** historical campaign whose `campaign-parameters.json` names `/home/xu/mmvq-rdna2-q4q5-iq3s-signmul-bin/llama-server` and a Windows NTFS-mounted GGUF. It is **not** the isolated current development build. Its performance measurements must not replace the current baseline.
- Profiling itself changes timing; several recorded counter runs had `profile_rc=-9`, despite 400 generated tokens. Never use their tokens/s as the baseline.

## What the complete included GPU timeline establishes

From `trace/rocprof/trace_kernel_stats.csv`, over 887,693 kernel launches totaling 19.476678 s of summed GPU kernel durations:

| Kernel group | Calls | Summed GPU execution | Share of summed kernel durations |
|---|---:|---:|---:|
| `mul_mat_vec_q` (MMVQ) | 190,081 | 14.884365 s | 76.42% |
| `mul_mat_q` | 3,024 | 1.917184 s | 9.84% |
| `concat_non_cont` | 17,760 | 0.460523 s | 2.36% |
| `gated_delta_net_cuda` | 17,808 | 0.348800 s | 1.79% |
| `rms_norm_f32` | 120,820 | 0.336580 s | 1.73% |
| `quantize_q8_1` | 190,081 | 0.276683 s | 1.42% |
| `flash_attn_tile` | 7,031 | 0.256733 s | 1.32% |
| `cpy_scalar` | 60,455 | 0.194408 s | 1.00% |

The archive's MMVQ share computes to **76.42%** for this particular trace; the previously documented **75.62%** is a result of an earlier/different capture, not a discrepancy to average together.

Particular MMVQ three-column groups in this capture:

| Quantization type identifier | Calls | GPU time | Share of total GPU kernel time |
|---|---:|---:|---:|
| `(ggml_type)23` (IQ4_XS) | 56,110 | 4.4900 s | 23.05% |
| `(ggml_type)12` (Q4_K) | 42,716 | 3.2754 s | 16.82% |
| `(ggml_type)13` (Q5_K) | 17,376 | 1.6976 s | 8.72% |
| `(ggml_type)21` (IQ3_S) | 15,566 | 1.4988 s | 7.70% |

**Do not reopen those completed kernel experiments automatically.** The table is background to target *new* bottleneck hypotheses.

## Host-to-GPU orchestration

From `trace/rocprof/trace_hip_api_stats.csv`:

- `hipStreamSynchronize`: **36,905** calls, **21.6093 s summed API durations**. This includes CPU waiting for actual GPU work. It **is not** 21.6 s of avoidable overhead or 86.81% of wall time. A callsite- and dependency-resolved analysis would be needed before changing synchronization.
- `hipGraphLaunch`: **1,434** calls, 1.0329 s summed API duration.
- `hipGraphExecUpdate`: **16** calls, 0.5368 s summed API duration. Not necessarily ongoing steady-state overhead; graph shape changes and warm-up matter.
- `hipLaunchKernel`: **43,693** calls. The kernel trace includes dispatches launched through graphs, so don't compare these counts as though they were the same instrumentation scope.

## Memory copies and redundant work

From `trace/rocprof/trace_memory_copy_stats.csv` and `trace_memory_copy_trace.csv`:

- Host-to-device: **3,480** copies, **0.51928 s summed GPU copy durations**.
- Device-to-host: **1,754** copies, **0.06670 s summed GPU copy durations**.
- Total: **5,234 copies**, **0.58598 s summed transfer durations**. Median H2D transfer duration ~8.44 us; median D2H ~7.52 us.
- Copies continue throughout the approximately 27.6 s copied-data timeline, not just during initialization. **Sizes and purpose are unavailable** in this archive's copy trace: it does not expose transfer bytes. The totals do not establish excessive PCIe bandwidth consumption or redundant payloads.
- `quantize_q8_1` and MMVQ both have **exactly 190,081 GPU kernel calls**, a concrete relationship worth investigating in the graph/operator pipeline. This count **suggests** frequent activation quantization coupled to MMVQ; it does **not** prove duplicate quantization or that fusion will improve performance. Removing all `quantize_q8_1` execution time would have an ideal upper bound around 1.42% of summed GPU kernel time, excluding any potential launch/scheduling benefit or negative effects of fusion.
- `concat_non_cont` 17,760 calls / 0.461 s and `cpy_scalar` 60,455 calls / 0.194 s are concrete data-movement/copy kernels. Whether they can be removed depends on tensor layout semantics and accuracy.

## What cannot yet be concluded

1. **Memory coalescing:** the L2 campaign planned `GL2C_HIT`, `GL2C_MISS`, `GL2C_MC_RDREQ`, `GL2C_MC_WRREQ`; these do not by themselves expose load-sector utilization / coalescing efficiency. The user supplied a sample of the collection CSV separately, but **the uploaded review archive omits the large `*_counter_collection.csv` files**. Do not claim full per-dispatch hardware-counter results from this reduced upload.
2. **True register-spill problems:** a separate audit of the *current* code object reported 325 kernels and zero SGPR/VGPR spills in compiler metadata. The 16-byte private segment in some kernels alone is not evidence of spills.
3. **MTP verification overhead vs useful compute:** a single MTP2 timeline does not cleanly separate draft, verification, CPU sampling, necessary synchronization, graph maintenance, and cache catch-up. A speedup percentage requires attribution, not the count of HIP API calls.

## Research decision

- **Completed/excluded from immediate reruns:** MMVQ scheduling sweeps; Q4_K/Q5_K 128-bit headers, Q5_K tiles, IQ3_S and IQ4_XS prior experiments; HIP Graphs basic operation; 32/64/128-bit instruction audit; INT4 dot investigation; register-spill check.
- **Best concrete *new* lead from this review:** determine which of the 36,905 HIP synchronizations are on the generation critical path and which are required by the MTP draft/verification handoff; related host↔device copies are small and frequent. This is a hypothesis, not a claimed improvement.
- **Second lead:** inspect whether any `quantize_q8_1`, `concat_non_cont`, or `cpy_scalar` work is semantically redundant across MTP verification steps. Avoid changing mathematically necessary operations.
- **Memory-coalescing determination remains inconclusive** from the compressed review subset.

## Source inventory

Uploaded archive: `review-upload.tar(1).gz`. Evidence: `campaign-parameters.json`, `trace/rocprof/trace_kernel_stats.csv`, `trace/rocprof/trace_hip_api_stats.csv`, `trace/rocprof/trace_memory_copy_stats.csv`, `trace/rocprof/trace_memory_copy_trace.csv`, `counter-plan.json`.
