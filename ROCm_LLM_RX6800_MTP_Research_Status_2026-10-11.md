# ROCm-LLM Research Status — Qwen3.8 27B / RX 6800 / MTP2

**Last updated:** 2026-10-11 (end-of-session handoff)  
**State:** Development build working; MTP draft-vocabulary masking experimentally validated; frequency-calibration study **prepared but not run**.  
**Primary goal:** Increase sustained MTP2 generation throughput from **45.723 output tokens/s** toward **50 tokens/s** while retaining model output behavior, strong acceptance, and stability.  
**Collaboration:** Research assisted by GPT, Gemini, DeepSeek, and occasionally Claude. Claims below are labeled by evidence level; this document is not an upstream llama.cpp performance claim.

> **For the next research session / AI collaborator:** Read this before proposing experiments. Do **not** restart the already-completed MMVQ tuning, HIP Graph feasibility checks, or ISA/register-spill inspections. Avoid claiming that profiled synchronization time is removable CPU overhead. Use **one terminal command at a time** when guiding interactive work. **Do not implement a compact GPU vocabulary head before collecting and validating realistic token-distribution data.**

## 1. Environment and reproducibility

| Item | Configuration |
|---|---|
| Operating system | Linux / Ubuntu |
| GPU | AMD Radeon RX 6800, RDNA2 `gfx1030`, 16 GiB VRAM |
| GPU backend | ROCm / HIP, ROCm 10.0 in the current research setup |
| Inference engine | Locally modified llama.cpp, `0.5.0-dev`, Git base `139997d` **plus uncommitted/local changes** |
| Model | `Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf` |
| GGUF architecture | `qwen35` (confirmed by `general.architecture` metadata) |
| MTP | Two-token speculative draft, `draft-mtp`, maximum 2 |
| Current source | `~/ROCm-LLM/development/source` |
| Development server | `~/ROCm-LLM/development/build/bin/llama-server` |
| Model path | `~/ROCm-LLM/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf` |
| Benchmark script | `~/ROCm-LLM/benchmarks/scripts/bench-qwen-rocm-linux.py` |
| Benchmark isolation | Dedicated port `8091`; do not assume a second GPU server can run concurrently without resource contention |

Representative benchmark runtime settings (maintain unchanged for comparable tests): 8,192 context, `-ngl 99`, `--fit off`, `--device ROCm0`, 8 CPU threads, 4 batch threads, batch 1024, microbatch 128, flash attention enabled, one parallel slot, `f16` target/draft KV, and the benchmark's fixed prompt and seed sequence. A commit hash alone does **not** reproduce the optimized binary: preserve both source diffs and build settings.

The standard comparison is **3 runs × 1,000 generated tokens**, with target output hashes, accepted/proposed counts, and arithmetic-mean generated tokens/s recorded. Small changes (~1%) need more repetitions/interleaved A/B measurements before attribution.

## 2. Current benchmark evidence (unprofiled)

All rows below correspond to completed runs on the existing full **GPU** output projection, **not** a 32K/64K compact output head. The prefix rows are *token-ID subsets*, not representative frequency-selected vocabularies.

| Experiment | Three runs (tokens/s) | Mean (tokens/s) | Draft accepted / proposed | Acceptance | Relative to CPU control | Execution status |
|---|---|---:|---:|---:|---:|---|
| **Original unmodified-sampling baseline** | 45.346 / 45.284 / 46.539 | **45.723** | 1,790 / 2,411 | **74.24%** | Not directly controlled for CPU sampling | ✅ Complete |
| **Patched, unrestricted, CPU draft sampling** | 44.892 / 44.847 / 46.094 | **45.278** | 1,790 / 2,411 | **74.24%** | Reference | ✅ Complete |
| **First 32,768 token IDs allowed** | 42.524 / 42.655 / 44.907 | **43.362** | 1,742 / 2,507 | **69.49%** | **−4.23%** speed; −4.75 pp acceptance | ✅ Complete |
| **First 65,536 token IDs allowed** | 43.870 / 44.373 / 45.665 | **44.636** | 1,777 / 2,437 | **72.92%** | **−1.42%** speed; −1.32 pp acceptance | ✅ Complete |
| **Frequency-ranked 32K / 64K** | — | **Not measured** | — | — | — | ⏳ Pending |
| **Real compact 32K / 64K GPU projection** | — | **Not measured** | — | — | — | 🚫 Not implemented |

**Benchmark output folders** (all relative to `~/ROCm-LLM/benchmarks/results/`):

- `speculate-20261011-202008/` — original 45.723 t/s baseline.
- `speculate-20261011-220345/` — patched, unrestricted CPU-sampling control.
- `speculate-20261011-220740/` — arbitrary 32K-prefix mask.
- `speculate-20261011-221045/` — arbitrary 64K-prefix mask.

**Correctness observation:** Across these four conditions, the completion hash for each matched seed was identical: run 1 `edd11c79537494c2`, run 2 `9ea431712f68d984`, run 3 `7c18a1c1808f2ed0`. This supports output invariance **for these three tests only**, not universal mathematical correctness for all prompts, grammar/sampler settings, or contexts. It also confirms that the mask did change draft acceptance counts while the target output remained equal in these tests.

**Conclusions:**

1. Draft-vocabulary filtering works at runtime and preserves target verification behavior in the tested workloads.
2. **32K arbitrary-prefix filtering costs substantial draft acceptance.** It is not a trustworthy test of an intelligently selected 32K token set.
3. **64K arbitrary-prefix filtering loses only 1.32 acceptance percentage points**, which is encouraging for further study, but still runs the **full-vocabulary head** and is slower today.
4. The 0.445 t/s difference between original and patched CPU-control means is not a confirmed performance regression at this sample size.
5. **No 50 t/s result has been demonstrated.** A compact GPU head is a hypothesis, not an existing optimization.

## 3. Confirmed source architecture and implementation boundaries

| Area | Relevant source | Confirmed behavior |
|---|---|---|
| Model architecture | `src/models/qwen35.cpp` | GGUF metadata uses `qwen35`; model has 64 trunk layers and a next-token-prediction block at `blk.64` |
| Standard target projection | `qwen35.cpp` normal `graph` | `build_lora_mm(model.output, ...)` produces full-vocabulary target logits |
| MTP projection | `qwen35.cpp`, `graph_mtp` around lines 624–643 | Uses `layer.nextn.shared_head_head` if present; otherwise falls back to `model.output` |
| Actual GGUF weights | Model metadata/tensor directory | `output.weight` and `blk.64.nextn.shared_head_norm.weight` exist; no separate `nextn.shared_head_head.weight` was found in the inspected tensor directory |
| MTP hidden state | `common/speculative.cpp`, MTP `process()` and `draft()` | Separate draft context; target `h_nextn` rows are held/copied through CPU-side buffers and supplied as draft embeddings on the non-shared-memory path |
| Draft candidate selection | `common/speculative.cpp` | Calls `common_sampler_sample(..., ctx_dft, ...)`, retrieves sorted candidates, chooses a draft token, and feeds its **original token ID** to later draft decoding |
| Candidate construction | `common/sampling.cpp::set_logits()` | Supports backend probabilities, sparse backend candidate logits, or full logits; keeps original token IDs |
| Verification | `common/sampling.cpp::common_sampler_sample_and_accept_n()` | Samples from the **target** at each verification position, accepts matching draft IDs, stops at the first mismatch, and samples a final token when all drafts match |
| Optional mask | Research patch applied to `common/sampling.cpp`, `common/sampling.h`, `common/speculative.cpp` | Attached to the MTP draft sampler only; filters **after** the GPU model has produced full logits |

**Important qualification:** The GGUF weight evidence strongly supports the MTP graph's fallback to `output.weight`, plus its separate MTP norm. Sharing a GGUF weight source does **not** prove which physical GPU allocation is reused by each loaded context. The compact head needs a separately packed draft-only matrix and logits/token-ID integration, not a destructive rewrite of the target weights.

## 4. Implemented local research patch — successfully compiled and runtime-tested

The `mtp-draft-experiments.patch` was applied with `git apply`, followed by a successful incremental `llama-server` build. The three experiment variables are **disabled by default**:

| Variable | Purpose | Status |
|---|---|---|
| `MTP_DRAFT_VOCAB_FILE=/path/to/ids.txt` | Restricts MTP CPU draft sampler to the specified original token IDs (one ID/line); full GPU projection still computed | ✅ Tested at 32K and 64K |
| `MTP_DRAFT_FORCE_CPU_SAMPLING=1` | Explicit CPU draft-sampling control for comparable masked/unmasked benchmarks | ✅ Tested |
| `MTP_DRAFT_FAST_TOP1=1` | Experimental linear top-1 selection instead of the final full candidate sorting step | ⚪ Implemented; **not** benchmarked independently |

Source file patch was distributed as `mtp-draft-experiments.patch`; accompanying notes are in `mtp-draft-experiments/README.md` (artifact names from this session; these files are **not automatically committed to GitHub**). The sorting shortcut is **low priority**: the active sampler's `top_k=10` typically leaves only a short post-sampling candidate list. It does not remove the expensive full-vocabulary top-k / projection work. Do not describe it as an established performance win.

**Precautions:** The patched tree already contains earlier kernel optimizations. Record `git status`, source diff, and build flags before making additional edits. Do not reset the working tree to `139997d`, reapply a patch blindly, or overwrite the original benchmark directories.

## 5. Archived ROCm profiling — useful diagnostics, not current throughput

These measurements come from a **different earlier optimized binary and profiled runs**. They must not be combined algebraically with current 45.723 t/s to promise a speedup.

### 5.1 GPU kernel budget

- A kernel trace reported **887,693 executions** with **19.477 seconds of summed GPU kernel durations**.
- Quantized matrix-vector kernels (`mul_mat_vec_q`) accounted for **190,081 calls / 14.884 s / 76.42%** of that trace's summed kernel time; an earlier different capture reported **75.62%**. Keep those captures distinct.
- Leading 3-column MMVQ shapes: IQ4_XS **4,489.97 ms**, Q4_K **3,275.42 ms**, Q5_K **1,697.63 ms**, IQ3_S **1,498.82 ms**.
- The dominant single-column Q5_K head-like kernel averaged roughly **1.65–1.68 ms per launch** under profiling and contributed about **6.5%** of recorded kernel time. At two draft passes per cycle, this suggests a potentially meaningful output-head cost, **but profiling overhead and workload heterogeneity make a 2–3 ms/cycle saving only a hypothesis**.
- Exactly **190,081 `quantize_q8_1`** and **190,081 MMVQ** launches in the archived trace; one-to-one occurrence **does not** prove redundant quantization or permit reuse without verifying identical activation tensors.

### 5.2 CPU/GPU synchronization and transfers

- **36,905** profiled `hipStreamSynchronize` calls summed to **21.609 s** in the HIP API timeline; **371** calls exceeded 20 ms and accounted for **19.320 s** of this sum.
- **370/371** long calls followed a D2H-related async copy in the traced host thread. However, much of the wait overlapped necessary GPU execution; this is **not 19 seconds of removable overhead**.
- **356** steady long target-like waits averaged **47.818 ms**, intersecting roughly **39.33 ms** of GPU kernel time. Observed timeline gaps of roughly **8.35 ms** were recorded **with profiling active** and are not a validated unprofiled optimization budget.
- **725** shorter D2H/draft-like cycles (consistent in count with MTP2) often contained **43 GPU kernels**, about **2.56 ms** GPU work, and **2.8 ms** observed synchronization; precise source-to-trace attribution is not complete.
- HIP Graphs **are operational**, including a prior debug summary of `graphs reused=144`. The open question is narrowly whether **short draft passes** benefit from graph capture/replay, not whether graphs work at all.
- GPU transfer operations were individually inexpensive in the archived trace; avoid conflating physical D2H copy duration with a long host wait for prior kernels.

### 5.3 Completed ISA/resource work

- RDNA2 MMVQ: retained **2 warps × 4 rows** for 3-column scheduling (previously fastest measured V1).
- Retained packed 128-bit quantization-header reads for Q4_K and Q5_K and IQ3_S sign-mask multiplication; IQ3_XXS gfx1030 sign-mask multiplication implemented/checked but isolated gain remains unproven.
- Static current-build MMVQ object inspection counted **1,010 x4 global loads**, **1,188 x2 loads**, **3,581 scalar dword loads**, zero `v_dot8.*i4`, and zero compiler-reported VGPR/SGPR spills across **325** MMVQ kernels. Static instruction counts are **not dynamic memory traffic measurements**.
- Nine profiler hardware-counter groups were recovered (**2,165,481 metric records**) but the campaign had abnormal profiler exits and must be used diagnostically, not as a direct t/s comparison.

## 6. Experiment decision log — avoid repeating exhausted work

| Research path | Result/status | Decision |
|---|---|---|
| RDNA2 MMVQ 3-col warp/row sweep | 2 warps × 4 rows won previous controlled tests | **Retain** |
| Q4_K scheduling sweep | No improvement over retained schedule | **Do not repeat without new evidence** |
| Q5_K eight tile variants | None beat V1 | **Close** |
| Q4_K header wave-shuffle / DPP | No established win | **Close** |
| Q5_K fifth-bit hoist and alternate tiles | No validated gain | **Close** |
| IQ3_S sign addition / LDS staging | No improvement | **Close** |
| IQ4_XS ordered lookup | No validated improvement | **Close** |
| HIP Graph basic functionality | Replay/reuse confirmed | **Do not retest basic operation** |
| Quantization Q8_1 sharing/fusion | Matching counts only, not tensor identity | **Deferred** |
| MTP CPU draft full-vocab control | 45.278 t/s; same acceptance as baseline | **Established** |
| MTP 32K / 64K *prefix* masks | Acceptance 69.49% / 72.92% | **Completed diagnostic** |
| MTP frequency-ranked token subsets | No corpus collected yet | **Next** |
| Compact output-head quantized packing | No source implementation | **Deferred pending vocabulary study** |

## 7. Active roadmap — research branches (not Git branches)

```text
ROCm-LLM / RX 6800 / Qwen3.8 27B
│
├── A. MTP correctness and architecture                         [CONFIRMED]
│   ├── A1. Target-token matching and full target sampling       [DONE]
│   ├── A2. qwen35 GGUF/MTP graph and output weight fallback     [DONE]
│   └── A3. MTP-only original-ID vocabulary mask                 [DONE]
│
├── B. Vocabulary-distribution study                             [NEXT]
│   ├── B1. Run opt-in unrestricted proposed/compared/accepted/target logger
│   ├── B2. Count distribution, top-token coverage and per-language effects
│   ├── B3. Generate 32K/64K frequency-ranked vocab lists
│   ├── B4. Re-run masks on separate representative/held-out prompts
│   └── B5. Record full vs 32K vs 64K acceptance, speed, hashes
│
├── C. Real compact MTP GPU output head                          [BLOCKED BY B]
│   ├── C1. Pack/gather quantized output.weight rows into separate draft tensor
│   ├── C2. Preserve target full-vocab projection and special-token semantics
│   ├── C3. Map compact candidate indices -> original token IDs throughout API
│   ├── C4. Confirm VRAM fit, output logits shape, LoRA and sampler compatibility
│   └── C5. Benchmark GPU head cost, end-to-end t/s, acceptance and correctness
│
├── D. MTP timing and orchestration                              [FUTURE]
│   ├── D1. Unprofiled per-phase timers: target verify / draft 1 / draft 2 / CPU
│   ├── D2. Test whether short draft passes use HIP Graph replay
│   ├── D3. Attribute target logits and h_nextn D2H/H2D buffers and byte sizes
│   └── D4. Explore device-side sampling/state handoff ONLY if timing supports it
│
├── E. Conditional graph/kernel opportunities                    [DEFERRED]
│   ├── E1. Q8_1 quantization reuse with proven tensor identity
│   ├── E2. 3-column gate/up fusion with correctness and occupancy validation
│   ├── E3. concat_non_cont / cpy_scalar tensor-specific attribution
│   └── E4. Additional format-specific kernel work only with new bottleneck proof
│
└── F. Completed MMVQ / ISA / hardware-counter sweeps             [ARCHIVED]
```

### 7.1 Immediate pending task: token statistics (not yet run)

An opt-in study script is prepared: **`mtp-vocab-study.sh`** (distributed separately). It expands a small bundle and executes `mtp-vocab-study/run-study.sh`, which:

1. Checks that the prior draft-mask patch exists and that its additional statistics patch applies cleanly.
2. Backs up the current `common/sampling.cpp`, adds **opt-in** `MTP_VOCAB_STATS_FILE` logging to standard target verification, and rebuilds `llama-server` incrementally.
3. Runs unrestricted CPU-draft MTP2 for **3 × 1,000 output tokens**, recording four event types: `proposed`, `compared`, `accepted`, `target`.
4. Generates ranked **32,768** and **65,536** ID lists using the **exploratory** score `8×accepted + 3×compared + 2×target + proposed`. Unseen slots are padded from the original vocabulary-ID prefix.
5. Runs the two whitelist masks on **the same** benchmark prompts and reports comparative speed, acceptance, and hashes.
6. Stores results under `~/ROCm-LLM/benchmarks/vocab-studies/<timestamp>/`.

**Do not mistake the study for completed research.** As of this handoff, its patch **has not been applied or compiled on Ubuntu**, its runtime statistics **have not been collected**, and the frequency-ranked 32K/64K masks **have not been benchmarked**.

**Statistical limitations:** Three thousand target outputs cannot characterize a ~250K-token vocabulary tail, so ranked lists will contain substantial arbitrary-prefix padding. Observed frequency is **not the complete model probability distribution**, and high target-token coverage does **not** guarantee the same draft proposal/acceptance probabilities after masking. The study's reuse of calibration prompts for masked comparisons causes **in-sample bias**. Treat it as a smoke test, then collect a larger **representative, language-diverse corpus** and evaluate on **held-out prompts** (e.g., English/Spanish/Japanese; code, technical prose, conversations, long-context continuations as relevant). File logging affects wall time; **exclude logged runs from speed A/B conclusions**. The runner does not modify `qwen35.cpp` or build a smaller GPU head.

### 7.2 Decision gates before any GPU-head implementation

Proceed only when all of the following have been demonstrated:

- The selected vocabulary has high **held-out** draft acceptance with the same sampler/grammar settings and sufficiently varied prompt seeds.
- Stable output hashes or equivalent target-distribution validation on paired test cases; account for seed-dependent divergence and context effects.
- A defensible cost estimate for **actual** compact GPU matvec/packing overhead, not the CPU-only mask runtime.
- A practical tensor layout for quantized row packing, an original-token-ID map, a safe logits shape for sampler/backends, correct special-token handling, and adequate VRAM headroom on the 16 GiB RX 6800.
- A measurable end-to-end performance win across repeated **unprofiled** runs; no correctness or memory regressions.

## 8. Throughput target and realistic gain accounting

Original throughput **45.723 t/s → 50 t/s** requires approximately **+9.35%** more output tokens per second. Under the previously estimated **~54.2 ms/cycle** and unchanged acceptance, this corresponds to roughly **49.6 ms/cycle**, or **~4.6 ms/cycle saved**. This is a planning calculation, **not a measured overhead budget**. If mask acceptance changes, the tokens produced per speculative cycle also change, so a smaller draft head can fail to improve total throughput despite faster individual draft passes.

Earlier profiled single-column draft-head work suggests a theoretical **~2–3 ms/cycle opportunity**, but that is **unverified**, depends on the selected vocabulary size, existing memory/bandwidth effects, and does not by itself justify a claim of 50 t/s. Draft-side Graph capture, phase orchestration, and device-side transfers are separate possible avenues once unprofiled phase timing shows real savings exist.

## 9. Next-session checklist

- [ ] **Before running anything:** confirm current `git status`, current build, GPU/server occupancy, and preserve outstanding optimized `mmvq.cu` / `vecdotq.cuh` diffs.
- [ ] Run the **prepared** vocabulary study (after closing competing inference jobs); inspect success/failure, file output, acceptance, candidate counts, and hashes.
- [ ] Review `calibration-summary.json`, vocabulary overlap/coverage, and the proportion of padded unobserved IDs.
- [ ] Expand beyond three benchmark prompts; create independent **calibration and held-out prompt sets** representative of intended tasks and languages.
- [ ] Re-run curated 32K/64K masks on holdout, comparing against unrestricted CPU-sampling control under matching conditions.
- [ ] Record **three or more paired repeats** for speed claims; monitor VRAM and GPU thermals when meaningful.
- [ ] Only if held-out results support it, design and A/B-test compact GPU projection with correct original-ID mapping.
- [ ] Separately, if vocabulary head is unattractive, measure draft-phase HIP Graph usage and CPU/GPU handoff with lightweight **unprofiled** timers rather than repeating the old PMC sweep.
- [ ] Update this handoff's **Results**, **Decision log**, and **Status** with new run folders, diffs, and benchmark hashes.

**Last action today:** The frequency-calibration script was prepared and shared, but **not run**. The patched full-head MTP2 server remains proven buildable and the most recent completed benchmark was **64K arbitrary-prefix masking: 44.636 t/s and 72.92% acceptance**.

## 10. GitHub organization / provenance

Suggested tracked location for this document: `docs/research/2026-10-11-rocm-qwen38-mtp.md` (or replace the research-status section of the repository README). Keep experiments organized as *document sections/directories*, not fictitious Git branches.

Suggested supplementary files if already present in the repository:

```text
README.md
/docs/research/2026-10-11-rocm-qwen38-mtp.md
/benchmarks/scripts/bench-qwen-rocm-linux.py
/benchmarks/results/                      # keep summary CSV/JSON selectively
/patches/mtp-draft-experiments.patch      # only if intentionally tracked
/tools/mtp-vocab-study.sh                 # optional, after reviewing the script
```

**Reference analysis materials** (external to this document and possibly not committed):

- `Qwen38_27B_RX6800_MTP_Optimization_Full_Handoff_2026-10-11.md`
- `2026-10-11-benchmarks.md`, `gfx1030_counter_analysis_2026-10-11.md`
- `qwen38_rx6800_profiling_review_2026-10-11.md`
- `qwen38_rx6800_mtp_trace_analysis_2026-10-11.md`
- `qwen38_rx6800_mtp_copy_attribution_2026-10-11.md`
- `rocm-mmvq-changes.diff`
- Original diagnostic archives: `review-upload.tar.gz`, `mtp-traces.tar.gz`, `mtp-copy-traces.tar.gz`, `counter-recovery-upload.zip`

Do not push the GGUF model, huge raw profiler traces, or private prompts/logs accidentally. Large diagnostic data should be archived externally or summarized with provenance. **No GitHub push or repository mutation is implied by creating this Markdown handoff.**
