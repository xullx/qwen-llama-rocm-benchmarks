# RX 6800 ROCm — MMVQ Optimization Research Summary
**Research status: October 10, 2026**

**Hardware:** AMD RX 6800, RDNA2 (`gfx1030`), 16 GB VRAM  
**Backend:** ROCm 10.0 / HIP / llama.cpp  
**Model:** Huihui-Qwen3.8-27B, UD-IQ4_XS, MTP speculative decoding  
**Benchmark:** 3,200 generated tokens, fixed seed 1234, two rounds

**Best validated performance: 45.245 tokens/s (+6.63% over the original scheduling).**

The current research is moving from high-level kernel scheduling toward **RDNA2-specific instruction optimization**, particularly wavefront reductions.

---

## 1. Completed MMVQ scheduling experiments

We initially investigated how many warps and output rows should be assigned to an MMVQ thread block.

### Results

| Configuration | Average t/s | vs. baseline |
|---|---:|---:|
| 1 warp × 1 row | 42.433 | Baseline |
| 2×1 | 41.784 | −1.53% |
| 2×2 | 44.410 | +4.66% |
| 1×2 | 44.805 | +5.59% |
| 1×3 | 44.562 | +5.02% |
| 1×4 | 44.637 | +5.20% |
| 2×3 | 44.998 | +6.05% |
| **2×4** | **45.245** | **+6.63%** |
| 4×2 | 42.260 | −0.41% |
| 4×4 | 42.973 | +1.27% |
| 2×8 | 42.363 | −0.16% |
| 1×8 | 41.883 | −1.30% |
| 4×8 | 39.431 | −7.78% |

### Why 2×4 wins

The available evidence supports a balance between:

- Parallel execution across two wavefronts.
- Q8_1 activation reuse across four weight rows.
- Reuse of decoded weights across three destination columns.
- Sufficient instruction-level parallelism.
- Manageable register pressure.
- Limited inter-warp reduction overhead.

**2×4 is the current reference configuration.**

The preserved binary is:

```text
~/mmvq-rdna2-2x4-bin/
```

The preserved source patch is:

```text
~/mmvq-rdna2-2x4.patch
```

---

## 2. Completed IQ4_XS optimization experiments

After establishing the winning geometry, we investigated the dominant IQ4_XS MMVQ kernel.

### Results

| Experiment | Throughput | Difference vs. winner | Outcome |
|---|---:|---:|---|
| **Original 2×4** | **45.245** | — | **Winner** |
| IQ4_XS-specific 1×4 | 44.880 | −0.81% | Rejected |
| Split SDOT4 accumulation | 45.072 | −0.38% | No improvement |
| Lookup reordering | 45.134 | −0.25% | No improvement |
| Disable inner-loop unrolling | 33.317 | −26.4% | Rejected |
| Partial unrolling ×2 | 37.484 | −17.2% | Rejected |
| Reverse row/column loop order | 45.072 | −0.38% | No improvement |
| Pointer induction | 44.812 | −0.96% | No improvement |
| **RDNA2 DPP reduction** | Pending | — | **Current experiment** |

Small differences under approximately 1% remain inconclusive with two rounds, but none of these variants demonstrated an improvement over the reference.

### Important findings

**Full unrolling is essential for performance.**

| Unrolling | VGPR | Throughput |
|---|---:|---:|
| No unrolling | 78 | 33.317 t/s |
| Partial ×2 | 79 | 37.484 t/s |
| **Full ×4** | **118** | **45.245 t/s** |

This suggests that the instruction-level parallelism and scheduling flexibility provided by full unrolling are more valuable than reducing register usage alone.

**Pointer induction did not help.**

It eliminated three 64-bit multiply-add instructions from the computation loop, but LLVM changed the memory-access instructions from `global_load` to `flat_load`.

The resulting kernel had:

- 22 additional static instructions in the identified hot-loop region.
- Six additional VGPRs.
- More address additions and memory waits.
- No reduction in the number of memory loads.

The original addressing implementation remains preferable.

**Changing loop order did not meaningfully help.**

Row-first scheduling changed instruction ordering but preserved nearly all instruction counts and the same register allocation.

This suggests LLVM already handles the original computation order reasonably efficiently.

---

## 3. GPU profiling: Where execution time goes

We profiled the winning 2×4 implementation using `rocprofv3`.

### GPU execution-time distribution

| Kernel category | Time | Share of total |
|---|---:|---:|
| **All MMVQ kernels** | **53,511 ms** | **82.74%** |
| IQ4_XS | 15,967 ms | 24.69% |
| Q4_K | 11,917 ms | 18.43% |
| Q5_K | 10,530 ms | 16.28% |
| IQ3_S | 6,048 ms | 9.35% |
| Q6_K | 2,884 ms | 4.46% |
| IQ3_XXS | 2,117 ms | 3.27% |
| Other MMVQ formats | Remaining | Smaller shares |

**This is one of our most important findings.**

More than 82% of measured GPU kernel execution time occurs in MMVQ.

This makes MMVQ a justified optimization target.

However, IQ4_XS is not the entire problem.

The combined Q4_K and Q5_K kernels account for approximately **34.7% of total GPU execution time**, more than IQ4_XS alone.

That gives us another significant optimization direction once the current IQ4_XS work is exhausted.

---

## 4. IQ4_XS assembly findings

We extracted and disassembled the actual `gfx1030` GPU code.

Target kernel:

```cpp
mul_mat_vec_q<(ggml_type)23, 3, false, false, false>
```

### Compiled kernel characteristics

| Metric | Value |
|---|---:|
| Code size | 5,632 bytes |
| Static instructions | 925 |
| VGPR | 118 in code-object metadata |
| SGPR | 30 |
| LDS | 1,536 bytes |
| Scratch | 0 |
| Workgroup size | 64 threads |
| Average kernel execution | Approximately 80 µs |

### Instruction distribution

| Instruction category | Count |
|---|---:|
| `v_perm_b32` | 96 |
| `v_dot4c_i32_i8` | 96 |
| Global loads | 21 |
| `ds_bpermute_b32` | 60 |
| All LDS operations | 78 |
| Memory waits | 85 |

### Main computation loop

We isolated approximately 455 static instructions in the repeating computation section.

| Operation | Count |
|---|---:|
| Integer dot products | 96 |
| Byte permutations | 96 |
| Global loads | 21 |
| Address additions | 28 |
| 64-bit MAD | 3 |
| Memory waits | 10 |

The key discovery is that **the compiler is already performing substantial reuse and optimization**.

It uses native integer dot-product instructions, groups memory loads, and appears to share decoded weight values across output columns.

This limits the usefulness of simple source-level modifications.

---

# 5. Current research path: RDNA2 DPP reduction

**Status: Experimental patch proposed; build, correctness, and performance results not yet reported.**

We discovered that the existing floating-point reduction uses:

```cpp
for (int offset = width/2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
}
```

On RDNA2, this compiles to sequences such as:

```asm
ds_bpermute_b32
s_waitcnt lgkmcnt(0)
v_add_f32_e32
```

There are five XOR shuffle stages per accumulator.

With twelve output accumulators, the compiler generates 60 `ds_bpermute_b32` instructions.

### Proposed change

Use RDNA2 Data Parallel Primitives for the XOR operations that remain within a 16-lane group.

| Reduction stage | Original | Proposed |
|---|---|---|
| XOR 16 | `ds_bpermute` | `ds_bpermute` |
| XOR 8 | `ds_bpermute` | DPP |
| XOR 4 | `ds_bpermute` | DPP |
| XOR 2 | `ds_bpermute` | DPP |
| XOR 1 | `ds_bpermute` | DPP |

**Theoretical target:**

- Reduce `ds_bpermute` instructions from 60 to approximately 12.
- Replace the other 48 with DPP operations.
- Reduce dependency on the LDS permutation pipeline.
- Preserve the same numerical reduction order.

The patch is intended to affect only the RDNA2 IQ4_XS, three-column, 2×4 specialization.

### What remains to be verified

The proposed DPP implementation is not yet validated. In particular, its lane controls must produce the intended XOR permutations, and the generated ISA must be checked rather than assumed.

The next sequence is to compile, run the backend correctness tests, inspect the generated DPP instructions, and benchmark for two rounds.

### Expected performance impact

This optimization targets the **post-loop reduction**, not the repeatedly executed IQ4_XS dot-product loop.

Therefore, even if DPP substantially improves the reduction itself, its effect on total generation throughput might be small.

It is nevertheless worth testing because it changes the hardware instructions rather than merely rearranging source operations.

---

# 6. Older improvement candidates worth revisiting

These are the remaining research directions, ordered by their likely usefulness given our current evidence.

| Priority | Candidate | Status | Potential benefit |
|---|---|---|---|
| **1** | RDNA2 DPP reductions | **Current experiment** | Lower wavefront-reduction overhead |
| **2** | Q4_K MMVQ optimization | Unexplored in depth | Large contribution to total GPU time |
| **3** | Q5_K MMVQ optimization | Unexplored in depth | Large contribution to total GPU time |
| **4** | IQ3_S MMVQ optimization | Unexplored in depth | Smaller but significant GPU-time share |
| 5 | IQ4_XS lookup dependency reduction | Partially explored | Shorter lookup/decode dependency chains |
| 6 | RDNA2-specific memory prefetch | Not tested | Potential latency hiding |
| 7 | IQ4_XS activation-loading strategy | Analyzed, no targeted alternative tested | Potential memory scheduling improvements |
| 8 | Alternative reduction layouts | Not tested | Fewer reductions or different inter-warp coordination |
| 9 | Different register/occupancy tradeoffs | Extensively explored indirectly | Improved active-wave utilization |
| 10 | Broader GGML graph/kernel optimization | Not investigated | Reduce overhead outside individual MMVQ kernels |

## Candidate A — Optimize Q4_K and Q5_K

This is now one of the strongest opportunities.

Our profiling showed:

**Q4_K + Q5_K = approximately 34.7% of total measured GPU kernel execution time.**

Both formats are used substantially in the model.

We have not yet performed the same detailed register, instruction, and memory analysis for these kernels that we performed for IQ4_XS.

Possible targets include their quantization decoding, native integer-dot-product utilization, instruction dependencies, and scheduling.

**Recommendation:** After DPP, investigate these formats before spending many more experiments on minor IQ4_XS changes.

## Candidate B — Reduce IQ4_XS nonlinear lookup dependencies

The IQ4_XS kernel uses:

```cpp
get_int_from_table_16(...)
```

LLVM generates 96 `v_perm_b32` instructions for the specialized kernel.

Our previous lookup-reordering experiment didn't improve throughput.

However, that experiment doesn't exhaust all possibilities for reducing the dependencies involved in nonlinear decoding.

Potential remaining approaches include alternative lookup representations or instruction sequences.

Any new approach should first demonstrate fewer instructions or a shorter dependency chain in the generated ISA.

## Candidate C — Prefetch IQ4_XS weights on RDNA2

The source contains an explicit prefetch path:

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
```

That path was designed for DGX Spark, not RDNA2.

We could investigate a ROCm-specific alternative that brings future weight blocks closer to the cache before they're consumed.

The challenge is that the compiler already interleaves loads and arithmetic reasonably well.

An explicit prefetch could therefore help, have no effect, or increase memory traffic.

It needs careful isolated testing.

## Candidate D — Alternative reduction strategy

Instead of replacing individual shuffle instructions, another possibility is to change how partial sums are distributed across wavefront lanes.

This could potentially reduce the number of reductions or their dependencies.

However, changing the accumulation layout risks reducing the data reuse that makes 2×4 effective.

This is a more invasive optimization than the DPP experiment.

## Candidate E — Occupancy and register pressure

We established:

| Configuration | VGPR | Performance |
|---|---:|---:|
| 1×4 IQ4_XS | 117 | 44.880 t/s |
| **2×4** | **118** | **45.245 t/s** |
| 2×8 | 181 | 42.363 t/s |
| 4×4 | 118 | 42.973 t/s |
| Partial unroll ×2 | 79 | 37.484 t/s |

Register pressure matters, particularly at 181 VGPRs.

But lower register usage is not sufficient to improve throughput.

Further occupancy experiments should be based on a specific architectural change rather than simply minimizing VGPR allocation.

---

# 7. Candidates we should deprioritize

Some directions have already produced enough negative evidence that repeating similar experiments would probably be inefficient.

| Direction | Reason |
|---|---|
| Disable IQ4_XS unrolling | −26.4% throughput |
| Partial unrolling | −17.2% throughput |
| Increase rows to 8 | Significant performance regression |
| Increase warps to 4 | Additional parallelism did not help |
| Reverse loop order | Changed ISA scheduling without improvement |
| Pointer induction | More instructions and registers, no improvement |
| Simple IQ4_XS lookup reordering | Already tested |
| Split SDOT4 accumulation | Already tested |
| Explicit caching of Q8_1 activations | Compiler already appears to exploit substantial reuse |

These aren't proven impossible avenues, but they have lower priority than unexplored work with stronger evidence of potential benefit.

---

# 8. Recommended research roadmap

### Phase 1 — Finish DPP reduction

**Current phase.**

Compile and validate the experimental RDNA2 DPP reduction.

Compare:

- `ds_bpermute` count.
- DPP instruction count.
- LDS wait instructions.
- VGPR allocation.
- Correctness.
- Generation throughput.

If it doesn't improve performance, preserve the experiment and restore 2×4.

### Phase 2 — Profile Q4_K and Q5_K individually

Extract the dominant kernel specializations and determine:

- Register allocation.
- Memory instruction distribution.
- Quantization-decoding cost.
- Native dot-product utilization.
- Reduction overhead.

Identify the strongest bottleneck for each quantization format.

### Phase 3 — Optimize the highest-cost quantization kernel

Select one format based on the evidence from Phase 2.

Prioritize an architectural or instruction-level improvement rather than another source-level loop rearrangement.

### Phase 4 — Reevaluate IQ4_XS

Return to IQ4_XS only if there is a specific new hypothesis, such as a demonstrably shorter nonlinear lookup dependency chain or a viable RDNA2-specific prefetch strategy.

### Phase 5 — Broader inference optimizations

Once MMVQ-specific gains begin to plateau, examine other costs in the inference pipeline.

Possible subjects include non-MMVQ kernels, launch overhead, synchronization, and interactions with MTP speculative decoding.

These need separate profiling before assigning priority.

---

## 9. Preserved research artifacts

| Artifact | Location |
|---|---|
| Winning 2×4 binary | `~/mmvq-rdna2-2x4-bin/` |
| Winning patch | `~/mmvq-rdna2-2x4.patch` |
| Main development worktree | `~/llama-rocm-mmvq-2x1/` |
| IQ4_XS disassembly | `~/mmvq-isa-2x4/iq4xs-2x4.s` |
| ISA comparisons | `~/mmvq-isa-compare/` |
| Profiling data | `~/mmvq-iq4-profile/` |
| Partial-unroll experiment | `~/mmvq-rdna2-iq4xs-unroll2-bin/` |
| Row-first experiment | `~/mmvq-rdna2-iq4xs-rowfirst-bin/` |
| Pointer-induction experiment | `~/mmvq-rdna2-iq4xs-pointer-bin/` |

The source worktree was last verified to match the winning 2×4 patch. The proposed DPP modification has not yet been reported as built or tested.

---

## Research conclusion

We have established three important things.

**First, geometry matters.** The 2×4 configuration improves throughput by 6.63% without changing the underlying quantization mathematics.

**Second, LLVM is already producing reasonably optimized IQ4_XS code.** Native SDOT4, activation reuse, weight-decoding reuse, and aggressive unrolling all contribute to performance. Several source-level rearrangements failed to improve the generated instructions.

**Third, register reduction alone is not the answer.** The experiments demonstrate that instruction-level parallelism and scheduling freedom can be more valuable than lower VGPR counts.

### Where I would focus next

Finish the **RDNA2 DPP reduction experiment**, then shift attention to **Q4_K and Q5_K**.

That shift is especially important because the two formats together account for approximately **34.7% of measured GPU execution time**.

We have investigated IQ4_XS extensively, but Q4_K and Q5_K remain comparatively unexplored.

**The main opportunity is no longer simply improving one quantization kernel. It is applying what we've learned to the other MMVQ kernels that collectively dominate inference time.**
