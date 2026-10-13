collection of reports regarding optimization of qwen3.8 27b, through llama-rocm and windows/linux
aim to improve performance using mtp on Q4 models, inspect kernel times, and optimize for gfx1030 (RX6800)
this research is assisted with AI (gpt, gemini, deepseek, and occasionally claude)

valuable kernel/program edits:
- 2 warps x 4 rows on RDNA2 gfx1030
- 128-bit header (very benefficial)
- sing mask multiplication (reduce instructions)
- mtp2 enabled
- potential combination of kernels is promising

did not read: make kernel faster by making it smaller, gpu good scheduler, human bad programmer.

progress
```
QWEN3.8 27B — PERFORMANCE OPTIMIZATION
│
├── 01. GPU Scheduling — mmvq.cu
│   │
│   ├── ✅ RDNA2: 2 warps × 4 rows (V1 preserved)
│   ├── ✅ Warp/row sweep completed
│   ├── ✅ 2×4 fastest in controlled tests
│   ├── ❌ Q4_K scheduling sweep — no improvement
│   └── ❌ Q5_K tile sweep — 8 variants, no improvement
│
├── 02. Quantized Kernels — vecdotq.cuh
│   │
│   ├── Q4_K
│   │   ├── ✅ Packed 128-bit header (V1 preserved)
│   │   ├── ❌ Wave-shuffle header broadcast
│   │   ├── ❌ DPP reduction / hybrid DPP
│   │   └── 🔬 Q8 activation-sum reuse — gain unconfirmed
│   │
│   ├── Q5_K
│   │   ├── ✅ Packed 128-bit header (V1 preserved)
│   │   ├── ❌ Fifth-bit extraction hoist
│   │   ├── ❌ Alternate tile shapes
│   │   └── 🔬 gfx1030 ISA / high-bit decode investigated
│   │
│   ├── IQ3_S
│   │   ├── ✅ Sign-mask multiplication (V1 preserved)
│   │   ├── ❌ Sign-mask addition
│   │   └── ❌ LDS lookup-table staging
│   │
│   ├── IQ4_XS
│   │   ├── ✅ Largest measured 3-column MMVQ hotspot
│   │   ├── ❌ Ordered-lookup experiment
│   │   └── 🟠 Load organization / latency / VGPR optimization
│   │
│   ├── IQ3_XXS — UPDATED
│   │   ├── ✅ gfx1030 sign-mask multiplication implemented
│   │   ├── ✅ Mathematical equivalence established
│   │   ├── ✅ Compiled RDNA2 instructions inspected
│   │   └── 🟠 Isolated performance improvement unconfirmed
│   │
│   ├── Q6_K    — Pending
│   ├── Q3_K    — Pending
│   ├── IQ2_S   — Pending
│   ├── Q8_0    — Pending
│   ├── IQ2_XS  — Pending
│   ├── Q2_K    — Pending
│   └── IQ4_NL  — Pending
│
├── 03. GPU Profiling — ROCm / gfx1030
│   │
│   ├── ✅ GPU kernel timeline captured
│   ├── ✅ MMVQ: 75.62% of measured GPU kernel time
│   ├── ✅ Nine hardware-counter groups captured
│   ├── ✅ 2,165,481 profiling records
│   ├── ✅ Primary hotspots identified:
│   │      IQ4_XS / Q4_K / Q5_K / IQ3_S
│   │
│   ├── ✅ IQ3_XXS GPU ISA inspected
│   ├── ✅ Packed integer dot-product instructions confirmed
│   └── 🟠 Per-dispatch resource and latency analysis pending
│
├── 04. HIP Graphs — UPDATED
│   │
│   ├── ✅ HIP Graphs enabled
│   ├── ✅ Graph capture and replay confirmed
│   ├── ✅ 144 graph reuses observed
│   ├── ✅ gfx1030 graph execution operational
│   └── 🟠 Independent throughput benefit not established
│
├── 05. MTP Speculative Decoding
│   │
│   ├── ✅ Two-token MTP operational
│   ├── ✅ F16 speculative KV cache
│   ├── ✅ Three-column MMVQ optimization applied
│   ├── ✅ Current MTP2 benchmark completed
│   └── 🟠 Further optimization / acceptance-efficiency analysis
│
├── 06. Platform Research
│   │
│   ├── Windows
│   │   └── Historical optimization and performance experiments
│   │
│   └── Linux
│       ├── ✅ ROCm 10.0
│       ├── ✅ RX 6800 / gfx1030
│       ├── ✅ Dedicated optimized development build
│       └── ✅ New live performance baseline
│
└── 07. Benchmarking — UPDATED
    │
    ├── ✅ Development binary isolated
    ├── ✅ Local IQ4_XS model
    ├── ✅ MTP2 / 8192 context / F16 KV
    ├── ✅ Three fresh 1000-token measurements
    ├── ✅ Results saved independently
    └── ✅ Historical performance figures excluded
```
