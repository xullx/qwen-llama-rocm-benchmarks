collection of reports regarding optimization of qwen3.8 27b, through llama-rocm and windows/linux
aim to improve performance using mtp on Q4 models, inspect kernel times, and optimize for gfx1030 (RX6800)
this research is assisted with AI (gpt, gemini, deepseek, and occasionally claude)

progress
```
ggml/src/ggml-cuda/
│
├── mmvq.cu
│   └── GPU Scheduling (RDNA2, ncols_dst == 3)
│       ├── ✅ Applied: 2 warps × 4 rows (preserved V1 baseline)
│       ├── ✅ Measured warp/row sweep: 2×4 fastest in controlled tests
│       ├── ❌ Q4_K scheduling sweep: no improvement over 2×4
│       └── ❌ Q5_K tile sweep: 8 variants; none beat V1
│
└── vecdotq.cuh
    └── Quantized Operations
        ├── Q4_K
        │   ├── ✅ Packed 128-bit header (preserved V1)
        │   ├── ❌ Wave-shuffle header broadcast
        │   ├── ❌ DPP reduction / hybrid DPP
        │   └── 🔬 Q8 activation-sum reuse tested; gain unconfirmed
        │
        ├── Q5_K
        │   ├── ✅ Packed 128-bit header (preserved V1)
        │   ├── ❌ Fifth-bit extraction hoist (no validated gain)
        │   ├── ❌ Alternate tile shapes (no measured win)
        │   └── 🔬 gfx1030 ISA / high-bit decode investigated
        │
        ├── IQ3_S
        │   ├── ✅ Sign-mask multiplication (preserved V1)
        │   ├── ❌ Sign-mask addition
        │   └── ❌ LDS lookup-table staging
        │
        ├── IQ4_XS
        │   ├── 🔬 Largest measured MMVQ kernel-time hotspot (3 columns)
        │   ├── ❌ Ordered-lookup experiment
        │   └── 🟠 Load organization / latency / VGPR analysis — Next
        │
        ├── IQ3_XXS — Profiled; optimization pending
        ├── Q6_K    — Pending
        ├── Q3_K    — Pending
        ├── IQ2_S   — Pending
        ├── Q8_0    — Pending
        ├── IQ2_XS  — Pending
        ├── Q2_K    — Pending
        └── IQ4_NL  — Pending

Profiling (ROCm / gfx1030)
├── ✅ GPU timeline: MMVQ ~75.62% of measured GPU kernel time
├── ✅ Nine-group hardware-counter capture: 2,165,481 records
├── ✅ Hotspots: IQ4_XS, Q4_K, Q5_K, IQ3_S
└── 🟠 Per-dispatch GPU ISA / resource deep dive — Pending
```
