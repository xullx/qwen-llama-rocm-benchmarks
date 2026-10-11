collection of reports regarding optimization of qwen3.8 27b, through llama-rocm and windows/linux
aim to improve performance using mtp on Q4 models, inspect kernel times, and optimize for gfx1030 (RX6800)
this research is assisted with AI (gpt, gemini, deepseek, and occasionally claude)

progress
ggml/src/ggml-cuda/
│
├── mmvq.cu
│   │
│   └── GPU Scheduling
│       │
│       ├── Applied
│       │   └── RDNA2: 2 warps × 4 rows
│       │
│       └── Tested / Rejected
│           └── Q4_K scheduling sweep
│               ├── 16 combinations (1×1 to 8×8)
│               └── Winner: 2×4 (original)
│
└── vecdotq.cuh
    │
    └── Quantized Operations
        │
        ├── Q4_K
        │   ├── ✅ Packed 128-bit header
        │   ├── ❌ Wave-shuffle header broadcast
        │   └── 🟠 Q8 sum reuse — Planned
        │
        ├── Q5_K
        │   └── ✅ Packed 128-bit header
        │
        ├── IQ3_S
        │   ├── ✅ Sign-mask multiplication
        │   ├── ❌ Sign-mask addition
        │   └── ❌ LDS lookup-table staging
        │
        ├── IQ4_XS
        │   └── 🔬 Memory bottleneck investigated
        │
        ├── IQ3_XXS — Pending
        ├── Q6_K    — Pending
        ├── Q3_K    — Pending
        ├── IQ2_S   — Pending
        ├── Q8_0    — Pending
        ├── IQ2_XS  — Pending
        ├── Q2_K    — Pending
        └── IQ4_NL  — Pending
