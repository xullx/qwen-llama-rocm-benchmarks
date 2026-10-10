# Qwen3.8 27B MTP — Optimization Report

AMD RX 6800 · ROCm 10.0 · llama.cpp · Experimental results through October 2026

We've explored sampling parameters, MTP speculative decoding, KV precision, and custom modifications to llama.cpp. The model is now substantially optimized, although 50 t/s sustained remains unachieved.

## 1. Current performance

Best long-run throughput

# 44.77

tokens/second

MTP draft acceptance

# 80.04%

Full 7,263-token run

Mean speculative length

# 2.60

At n-max = 2

Best measured sustained throughput

44.77 / 50 t/s

89.5% of the 50 t/s target. Short three-second windows have exceeded 50 t/s, but the full-run average has not.

The best original ROCm run produced 7,263 tokens in 162.19 seconds of decode time, with an 80.039% draft acceptance rate and 2.60 mean speculative length.

Reasoning remained enabled throughout the main optimization experiments.

## 2. Final best-known model configuration

| Component                | Setting                                                 |
| ------------------------ | ------------------------------------------------------- |
| GPU                      | AMD RX 6800, 16 GB, gfx1030                             |
| Backend                  | Official llama.cpp ROCm                                 |
| Model                    | Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_XXS-mtp.gguf |
| Context                  | 8,192                                                   |
| GPU layers               | 99                                                      |
| CPU threads              | 8                                                       |
| Batch threads            | 4                                                       |
| Batch / microbatch       | 1024 / 128                                              |
| Flash Attention          | ON                                                      |
| Main KV                  | F16 / F16                                               |
| Draft KV                 | F16 / F16                                               |
| MTP speculation          | ON, `draft-mtp`                                         |
| Draft maximum            | 2                                                       |
| Draft probability cutoff | 0.0                                                     |
| Parallel slots           | 1                                                       |
| Reasoning                | ON                                                      |
| Reasoning effort         | low                                                     |
| Reasoning budget         | −1, unlimited                                           |
| Reasoning preservation   | OFF                                                     |
| Multimodal projector     | OFF                                                     |
| Backend sampling         | OFF                                                     |

### Optimized sampler settings

| Parameter        | Selected value |
| ---------------- | -------------- |
| Temperature      | 0.5            |
| Top-p            | 1.0            |
| Top-k            | 20             |
| Min-p            | 0.08           |
| Repeat penalty   | 1.0            |
| Presence penalty | 0.0            |

These are the best-known practical settings from our exploratory tests, rather than statistically proven global optima.

## 3. Sampling experiments

### Temperature

| Temperature | Observed performance           | Finding           |
| ----------- | ------------------------------ | ----------------- |
| 0.4         | \~43.26 t/s over \~7.3K tokens | Good              |
| 0.5         | 44.77 t/s over 7.3K tokens     | Selected          |
| 0.6         | \~42.69 t/s at 1,294 tokens    | Worse in this run |

The temperature 0.4 setup also differed in draft-KV configuration during earlier testing, so this was not a perfectly isolated temperature comparison.

Temperature 0.5 produced the best longer sustained result.

### Min-p

40 t/s42 t/s44 t/s46 t/s48 t/s0.040.060.080.100.12

Exploratory runs of different lengths. Values are cumulative speeds near the indicated stopping points, not controlled matched-workload averages.

| Min-p | Measured tg | Generated tokens |
| ----- | ----------- | ---------------- |
| 0.04  | 42.53       | 1,548            |
| 0.06  | 43.43       | 2,374            |
| 0.08  | 44.54       | 2,839            |
| 0.10  | 42.87       | 2,856            |
| 0.12  | 43.32       | 2,234            |

Conclusion: 0.08 performed best in the recorded sweep. Both higher and lower values gave worse results in their respective runs.

### Top-p

With temperature 0.5, min-p 0.08 and top-k 20:

| Top-p | Measured tg | Approx. depth |
| ----- | ----------- | ------------- |
| 0.90  | 43.53       | 3,162         |
| 0.95  | 44.20       | 3,221         |
| 1.00  | 44.39       | 3,228         |

Setting top-p to 1.0 effectively removes nucleus pruning, leaving top-k and min-p to restrict candidate tokens.

### Top-k

With temperature 0.5, min-p 0.08 and top-p 1.0:

40 t/s42 t/s44 t/s46 t/s48 t/s010152030

Cumulative generation speed at approximately 3,000–3,350 generated tokens. Higher is better.

| Top-k | Result | Finding                   |
| ----- | ------ | ------------------------- |
| 0     | 42.48  | Disabling performed worse |
| 10    | 43.99  | Competitive               |
| 15    | 42.22  | Unexpectedly poor         |
| 20    | 44.39  | Selected                  |
| 30    | 43.06  | Worse than 20             |

The non-monotonic results, particularly at k=15, suggest generated-text variability matters substantially. Still, k=20 is the best observed setting.

## 4. MTP speculative decoding experiments

This is where most of our remaining optimization effort has been concentrated.

### Speculative depth

| MTP n-max | Measured generation speed | Outcome    |
| --------- | ------------------------- | ---------- |
| 2         | 44.77 t/s, long run       | Best       |
| 3         | 39.31 t/s at 478 tokens   | Worse      |
| 4         | 36.29 t/s at 773 tokens   | Much worse |

Increasing speculative depth increases the amount of draft and verification work. With this model, higher depths became slower very quickly.

Decision: keep `n-max = 2`.

### Draft KV precision

| Draft KV    | Observed throughput                                                | Finding                    |
| ----------- | ------------------------------------------------------------------ | -------------------------- |
| F16 / F16   | \~44.3 t/s around 1.6K tokens                                      | Selected                   |
| Q8_0 / Q8_0 | \~43.6 t/s around 1.7K tokens                                      | Slightly slower            |
| F32 / F32   | Not tested                                                         | Unexplored                 |
| Q4_0 / Q4_0 | Earlier experimentation, not an equally controlled comparison here | No established improvement |

Q8 reduces draft-cache precision and storage, but it did not improve generation speed in our observed run.

### Backend sampling

We enabled `--backend-sampling` hoping to reduce sampling overhead.

However, llama.cpp logged:

```
backend sampling is not compatible with reasoning budget, disabling
```

The long run produced 44.77 t/s, but that result cannot be attributed to backend sampling, because llama.cpp disabled it.

Decision: leave it OFF.

## 5. Custom llama.cpp source modifications

We moved from command-line optimizations into modifying the MTP implementation itself.

### Experiment A — Adaptive n=1/2 drafting

Objective: Avoid generating the second speculative token when the confidence score is too low.

We added a new environment variable:

`LLAMA_MTP_P2_MIN`

The following thresholds were tested:

| Token-2 threshold | Measured tg | Generated tokens |
| ----------------- | ----------- | ---------------- |
| 0.60              | 43.14       | 1,827            |
| 0.70              | 43.76       | 3,051            |
| 0.80              | 42.23       | 1,536            |

All three were less compelling than fixed n=2. The differing generation lengths prevent a precise estimate of the regression.

The likely explanation is that skipping token #2 sacrifices enough successful speculation to outweigh the work saved.

Decision: adaptive confidence-based n=1/2 drafting was not adopted.

### Experiment B — Deferred verification catch-up

This was our more substantial source-level optimization.

Previously, the draft context caught up on every target-verified row, including rows that might subsequently be rejected.

We modified the code to defer catch-up until the accepted prefix was known.

Original

Target verifies 3 rows

Process all 3 in draft context

Discard rejected state

Modified

Target verifies 3 rows

Determine accepted prefix

Process only needed rows

The modified source compiled successfully using your custom gfx1030 build with HIP graphs OFF.

### Actual computational savings

Original catch-up rows

# 3,933

Optimized rows

# 3,247

Rows eliminated

# 17.44%

The optimization eliminated 686 rows across 1,311 verification calls.

### End-to-end performance

| Mode                          | Tokens | Generation speed |
| ----------------------------- | ------ | ---------------- |
| Custom binary, patch disabled | 3,107  | 42.72 t/s        |
| Custom binary, patch enabled  | 3,135  | 43.16 t/s        |

Observed difference: +1.03%, one run per mode.

We successfully reduced redundant draft-context work. The improvement in overall throughput is much smaller and remains unconfirmed statistically.

Also, compilation and one successful generation run do not yet establish complete behavioral correctness of the modified implementation.

## 6. Official ROCm versus custom gfx1030 build

This is the current unresolved question.

| Build                                      | Status                            |
| ------------------------------------------ | --------------------------------- |
| Official ROCm, unmodified                  | Best observed long run: 44.77 t/s |
| Custom gfx1030, graphs OFF, patch disabled | 42.72 t/s at \~3.1K               |
| Custom gfx1030, graphs OFF, patch enabled  | 43.16 t/s at \~3.1K               |

The original binary used a different compiler/toolchain from our custom build. Earlier IQ4_XS tests did not show a meaningful standalone gfx1030 compilation advantage either.

But the runs above have not yet isolated build differences from the variability of sampled MTP output.

Current experiment underway: an automated three-way comparison using the same prompt, seed, settings, and generation limit.

That result should tell us whether to retain the custom binary, reproduce the official build configuration, or discard the deferred-catch-up optimization.

## 7. Optimization status and remaining research

| Experiment                             | Status                     | Conclusion                                       |
| -------------------------------------- | -------------------------- | ------------------------------------------------ |
| ROCm + IQ3_XXS MTP                     | Working                    | Selected                                         |
| Sampler tuning                         | Complete exploratory sweep | Temp .5, top-p 1, top-k 20, min-p .08            |
| MTP depths 2/3/4                       | Tested                     | Depth 2 best                                     |
| Draft F16 vs Q8                        | Tested                     | F16 favored                                      |
| Backend sampling                       | Tested                     | Disabled due to reasoning-budget incompatibility |
| Adaptive second-token drafting         | Tested                     | No improvement                                   |
| Deferred verify catch-up               | Implemented                | 17.44% less catch-up work                        |
| Custom vs official binary              | In progress                | Not yet resolved                                 |
| Accepted-prefix correctness validation | Pending                    | Needed before daily adoption                     |
| Per-stage GPU/kernel profiling         | Pending                    | Needed to locate the next bottleneck             |
| MTP kernel fusion                      | Pending                    | More complex source-level research               |
| ROCm MMQ / hipBLAS path                | Pending                    | Separate possible optimization                   |
| Fast-math compiler variants            | Pending                    | Potential speed/correctness tradeoff             |

## 8. Current conclusion

Your best-known production configuration remains the original official ROCm binary with fixed two-token MTP and F16 draft KV.

The sampler combination to retain is:

```
'--temp', '0.5',
'--top-p', '1.0',
'--top-k', '20',
'--min-p', '0.08'
```

And MTP:

```
'--spec-type', 'draft-mtp',
'--spec-draft-n-max', '2',
'--spec-draft-type-k', 'f16',
'--spec-draft-type-v', 'f16',
'--spec-draft-p-min', '0.0'
```

We now have two important findings: 80% accepted draft tokens make two-token MTP highly effective, while eliminating 17.44% of redundant catch-up rows yielded only about 1% observed end-to-end improvement.

The next serious gain is more likely to come from reducing the actual GPU time spent in draft generation or target verification, rather than squeezing another percentage point from sampling parameters.

The immediate next step is to finish the official-versus-custom controlled benchmark. That will establish the right binary to optimize further.
