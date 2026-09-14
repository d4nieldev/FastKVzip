# Complete method comparison

All scores are absolute scores on a 0-100 scale; higher is better. Columns are retained-context ratios, not averages over ratios. The full-cache column is measured independently for each method. The same suite has 60 configurations, 18,531 contexts, and 28,236 questions per method.

Bold aggregate rows are arithmetic means of unrounded configuration scores, computed independently for each column. SCBench 11-base-task mean excludes the 16 length variants; SCBench 27-configuration mean includes them equally. RULER selected 33 weights the 4K, 8K, and selected 64K groups by 13/33, 13/33, and 7/33. These are descriptive prepared-split summaries, not official complete-benchmark scores; the overall mean also mixes heterogeneous metrics. Display values are rounded only after aggregation.

The 5% and 10% columns are additionally reported for the 11 SCBench base-task configurations and their aggregate mean only. The 16 SCBench length variants and all 33 RULER configurations were not evaluated at these two ratios and show — in both columns, as do the aggregate rows that include them.

All methods use Qwen2.5-7B-Instruct-1M. FastKVzip uses its official gate and native 16,000-token chunked eviction; GraphKV and KVzip prune after full-context prefill. GraphKV and FastKVzip have a 0.02 protected window; KVzip has none. The inherited 48-token generation limit for Summary + needles applies to both summary and retrieval questions. Mixed-task configurations retain one combined score each. This compares deployed method pipelines, not just architectures.

## GraphKV (ours)

post-prefill pruning; 2% protected local window. Source: [c0s997un](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/c0s997un).

| Benchmark / configuration | 5% | 10% | 20% | 30% | 40% | 50% | 75% | Full (100%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCBench KV | 1.20 | 32.40 | 71.60 | 69.60 | 72.00 | 72.40 | 68.60 | 68.20 |
| SCBench KV (mid) | — | — | 63.20 | 68.60 | 70.40 | 70.60 | 69.20 | 68.90 |
| SCBench KV (short) | — | — | 70.50 | 83.40 | 85.30 | 86.30 | 84.70 | 85.50 |
| SCBench KV (tiny) | — | — | 68.20 | 85.90 | 89.70 | 90.20 | 88.90 | 88.00 |
| SCBench MF | 16.50 | 29.67 | 35.33 | 37.33 | 35.83 | 33.50 | 33.67 | 33.00 |
| SCBench MF (mid) | — | — | 35.50 | 36.83 | 35.00 | 32.67 | 34.33 | 34.83 |
| SCBench MF (short) | — | — | 38.67 | 37.83 | 36.50 | 34.67 | 37.00 | 37.83 |
| SCBench MF (tiny) | — | — | 40.00 | 40.50 | 39.00 | 37.33 | 38.33 | 39.50 |
| SCBench QA (English) | 12.23 | 27.29 | 42.51 | 45.64 | 42.85 | 45.00 | 44.02 | 42.92 |
| SCBench RepoQA | 2.05 | 38.86 | 55.68 | 58.64 | 59.55 | 59.55 | 59.77 | 59.09 |
| SCBench RepoQA (short) | — | — | 70.00 | 70.00 | 70.91 | 70.91 | 72.73 | 72.73 |
| SCBench RepoQA (tiny) | — | — | 64.44 | 73.33 | 75.56 | 71.11 | 75.56 | 75.56 |
| SCBench Summary | 27.65 | 32.22 | 36.05 | 36.30 | 36.23 | 36.51 | 36.77 | 36.82 |
| SCBench Summary (mid) | — | — | 40.36 | 40.60 | 39.76 | 39.94 | 39.95 | 39.17 |
| SCBench Summary (short) | — | — | 40.33 | 40.52 | 39.06 | 40.51 | 40.63 | 39.31 |
| SCBench Summary (tiny) | — | — | 38.08 | 38.21 | 36.95 | 38.29 | 38.34 | 37.12 |
| SCBench Choice (English) | 49.54 | 68.98 | 72.22 | 73.61 | 80.56 | 80.56 | 76.39 | 79.17 |
| SCBench Many-shot | 32.96 | 30.37 | 34.07 | 34.44 | 37.04 | 38.15 | 38.15 | 38.52 |
| SCBench Many-shot (short) | — | — | 32.96 | 33.70 | 35.56 | 36.67 | 34.81 | 35.56 |
| SCBench Many-shot (tiny) | — | — | 33.70 | 34.81 | 36.30 | 34.81 | 35.56 | 35.19 |
| SCBench Prefix/suffix | 0.60 | 10.80 | 30.00 | 42.80 | 43.40 | 40.00 | 44.80 | 51.20 |
| SCBench Prefix/suffix (mid) | — | — | 42.20 | 58.00 | 65.60 | 65.80 | 70.20 | 69.60 |
| SCBench Prefix/suffix (short) | — | — | 34.40 | 61.40 | 76.80 | 81.60 | 86.60 | 88.80 |
| SCBench Prefix/suffix (tiny) | — | — | 62.40 | 82.40 | 87.80 | 88.80 | 90.60 | 91.40 |
| SCBench VT | 40.00 | 45.96 | 42.31 | 40.44 | 41.07 | 41.78 | 41.38 | 41.29 |
| SCBench Summary + needles | 19.33 | 56.59 | 65.47 | 67.63 | 67.97 | 67.92 | 68.18 | 68.35 |
| SCBench RepoQA + KV | 3.55 | 52.41 | 74.86 | 79.26 | 80.11 | 80.40 | 81.25 | 80.82 |
| **SCBench 11-base-task mean** | **18.69** | **38.69** | **50.92** | **53.25** | **54.24** | **54.16** | **53.91** | **54.49** |
| **SCBench 27-configuration mean** | **—** | **—** | **49.45** | **54.51** | **56.18** | **56.15** | **56.68** | **56.98** |
| RULER 4K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 4K NIAH single 2 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 4K NIAH single 3 | — | — | 99.40 | 99.60 | 99.80 | 99.80 | 99.60 | 98.80 |
| RULER 4K NIAH multikey 1 | — | — | 96.00 | 98.40 | 98.60 | 99.20 | 99.20 | 99.00 |
| RULER 4K NIAH multikey 2 | — | — | 98.20 | 99.40 | 99.40 | 99.60 | 99.80 | 99.80 |
| RULER 4K NIAH multikey 3 | — | — | 98.00 | 98.80 | 98.80 | 99.40 | 99.60 | 99.60 |
| RULER 4K NIAH multivalue | — | — | 66.30 | 73.70 | 82.70 | 90.75 | 93.90 | 92.20 |
| RULER 4K NIAH multiquery | — | — | 99.50 | 99.70 | 99.95 | 99.90 | 99.90 | 100.00 |
| RULER 4K VT | — | — | 94.24 | 95.12 | 96.68 | 96.96 | 98.64 | 98.88 |
| RULER 4K CWE | — | — | 93.32 | 97.08 | 97.46 | 97.90 | 98.34 | 98.62 |
| RULER 4K FWE | — | — | 80.33 | 83.13 | 81.73 | 81.67 | 84.67 | 85.80 |
| RULER 4K QA 1 | — | — | 80.40 | 83.40 | 86.20 | 85.60 | 85.20 | 85.20 |
| RULER 4K QA 2 | — | — | 60.40 | 60.80 | 61.80 | 59.60 | 58.80 | 59.60 |
| **RULER 4K 13-task mean** | **—** | **—** | **89.70** | **91.47** | **92.55** | **93.11** | **93.67** | **93.65** |
| RULER 8K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 8K NIAH single 2 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 8K NIAH single 3 | — | — | 99.60 | 99.40 | 99.60 | 99.60 | 99.60 | 99.40 |
| RULER 8K NIAH multikey 1 | — | — | 96.00 | 98.20 | 98.80 | 98.80 | 99.40 | 99.60 |
| RULER 8K NIAH multikey 2 | — | — | 97.20 | 98.60 | 98.80 | 99.00 | 99.80 | 99.80 |
| RULER 8K NIAH multikey 3 | — | — | 90.80 | 95.40 | 95.80 | 95.80 | 97.00 | 97.20 |
| RULER 8K NIAH multivalue | — | — | 61.90 | 74.65 | 78.75 | 86.50 | 85.90 | 84.70 |
| RULER 8K NIAH multiquery | — | — | 99.60 | 99.75 | 99.85 | 99.95 | 99.95 | 99.90 |
| RULER 8K VT | — | — | 89.24 | 88.96 | 94.24 | 94.92 | 97.36 | 97.88 |
| RULER 8K CWE | — | — | 86.80 | 91.26 | 90.58 | 90.70 | 90.70 | 91.02 |
| RULER 8K FWE | — | — | 73.27 | 81.73 | 82.47 | 81.73 | 82.67 | 83.20 |
| RULER 8K QA 1 | — | — | 75.40 | 79.20 | 79.80 | 80.40 | 81.40 | 81.40 |
| RULER 8K QA 2 | — | — | 56.40 | 58.20 | 57.60 | 56.00 | 55.20 | 55.00 |
| **RULER 8K 13-task mean** | **—** | **—** | **86.63** | **89.64** | **90.48** | **91.03** | **91.46** | **91.47** |
| RULER 64K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 64K NIAH multikey 3 | — | — | 89.00 | 90.40 | 92.40 | 91.80 | 91.80 | 91.60 |
| RULER 64K NIAH multivalue | — | — | 64.05 | 76.85 | 80.60 | 86.75 | 85.35 | 82.20 |
| RULER 64K VT | — | — | 80.28 | 78.80 | 79.32 | 80.28 | 78.32 | 77.04 |
| RULER 64K CWE | — | — | 36.74 | 42.80 | 42.08 | 38.50 | 27.74 | 33.80 |
| RULER 64K FWE | — | — | 80.93 | 85.53 | 86.47 | 87.07 | 86.53 | 86.00 |
| RULER 64K QA 2 | — | — | 47.60 | 49.20 | 48.00 | 48.60 | 47.60 | 46.60 |
| **RULER 64K selected 7 mean** | **—** | **—** | **71.23** | **74.80** | **75.55** | **76.14** | **73.91** | **73.89** |
| **RULER selected 33 mean** | **—** | **—** | **84.57** | **87.21** | **88.13** | **88.69** | **88.61** | **88.60** |
| **Overall selected 60 mean** | **—** | **—** | **68.77** | **72.50** | **73.75** | **74.05** | **74.24** | **74.37** |

## Official FastKVzip

native chunked-prefill eviction; 2% protected local window. Source: [vi31uf3h](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/vi31uf3h).

| Benchmark / configuration | 5% | 10% | 20% | 30% | 40% | 50% | 75% | Full (100%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCBench KV | 2.80 | 32.80 | 47.00 | 66.80 | 66.40 | 72.80 | 68.20 | 68.20 |
| SCBench KV (mid) | — | — | 61.80 | 70.10 | 72.50 | 71.00 | 68.90 | 68.90 |
| SCBench KV (short) | — | — | 71.10 | 83.10 | 85.40 | 85.70 | 85.70 | 85.20 |
| SCBench KV (tiny) | — | — | 65.90 | 83.20 | 90.10 | 90.50 | 89.40 | 88.10 |
| SCBench MF | 19.33 | 29.33 | 32.83 | 33.50 | 34.00 | 33.83 | 32.83 | 33.00 |
| SCBench MF (mid) | — | — | 34.33 | 33.00 | 34.50 | 35.17 | 34.17 | 34.83 |
| SCBench MF (short) | — | — | 38.17 | 35.33 | 34.50 | 33.83 | 35.83 | 37.83 |
| SCBench MF (tiny) | — | — | 41.33 | 39.33 | 37.00 | 38.00 | 37.83 | 39.83 |
| SCBench QA (English) | 13.07 | 21.50 | 44.20 | 44.74 | 42.02 | 41.76 | 42.54 | 41.52 |
| SCBench RepoQA | 4.32 | 44.09 | 57.73 | 59.32 | 60.23 | 61.36 | 58.18 | 59.09 |
| SCBench RepoQA (short) | — | — | 69.09 | 67.27 | 70.00 | 71.82 | 72.73 | 72.73 |
| SCBench RepoQA (tiny) | — | — | 71.11 | 73.33 | 75.56 | 75.56 | 75.56 | 75.56 |
| SCBench Summary | 27.64 | 33.18 | 35.97 | 36.82 | 36.49 | 36.80 | 36.83 | 36.82 |
| SCBench Summary (mid) | — | — | 40.60 | 39.88 | 40.03 | 39.55 | 38.76 | 39.17 |
| SCBench Summary (short) | — | — | 40.60 | 39.88 | 40.03 | 39.55 | 38.76 | 39.17 |
| SCBench Summary (tiny) | — | — | 38.45 | 38.19 | 37.04 | 37.21 | 37.12 | 37.12 |
| SCBench Choice (English) | 46.30 | 68.98 | 73.61 | 75.00 | 75.00 | 75.00 | 79.17 | 79.17 |
| SCBench Many-shot | 34.81 | 32.59 | 33.70 | 36.67 | 37.04 | 36.30 | 36.30 | 38.52 |
| SCBench Many-shot (short) | — | — | 34.44 | 35.93 | 34.81 | 34.07 | 34.07 | 35.56 |
| SCBench Many-shot (tiny) | — | — | 34.07 | 34.44 | 32.96 | 33.70 | 35.19 | 35.19 |
| SCBench Prefix/suffix | 0.60 | 11.20 | 38.80 | 48.40 | 55.60 | 47.40 | 36.80 | 51.20 |
| SCBench Prefix/suffix (mid) | — | — | 47.80 | 63.20 | 74.80 | 70.80 | 67.20 | 69.60 |
| SCBench Prefix/suffix (short) | — | — | 45.00 | 62.00 | 80.20 | 85.00 | 88.00 | 88.80 |
| SCBench Prefix/suffix (tiny) | — | — | 64.00 | 80.40 | 88.80 | 90.00 | 92.00 | 91.40 |
| SCBench VT | 43.02 | 49.02 | 43.51 | 41.73 | 40.58 | 40.53 | 41.38 | 41.29 |
| SCBench Summary + needles | 16.96 | 54.54 | 65.64 | 68.29 | 67.86 | 67.56 | 68.17 | 68.35 |
| SCBench RepoQA + KV | 7.67 | 54.55 | 77.84 | 79.55 | 80.40 | 80.26 | 80.68 | 80.82 |
| **SCBench 11-base-task mean** | **19.68** | **39.25** | **50.08** | **53.71** | **54.15** | **53.96** | **52.83** | **54.36** |
| **SCBench 27-configuration mean** | **—** | **—** | **49.95** | **54.42** | **56.44** | **56.48** | **56.01** | **56.92** |
| RULER 4K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 4K NIAH single 2 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 4K NIAH single 3 | — | — | 99.40 | 99.60 | 99.40 | 99.80 | 99.40 | 99.00 |
| RULER 4K NIAH multikey 1 | — | — | 96.80 | 98.40 | 98.80 | 99.00 | 99.00 | 98.80 |
| RULER 4K NIAH multikey 2 | — | — | 98.80 | 99.40 | 99.80 | 99.80 | 100.00 | 99.80 |
| RULER 4K NIAH multikey 3 | — | — | 97.60 | 98.60 | 98.60 | 99.60 | 99.60 | 99.60 |
| RULER 4K NIAH multivalue | — | — | 64.25 | 81.20 | 83.30 | 88.45 | 92.90 | 91.90 |
| RULER 4K NIAH multiquery | — | — | 99.50 | 99.95 | 99.75 | 99.80 | 99.95 | 100.00 |
| RULER 4K VT | — | — | 97.40 | 97.44 | 98.12 | 97.60 | 98.52 | 98.88 |
| RULER 4K CWE | — | — | 94.20 | 97.10 | 97.64 | 97.68 | 98.42 | 98.62 |
| RULER 4K FWE | — | — | 79.67 | 82.00 | 81.27 | 81.80 | 84.73 | 85.60 |
| RULER 4K QA 1 | — | — | 79.40 | 83.60 | 84.80 | 84.20 | 85.00 | 85.20 |
| RULER 4K QA 2 | — | — | 58.00 | 60.60 | 60.60 | 59.60 | 59.20 | 59.60 |
| **RULER 4K 13-task mean** | **—** | **—** | **89.62** | **92.15** | **92.47** | **92.87** | **93.59** | **93.62** |
| RULER 8K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 8K NIAH single 2 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 8K NIAH single 3 | — | — | 99.80 | 99.60 | 99.40 | 99.60 | 99.40 | 99.60 |
| RULER 8K NIAH multikey 1 | — | — | 95.80 | 98.40 | 99.00 | 99.40 | 99.60 | 99.60 |
| RULER 8K NIAH multikey 2 | — | — | 98.60 | 99.20 | 99.40 | 99.60 | 99.80 | 100.00 |
| RULER 8K NIAH multikey 3 | — | — | 88.00 | 95.20 | 96.00 | 96.60 | 96.80 | 97.60 |
| RULER 8K NIAH multivalue | — | — | 61.20 | 77.90 | 80.30 | 82.70 | 83.45 | 84.40 |
| RULER 8K NIAH multiquery | — | — | 99.55 | 99.95 | 99.90 | 99.95 | 99.95 | 99.95 |
| RULER 8K VT | — | — | 93.68 | 95.52 | 96.68 | 95.92 | 97.68 | 98.04 |
| RULER 8K CWE | — | — | 88.32 | 91.98 | 92.00 | 91.04 | 90.66 | 90.88 |
| RULER 8K FWE | — | — | 79.07 | 81.53 | 81.20 | 82.20 | 82.47 | 82.73 |
| RULER 8K QA 1 | — | — | 77.00 | 78.60 | 80.80 | 81.20 | 82.20 | 81.60 |
| RULER 8K QA 2 | — | — | 55.80 | 55.20 | 57.20 | 56.00 | 54.80 | 54.40 |
| **RULER 8K 13-task mean** | **—** | **—** | **87.45** | **90.24** | **90.91** | **91.09** | **91.29** | **91.45** |
| RULER 64K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 64K NIAH multikey 3 | — | — | 90.80 | 90.80 | 92.00 | 90.80 | 91.00 | 91.60 |
| RULER 64K NIAH multivalue | — | — | 63.95 | 80.55 | 86.90 | 87.60 | 84.90 | 82.20 |
| RULER 64K VT | — | — | 78.96 | 80.08 | 84.36 | 83.88 | 77.52 | 77.04 |
| RULER 64K CWE | — | — | 32.38 | 43.60 | 33.72 | 28.04 | 28.00 | 33.58 |
| RULER 64K FWE | — | — | 80.93 | 85.20 | 85.13 | 85.93 | 86.60 | 86.00 |
| RULER 64K QA 2 | — | — | 50.20 | 50.20 | 48.80 | 47.40 | 47.60 | 45.60 |
| **RULER 64K selected 7 mean** | **—** | **—** | **71.03** | **75.78** | **75.84** | **74.81** | **73.66** | **73.72** |
| **RULER selected 33 mean** | **—** | **—** | **84.82** | **87.92** | **88.33** | **88.34** | **88.46** | **88.54** |
| **Overall selected 60 mean** | **—** | **—** | **69.13** | **72.85** | **73.98** | **74.00** | **73.86** | **74.31** |

## KVzip

native post-prefill reconstruction scoring; no protected local window. Source: [p2wsdvv5](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/p2wsdvv5).

| Benchmark / configuration | 5% | 10% | 20% | 30% | 40% | 50% | 75% | Full (100%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCBench KV | 0.00 | 11.00 | 36.80 | 68.40 | 67.80 | 66.60 | 66.40 | 68.20 |
| SCBench KV (mid) | — | — | 42.50 | 65.80 | 70.80 | 69.90 | 67.90 | 68.90 |
| SCBench KV (short) | — | — | 63.90 | 80.40 | 85.90 | 85.50 | 84.30 | 85.50 |
| SCBench KV (tiny) | — | — | 72.60 | 84.60 | 88.20 | 89.00 | 88.90 | 88.00 |
| SCBench MF | 13.83 | 30.33 | 35.33 | 35.17 | 34.17 | 34.17 | 32.67 | 33.00 |
| SCBench MF (mid) | — | — | 34.67 | 34.33 | 34.00 | 34.50 | 35.00 | 34.83 |
| SCBench MF (short) | — | — | 38.83 | 36.00 | 35.67 | 35.67 | 37.50 | 37.83 |
| SCBench MF (tiny) | — | — | 42.17 | 39.50 | 38.00 | 38.17 | 38.00 | 39.83 |
| SCBench QA (English) | 15.03 | 37.33 | 42.58 | 46.37 | 43.96 | 43.34 | 40.31 | 41.52 |
| SCBench RepoQA | 2.95 | 32.05 | 51.82 | 57.27 | 59.09 | 58.18 | 59.09 | 59.09 |
| SCBench RepoQA (short) | — | — | 67.27 | 67.27 | 70.00 | 70.91 | 72.73 | 72.73 |
| SCBench RepoQA (tiny) | — | — | 68.89 | 71.11 | 71.11 | 75.56 | 75.56 | 75.56 |
| SCBench Summary | 29.27 | 34.24 | 35.58 | 36.00 | 36.37 | 36.46 | 36.94 | 36.82 |
| SCBench Summary (mid) | — | — | 39.78 | 39.30 | 39.88 | 39.02 | 39.49 | 39.31 |
| SCBench Summary (short) | — | — | 39.78 | 39.30 | 39.88 | 39.02 | 39.49 | 39.31 |
| SCBench Summary (tiny) | — | — | 37.58 | 37.11 | 37.67 | 36.91 | 37.28 | 37.12 |
| SCBench Choice (English) | 57.41 | 68.06 | 72.22 | 77.78 | 76.39 | 77.78 | 77.78 | 79.17 |
| SCBench Many-shot | 32.59 | 32.22 | 33.33 | 34.44 | 35.93 | 36.67 | 37.41 | 38.52 |
| SCBench Many-shot (short) | — | — | 33.33 | 32.22 | 33.70 | 34.07 | 34.07 | 35.56 |
| SCBench Many-shot (tiny) | — | — | 35.93 | 34.07 | 33.33 | 34.44 | 35.56 | 35.19 |
| SCBench Prefix/suffix | 0.00 | 0.00 | 2.80 | 23.00 | 33.00 | 41.00 | 50.20 | 51.20 |
| SCBench Prefix/suffix (mid) | — | — | 8.60 | 43.40 | 59.00 | 65.80 | 69.40 | 69.60 |
| SCBench Prefix/suffix (short) | — | — | 19.20 | 54.20 | 78.40 | 85.00 | 87.60 | 88.80 |
| SCBench Prefix/suffix (tiny) | — | — | 49.80 | 78.20 | 87.60 | 90.80 | 91.00 | 91.40 |
| SCBench VT | 45.51 | 43.64 | 40.67 | 40.84 | 40.71 | 40.80 | 40.31 | 41.29 |
| SCBench Summary + needles | 18.82 | 58.21 | 66.09 | 67.50 | 68.13 | 68.10 | 68.41 | 68.35 |
| SCBench RepoQA + KV | 25.28 | 58.24 | 73.15 | 79.26 | 79.83 | 79.83 | 81.11 | 80.82 |
| **SCBench 11-base-task mean** | **21.88** | **36.85** | **44.58** | **51.46** | **52.31** | **52.99** | **53.69** | **54.36** |
| **SCBench 27-configuration mean** | **—** | **—** | **43.90** | **51.96** | **54.76** | **55.82** | **56.46** | **56.94** |
| RULER 4K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 4K NIAH single 2 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 4K NIAH single 3 | — | — | 99.20 | 99.60 | 99.80 | 99.80 | 99.20 | 98.80 |
| RULER 4K NIAH multikey 1 | — | — | 97.40 | 99.00 | 99.20 | 99.20 | 99.00 | 99.00 |
| RULER 4K NIAH multikey 2 | — | — | 98.80 | 99.40 | 99.80 | 99.80 | 99.80 | 99.80 |
| RULER 4K NIAH multikey 3 | — | — | 97.20 | 99.00 | 99.20 | 99.40 | 99.60 | 99.60 |
| RULER 4K NIAH multivalue | — | — | 83.00 | 87.40 | 91.20 | 92.35 | 94.30 | 91.90 |
| RULER 4K NIAH multiquery | — | — | 99.40 | 99.70 | 99.80 | 99.80 | 100.00 | 100.00 |
| RULER 4K VT | — | — | 97.28 | 97.80 | 98.44 | 98.52 | 98.80 | 98.64 |
| RULER 4K CWE | — | — | 94.66 | 97.18 | 97.78 | 97.86 | 98.44 | 98.70 |
| RULER 4K FWE | — | — | 83.33 | 82.53 | 82.53 | 83.60 | 85.13 | 85.80 |
| RULER 4K QA 1 | — | — | 83.00 | 85.00 | 84.60 | 84.60 | 84.80 | 85.20 |
| RULER 4K QA 2 | — | — | 61.00 | 60.80 | 59.40 | 60.40 | 59.20 | 59.60 |
| **RULER 4K 13-task mean** | **—** | **—** | **91.87** | **92.88** | **93.21** | **93.49** | **93.71** | **93.62** |
| RULER 8K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 8K NIAH single 2 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 8K NIAH single 3 | — | — | 99.60 | 99.40 | 99.40 | 99.60 | 99.40 | 99.60 |
| RULER 8K NIAH multikey 1 | — | — | 96.40 | 98.40 | 99.40 | 99.40 | 99.60 | 99.60 |
| RULER 8K NIAH multikey 2 | — | — | 97.40 | 98.60 | 99.20 | 99.60 | 99.80 | 100.00 |
| RULER 8K NIAH multikey 3 | — | — | 88.40 | 94.80 | 95.60 | 96.20 | 96.80 | 97.20 |
| RULER 8K NIAH multivalue | — | — | 77.45 | 79.50 | 81.50 | 82.45 | 83.95 | 84.40 |
| RULER 8K NIAH multiquery | — | — | 99.80 | 99.95 | 99.90 | 99.90 | 99.95 | 99.90 |
| RULER 8K VT | — | — | 95.88 | 96.04 | 96.20 | 96.88 | 97.76 | 97.96 |
| RULER 8K CWE | — | — | 87.18 | 91.00 | 91.02 | 90.56 | 90.76 | 90.88 |
| RULER 8K FWE | — | — | 75.73 | 79.87 | 80.67 | 81.47 | 82.40 | 82.67 |
| RULER 8K QA 1 | — | — | 76.00 | 78.00 | 80.60 | 81.00 | 81.60 | 81.60 |
| RULER 8K QA 2 | — | — | 58.00 | 55.40 | 55.80 | 56.40 | 54.20 | 54.40 |
| **RULER 8K 13-task mean** | **—** | **—** | **88.60** | **90.07** | **90.71** | **91.04** | **91.25** | **91.40** |
| RULER 64K NIAH single 1 | — | — | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RULER 64K NIAH multikey 3 | — | — | 68.80 | 89.80 | 93.20 | 92.20 | 91.80 | 91.60 |
| RULER 64K NIAH multivalue | — | — | 73.65 | 88.95 | 84.90 | 85.00 | 82.20 | 82.20 |
| RULER 64K VT | — | — | 76.20 | 73.76 | 72.68 | 73.32 | 74.56 | 77.04 |
| RULER 64K CWE | — | — | 38.58 | 48.22 | 41.92 | 32.38 | 31.16 | 33.58 |
| RULER 64K FWE | — | — | 72.73 | 76.00 | 84.07 | 85.87 | 86.20 | 86.00 |
| RULER 64K QA 2 | — | — | 49.00 | 48.00 | 48.40 | 48.00 | 46.20 | 45.60 |
| **RULER 64K selected 7 mean** | **—** | **—** | **68.42** | **74.96** | **75.02** | **73.82** | **73.16** | **73.72** |
| **RULER selected 33 mean** | **—** | **—** | **85.61** | **87.97** | **88.37** | **88.35** | **88.38** | **88.52** |
| **Overall selected 60 mean** | **—** | **—** | **66.84** | **71.77** | **73.25** | **73.71** | **74.02** | **74.31** |

Verified source snapshot: 2026-09-10T10:30:14.582966+00:00. All 180 method/configuration curves are complete at the original six ratios (1,080 raw score values), with no conflicting values, plus 66 additional 5%/10% values for the 11 SCBench base tasks across 3 methods (1,146 raw score values in total). 5%/10% values for the 11 SCBench base tasks were added 2026-09-13T10:38:02.861839+00:00, sourced from the same three runs. See [the full-precision source snapshot](data/method-scores.json) for source identities and pinned dataset revisions.
