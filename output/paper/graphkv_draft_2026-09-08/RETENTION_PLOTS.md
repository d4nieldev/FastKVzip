# Retention-score plot index

All scores and task memberships are archived in [the plot snapshot](data/retention-plot-scores.json). The standalone atlas has PDF bookmarks for each section and page. Each panel uses raw scores, three solid method curves and measured full-cache dashed references. Each panel has a zoomed linear y-axis; ranges vary by panel, and ticks show actual scores. All scores and baselines remain visible. Identical constant curves are labeled. The tables and relative context-length Figure 2 are unchanged.

17 of 98 panels (marked † below) additionally plot 5% and 10% retention: the 11 SCBench base tasks, their aggregate mean, and their four family panels. Every other panel is unaffected and still spans 20-100%.

Official grouping sources: [SCBench](https://arxiv.org/html/2412.10319v2#S3) and [RULER](https://arxiv.org/html/2404.06654v3#S3). Category mappings and coverage notes are in [the grouping snapshot](data/retention-plot-groups.json).

## Benchmark summaries

### Atlas page 1: [Benchmark summaries](figures/retention-curves/01-suite-overview.pdf)

| Panel | Configuration(s) |
|---|---|
| Combined evaluated suite | `scbench_kv`, `scbench_kv_mid`, `scbench_kv_short`, `scbench_kv_tiny`, `scbench_mf`, `scbench_mf_mid`, `scbench_mf_short`, `scbench_mf_tiny`, `scbench_qa_eng`, `scbench_repoqa`, `scbench_repoqa_short`, `scbench_repoqa_tiny`, `scbench_summary`, `scbench_summary_mid`, `scbench_summary_short`, `scbench_summary_tiny`, `scbench_choice_eng`, `scbench_many_shot`, `scbench_many_shot_short`, `scbench_many_shot_tiny`, `scbench_prefix_suffix`, `scbench_prefix_suffix_mid`, `scbench_prefix_suffix_short`, `scbench_prefix_suffix_tiny`, `scbench_vt`, `scbench_summary_with_needles`, `scbench_repoqa_and_kv`, `ruler_niah_single_1_4k`, `ruler_niah_single_2_4k`, `ruler_niah_single_3_4k`, `ruler_niah_multikey_1_4k`, `ruler_niah_multikey_2_4k`, `ruler_niah_multikey_3_4k`, `ruler_niah_multivalue_4k`, `ruler_niah_multiquery_4k`, `ruler_vt_4k`, `ruler_cwe_4k`, `ruler_fwe_4k`, `ruler_qa_1_4k`, `ruler_qa_2_4k`, `ruler_niah_single_1_8k`, `ruler_niah_single_2_8k`, `ruler_niah_single_3_8k`, `ruler_niah_multikey_1_8k`, `ruler_niah_multikey_2_8k`, `ruler_niah_multikey_3_8k`, `ruler_niah_multivalue_8k`, `ruler_niah_multiquery_8k`, `ruler_vt_8k`, `ruler_cwe_8k`, `ruler_fwe_8k`, `ruler_qa_1_8k`, `ruler_qa_2_8k`, `ruler_niah_single_1_64k`, `ruler_niah_multikey_3_64k`, `ruler_niah_multivalue_64k`, `ruler_vt_64k`, `ruler_cwe_64k`, `ruler_fwe_64k`, `ruler_qa_2_64k` |
| SCBench: base tasks † | `scbench_kv`, `scbench_mf`, `scbench_qa_eng`, `scbench_repoqa`, `scbench_summary`, `scbench_choice_eng`, `scbench_many_shot`, `scbench_prefix_suffix`, `scbench_vt`, `scbench_summary_with_needles`, `scbench_repoqa_and_kv` |
| SCBench: all variants | `scbench_kv`, `scbench_kv_mid`, `scbench_kv_short`, `scbench_kv_tiny`, `scbench_mf`, `scbench_mf_mid`, `scbench_mf_short`, `scbench_mf_tiny`, `scbench_qa_eng`, `scbench_repoqa`, `scbench_repoqa_short`, `scbench_repoqa_tiny`, `scbench_summary`, `scbench_summary_mid`, `scbench_summary_short`, `scbench_summary_tiny`, `scbench_choice_eng`, `scbench_many_shot`, `scbench_many_shot_short`, `scbench_many_shot_tiny`, `scbench_prefix_suffix`, `scbench_prefix_suffix_mid`, `scbench_prefix_suffix_short`, `scbench_prefix_suffix_tiny`, `scbench_vt`, `scbench_summary_with_needles`, `scbench_repoqa_and_kv` |
| RULER: evaluated lengths | `ruler_niah_single_1_4k`, `ruler_niah_single_2_4k`, `ruler_niah_single_3_4k`, `ruler_niah_multikey_1_4k`, `ruler_niah_multikey_2_4k`, `ruler_niah_multikey_3_4k`, `ruler_niah_multivalue_4k`, `ruler_niah_multiquery_4k`, `ruler_vt_4k`, `ruler_cwe_4k`, `ruler_fwe_4k`, `ruler_qa_1_4k`, `ruler_qa_2_4k`, `ruler_niah_single_1_8k`, `ruler_niah_single_2_8k`, `ruler_niah_single_3_8k`, `ruler_niah_multikey_1_8k`, `ruler_niah_multikey_2_8k`, `ruler_niah_multikey_3_8k`, `ruler_niah_multivalue_8k`, `ruler_niah_multiquery_8k`, `ruler_vt_8k`, `ruler_cwe_8k`, `ruler_fwe_8k`, `ruler_qa_1_8k`, `ruler_qa_2_8k`, `ruler_niah_single_1_64k`, `ruler_niah_multikey_3_64k`, `ruler_niah_multivalue_64k`, `ruler_vt_64k`, `ruler_cwe_64k`, `ruler_fwe_64k`, `ruler_qa_2_64k` |

Unweighted configuration means matching the appendix tables. These are summaries of the evaluated suite, not official complete-benchmark results.

### Atlas page 2: [RULER summaries by context length](figures/retention-curves/02-ruler-lengths.pdf)

| Panel | Configuration(s) |
|---|---|
| RULER 4K | `ruler_niah_single_1_4k`, `ruler_niah_single_2_4k`, `ruler_niah_single_3_4k`, `ruler_niah_multikey_1_4k`, `ruler_niah_multikey_2_4k`, `ruler_niah_multikey_3_4k`, `ruler_niah_multivalue_4k`, `ruler_niah_multiquery_4k`, `ruler_vt_4k`, `ruler_cwe_4k`, `ruler_fwe_4k`, `ruler_qa_1_4k`, `ruler_qa_2_4k` |
| RULER 8K | `ruler_niah_single_1_8k`, `ruler_niah_single_2_8k`, `ruler_niah_single_3_8k`, `ruler_niah_multikey_1_8k`, `ruler_niah_multikey_2_8k`, `ruler_niah_multikey_3_8k`, `ruler_niah_multivalue_8k`, `ruler_niah_multiquery_8k`, `ruler_vt_8k`, `ruler_cwe_8k`, `ruler_fwe_8k`, `ruler_qa_1_8k`, `ruler_qa_2_8k` |
| RULER 64K | `ruler_niah_single_1_64k`, `ruler_niah_multikey_3_64k`, `ruler_niah_multivalue_64k`, `ruler_vt_64k`, `ruler_cwe_64k`, `ruler_fwe_64k`, `ruler_qa_2_64k` |

All 13 tasks are included at 4K and 8K; 64K contains seven selected tasks. Differences across lengths also reflect this unequal task coverage.

## SCBench task families

### Atlas page 3: [SCBench task families: base tasks](figures/retention-curves/03-scbench-families-base-tasks.pdf)

| Panel | Configuration(s) |
|---|---|
| String retrieval † | `scbench_kv`, `scbench_prefix_suffix`, `scbench_vt` |
| Semantic retrieval † | `scbench_repoqa`, `scbench_qa_eng`, `scbench_choice_eng` |
| Global information processing † | `scbench_mf`, `scbench_many_shot`, `scbench_summary` |
| Multi-tasking † | `scbench_summary_with_needles`, `scbench_repoqa_and_kv` |

Categories follow SCBench Section 3.1 and Table 2. Semantic retrieval excludes the unevaluated Chinese-QA task. Only the 11 evaluated base tasks enter these means; prepared length variants are excluded.

### Atlas page 4: [SCBench task families: all variants](figures/retention-curves/04-scbench-families-all-variants.pdf)

| Panel | Configuration(s) |
|---|---|
| String retrieval | `scbench_kv`, `scbench_kv_mid`, `scbench_kv_short`, `scbench_kv_tiny`, `scbench_prefix_suffix`, `scbench_prefix_suffix_mid`, `scbench_prefix_suffix_short`, `scbench_prefix_suffix_tiny`, `scbench_vt` |
| Semantic retrieval | `scbench_repoqa`, `scbench_repoqa_short`, `scbench_repoqa_tiny`, `scbench_qa_eng`, `scbench_choice_eng` |
| Global information processing | `scbench_mf`, `scbench_mf_mid`, `scbench_mf_short`, `scbench_mf_tiny`, `scbench_many_shot`, `scbench_many_shot_short`, `scbench_many_shot_tiny`, `scbench_summary`, `scbench_summary_mid`, `scbench_summary_short`, `scbench_summary_tiny` |
| Multi-tasking † | `scbench_summary_with_needles`, `scbench_repoqa_and_kv` |

Categories follow SCBench Section 3.1 and Table 2. Semantic retrieval excludes the unevaluated Chinese-QA task. Prepared length variants inherit their base task's category and receive equal weight.

## RULER task families

### Atlas page 5: [RULER 4K: task families](figures/retention-curves/05-ruler-4k.pdf)

| Panel | Configuration(s) |
|---|---|
| Retrieval | `ruler_niah_single_1_4k`, `ruler_niah_single_2_4k`, `ruler_niah_single_3_4k`, `ruler_niah_multikey_1_4k`, `ruler_niah_multikey_2_4k`, `ruler_niah_multikey_3_4k`, `ruler_niah_multivalue_4k`, `ruler_niah_multiquery_4k` |
| Multi-hop tracing | `ruler_vt_4k` |
| Aggregation | `ruler_cwe_4k`, `ruler_fwe_4k` |
| Question answering | `ruler_qa_1_4k`, `ruler_qa_2_4k` |

Categories follow RULER Section 3. A one-configuration family reproduces its individual task curve.

### Atlas page 6: [RULER 8K: task families](figures/retention-curves/06-ruler-8k.pdf)

| Panel | Configuration(s) |
|---|---|
| Retrieval | `ruler_niah_single_1_8k`, `ruler_niah_single_2_8k`, `ruler_niah_single_3_8k`, `ruler_niah_multikey_1_8k`, `ruler_niah_multikey_2_8k`, `ruler_niah_multikey_3_8k`, `ruler_niah_multivalue_8k`, `ruler_niah_multiquery_8k` |
| Multi-hop tracing | `ruler_vt_8k` |
| Aggregation | `ruler_cwe_8k`, `ruler_fwe_8k` |
| Question answering | `ruler_qa_1_8k`, `ruler_qa_2_8k` |

Categories follow RULER Section 3. A one-configuration family reproduces its individual task curve.

### Atlas page 7: [RULER 64K: task families](figures/retention-curves/07-ruler-64k.pdf)

| Panel | Configuration(s) |
|---|---|
| Retrieval | `ruler_niah_single_1_64k`, `ruler_niah_multikey_3_64k`, `ruler_niah_multivalue_64k` |
| Multi-hop tracing | `ruler_vt_64k` |
| Aggregation | `ruler_cwe_64k`, `ruler_fwe_64k` |
| Question answering | `ruler_qa_2_64k` |

Categories follow RULER Section 3. Only the seven evaluated 64K tasks are included. A one-configuration family reproduces its individual task curve.

## RULER retrieval subtypes

### Atlas page 8: [RULER 4K: retrieval subtypes](figures/retention-curves/08-ruler-retrieval-subtypes-4k.pdf)

| Panel | Configuration(s) |
|---|---|
| Single NIAH | `ruler_niah_single_1_4k`, `ruler_niah_single_2_4k`, `ruler_niah_single_3_4k` |
| Multi-keys NIAH | `ruler_niah_multikey_1_4k`, `ruler_niah_multikey_2_4k`, `ruler_niah_multikey_3_4k` |
| Multi-values NIAH | `ruler_niah_multivalue_4k` |
| Multi-queries NIAH | `ruler_niah_multiquery_4k` |

Categories follow RULER Section 3. Single and multi-key means combine their available variants. A one-configuration family reproduces its individual task curve.

### Atlas page 9: [RULER 8K: retrieval subtypes](figures/retention-curves/09-ruler-retrieval-subtypes-8k.pdf)

| Panel | Configuration(s) |
|---|---|
| Single NIAH | `ruler_niah_single_1_8k`, `ruler_niah_single_2_8k`, `ruler_niah_single_3_8k` |
| Multi-keys NIAH | `ruler_niah_multikey_1_8k`, `ruler_niah_multikey_2_8k`, `ruler_niah_multikey_3_8k` |
| Multi-values NIAH | `ruler_niah_multivalue_8k` |
| Multi-queries NIAH | `ruler_niah_multiquery_8k` |

Categories follow RULER Section 3. Single and multi-key means combine their available variants. A one-configuration family reproduces its individual task curve.

### Atlas page 10: [RULER 64K: retrieval subtypes](figures/retention-curves/10-ruler-retrieval-subtypes-64k.pdf)

| Panel | Configuration(s) |
|---|---|
| Single NIAH | `ruler_niah_single_1_64k` |
| Multi-keys NIAH | `ruler_niah_multikey_3_64k` |
| Multi-values NIAH | `ruler_niah_multivalue_64k` |

Categories follow RULER Section 3. Single and multi-key means combine their available variants. Multi-query retrieval is unevaluated at 64K and has no curve. Only the seven evaluated 64K tasks are included. A one-configuration family reproduces its individual task curve.

## SCBench individual tasks

### Atlas page 11: [SCBench individual tasks (1/5)](figures/retention-curves/11-scbench-tasks-1.pdf)

| Panel | Configuration(s) |
|---|---|
| KV (base) † | `scbench_kv` |
| KV (mid) | `scbench_kv_mid` |
| KV (short) | `scbench_kv_short` |
| KV (tiny) | `scbench_kv_tiny` |
| MF (base) † | `scbench_mf` |
| MF (mid) | `scbench_mf_mid` |

Each panel is one evaluated configuration. Base, mid, short and tiny name the prepared variants used in the tables; mixed tasks retain their deployed combined score.

### Atlas page 12: [SCBench individual tasks (2/5)](figures/retention-curves/12-scbench-tasks-2.pdf)

| Panel | Configuration(s) |
|---|---|
| MF (short) | `scbench_mf_short` |
| MF (tiny) | `scbench_mf_tiny` |
| QA (English) (base) † | `scbench_qa_eng` |
| RepoQA (base) † | `scbench_repoqa` |
| RepoQA (short) | `scbench_repoqa_short` |
| RepoQA (tiny) | `scbench_repoqa_tiny` |

Each panel is one evaluated configuration. Base, mid, short and tiny name the prepared variants used in the tables; mixed tasks retain their deployed combined score.

### Atlas page 13: [SCBench individual tasks (3/5)](figures/retention-curves/13-scbench-tasks-3.pdf)

| Panel | Configuration(s) |
|---|---|
| Summary (base) † | `scbench_summary` |
| Summary (mid) | `scbench_summary_mid` |
| Summary (short) | `scbench_summary_short` |
| Summary (tiny) | `scbench_summary_tiny` |
| Choice (English) (base) † | `scbench_choice_eng` |

Each panel is one evaluated configuration. Base, mid, short and tiny name the prepared variants used in the tables; mixed tasks retain their deployed combined score.

### Atlas page 14: [SCBench individual tasks (4/5)](figures/retention-curves/14-scbench-tasks-4.pdf)

| Panel | Configuration(s) |
|---|---|
| Many-shot (base) † | `scbench_many_shot` |
| Many-shot (short) | `scbench_many_shot_short` |
| Many-shot (tiny) | `scbench_many_shot_tiny` |
| Prefix/suffix (base) † | `scbench_prefix_suffix` |
| Prefix/suffix (mid) | `scbench_prefix_suffix_mid` |

Each panel is one evaluated configuration. Base, mid, short and tiny name the prepared variants used in the tables; mixed tasks retain their deployed combined score.

### Atlas page 15: [SCBench individual tasks (5/5)](figures/retention-curves/15-scbench-tasks-5.pdf)

| Panel | Configuration(s) |
|---|---|
| Prefix/suffix (short) | `scbench_prefix_suffix_short` |
| Prefix/suffix (tiny) | `scbench_prefix_suffix_tiny` |
| VT (base) † | `scbench_vt` |
| Summary + needles (base) † | `scbench_summary_with_needles` |
| RepoQA + KV (base) † | `scbench_repoqa_and_kv` |

Each panel is one evaluated configuration. Base, mid, short and tiny name the prepared variants used in the tables; mixed tasks retain their deployed combined score.

## RULER 4K individual tasks

### Atlas page 16: [RULER 4K individual tasks (1/3)](figures/retention-curves/16-ruler-4k-tasks-1.pdf)

| Panel | Configuration(s) |
|---|---|
| Single NIAH 1 | `ruler_niah_single_1_4k` |
| Single NIAH 2 | `ruler_niah_single_2_4k` |
| Single NIAH 3 | `ruler_niah_single_3_4k` |
| Multi-keys NIAH 1 | `ruler_niah_multikey_1_4k` |
| Multi-keys NIAH 2 | `ruler_niah_multikey_2_4k` |

Each panel is one evaluated 4K configuration. All 13 RULER task configurations are included across this section.

### Atlas page 17: [RULER 4K individual tasks (2/3)](figures/retention-curves/17-ruler-4k-tasks-2.pdf)

| Panel | Configuration(s) |
|---|---|
| Multi-keys NIAH 3 | `ruler_niah_multikey_3_4k` |
| Multi-values NIAH | `ruler_niah_multivalue_4k` |
| Multi-queries NIAH | `ruler_niah_multiquery_4k` |
| Variable tracking | `ruler_vt_4k` |

Each panel is one evaluated 4K configuration. All 13 RULER task configurations are included across this section.

### Atlas page 18: [RULER 4K individual tasks (3/3)](figures/retention-curves/18-ruler-4k-tasks-3.pdf)

| Panel | Configuration(s) |
|---|---|
| Common-word extraction | `ruler_cwe_4k` |
| Frequent-word extraction | `ruler_fwe_4k` |
| QA 1 (SQuAD) | `ruler_qa_1_4k` |
| QA 2 (HotpotQA) | `ruler_qa_2_4k` |

Each panel is one evaluated 4K configuration. All 13 RULER task configurations are included across this section.

## RULER 8K individual tasks

### Atlas page 19: [RULER 8K individual tasks (1/3)](figures/retention-curves/19-ruler-8k-tasks-1.pdf)

| Panel | Configuration(s) |
|---|---|
| Single NIAH 1 | `ruler_niah_single_1_8k` |
| Single NIAH 2 | `ruler_niah_single_2_8k` |
| Single NIAH 3 | `ruler_niah_single_3_8k` |
| Multi-keys NIAH 1 | `ruler_niah_multikey_1_8k` |
| Multi-keys NIAH 2 | `ruler_niah_multikey_2_8k` |

Each panel is one evaluated 8K configuration. All 13 RULER task configurations are included across this section.

### Atlas page 20: [RULER 8K individual tasks (2/3)](figures/retention-curves/20-ruler-8k-tasks-2.pdf)

| Panel | Configuration(s) |
|---|---|
| Multi-keys NIAH 3 | `ruler_niah_multikey_3_8k` |
| Multi-values NIAH | `ruler_niah_multivalue_8k` |
| Multi-queries NIAH | `ruler_niah_multiquery_8k` |
| Variable tracking | `ruler_vt_8k` |

Each panel is one evaluated 8K configuration. All 13 RULER task configurations are included across this section.

### Atlas page 21: [RULER 8K individual tasks (3/3)](figures/retention-curves/21-ruler-8k-tasks-3.pdf)

| Panel | Configuration(s) |
|---|---|
| Common-word extraction | `ruler_cwe_8k` |
| Frequent-word extraction | `ruler_fwe_8k` |
| QA 1 (SQuAD) | `ruler_qa_1_8k` |
| QA 2 (HotpotQA) | `ruler_qa_2_8k` |

Each panel is one evaluated 8K configuration. All 13 RULER task configurations are included across this section.

## RULER 64K individual tasks

### Atlas page 22: [RULER 64K individual tasks (1/2)](figures/retention-curves/22-ruler-64k-tasks-1.pdf)

| Panel | Configuration(s) |
|---|---|
| Single NIAH 1 | `ruler_niah_single_1_64k` |
| Multi-keys NIAH 3 | `ruler_niah_multikey_3_64k` |
| Multi-values NIAH | `ruler_niah_multivalue_64k` |
| Variable tracking | `ruler_vt_64k` |

Each panel is one evaluated 64K configuration. The seven selected 64K tasks are the complete matched subset; the other six have no curves.

### Atlas page 23: [RULER 64K individual tasks (2/2)](figures/retention-curves/23-ruler-64k-tasks-2.pdf)

| Panel | Configuration(s) |
|---|---|
| Common-word extraction | `ruler_cwe_64k` |
| Frequent-word extraction | `ruler_fwe_64k` |
| QA 2 (HotpotQA) | `ruler_qa_2_64k` |

Each panel is one evaluated 64K configuration. The seven selected 64K tasks are the complete matched subset; the other six have no curves.
