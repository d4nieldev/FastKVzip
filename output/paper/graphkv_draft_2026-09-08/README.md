# Paper review copy

This standalone copy was prepared from `~/Downloads/graphkv_initial_draft_skeleton.zip`. It has no connection to an Overleaf project. The original archive was not modified.

## Contents

- `iclr2027_conference.tex`: main document in the supplied ICLR 2027 template.
- `sections/introduction.tex`: agent-context motivation and approach; the final results/contributions paragraph is deliberately reserved.
- `sections/related_work.tex`: attention projections and KV caching, soft compression, prompt/state pruning, KV eviction, and graph-based selection.
- `sections/method.tex`: the Fast KVzip gate equation, graph construction, message passing, and two-stage training, with brief pointers to the surrogate, factorized computation, and score-gradient replay.
- `sections/appendix_method.tex`: normalization, learning through discrete cache selection and its explicit surrogate gradient, factorized computation, replay correctness, and memory accounting.
- `sections/appendix_results.tex`: three complete, two-page method tables, including all 60 evaluated configurations and explicitly scoped macro-averages at each retention ratio.
- `sections/appendix_benchmarks.tex`: task descriptions, approximate lengths, and reported/unreported coverage for all 107 configured benchmark variants.
- `sections/appendix_retention_curves.tex`: indexed retention-score appendix, with 98 panels on 23 plot pages: all 60 individual configurations plus 38 benchmark, family, and retrieval-subtype panels.
- `RETENTION_PLOTS.md`: plot-page index with links to vector PDFs and exact configuration membership for every panel.
- `plot_retention_scores.py`: regenerates the curve appendix, vector plot pages, bookmarked standalone atlas, index, and full-precision plot snapshot; `--check` validates data, coverage, category partitions, and agreement with table aggregates.
- `data/retention-plot-groups.json`: official SCBench and RULER task-family mappings, paper references, and coverage limitations.
- `data/retention-plot-scores.json`: all plotted curves, method-specific full-cache references, configuration memberships, and source hashes.
- `figures/retention-curves/`: 23 numbered vector plot pages, ordered as in the appendix and standalone atlas (`../../pdf/retention-score-atlas.pdf`).
- `results_tables.md`: the same three tables in Markdown for convenient review.
- `render_result_tables.py`: renders both table formats from the verified score snapshot, emitting an explicit patch; assertions check task/ratio coverage and sample counts.
- `merge_extreme_scores.py`: merges the 5%/10% SCBench base-task scores into `data/method-scores.json` from the same three W&B runs; `--check` validates an already-merged file without writing.
- `data/method-scores.json`: full-precision score snapshot, source run identities, configuration, and extraction time used for the tables; the 11 SCBench base tasks additionally carry 5% and 10% under `extreme_ratio_extension`.
- `data/benchmark-context-lengths.json`: full-precision context statistics and published RULER input-length proxies, with source revisions and measurement provenance.
- `figures/context-score-buckets.pdf`: full-cache-relative scores by measured token-length bucket, with separate panels for each retention ratio and full cache.
- `data/context-analysis/context-scores.json`: matched per-context scores, exact lengths, bucket membership/composition, scorer/source hashes, and reconciliation against all 1,080 reported task scores.
- `data/context-analysis/context-relative-scores.json`: derived relative curves, each configuration/bucket/method's full-cache denominator, context counts, and the source snapshot hash.
- `plot_context_scores.py`: reproduces the relative plot from the raw snapshot; `--recompute` reuses the original scorer and locally retrieved outputs/logs, while `--check` checks boundaries, raw/relative weighting, method-specific and bucket-specific baselines, and zero-denominator handling.
- `iclr2027_conference.bib`: verified bibliography; uncited optional entries do not appear in the PDF.
- `figures/mixkv-overview.png`: supplied Figure 1, showing context-only cache selection and two-stage selector training.
- `FIGURE_BRIEFS.md`: original design prompts for the architecture/training overview and optional replay/motivation figures.
- `SOURCE_NOTES.md`: literature provenance and distinctions useful for later revisions. These notes are not part of the manuscript.

The Main Results subsection now describes the completed three-method comparison and references its detailed appendix tables. The other experimental placeholders and all AI-use, ethics, reproducibility, contribution, and acknowledgment text remain draft material; their presence in the PDF does not mean those sections have been completed. The author block and supplied style files are unchanged; the PDF uses the anonymous submission mode.

## Result tables

The comparison uses the GraphKV experiment `c0s997un` (our graph-based selector, window 0.02), official Fast KVzip `vi31uf3h` (native chunked eviction, window 0.02), and KVzip `p2wsdvv5` (native reconstruction scoring, no protected window). All runs are in the existing `graphkv-answer-qwen25-7b1m-s40n40-grid-v1` W&B project. The JSON snapshot records the full source URLs; these identifying links are deliberately not printed in the anonymous manuscript.

Every method table uses the same 60 configurations: 13 RULER tasks at 4K, 13 at 8K, seven selected tasks at 64K, and all 27 prepared SCBench configurations (11 base tasks and 16 length variants). Columns are 20%, 30%, 40%, 50%, 75%, and full-cache retention. Scores are absolute benchmark scores on a 0–100 scale, not relative-to-full scores. Each method retains its separately measured full-cache column. The original 38 configurations' numeric values are unchanged.

The seven aggregate rows are arithmetic means of the unrounded configuration scores, independently for each column: SCBench 11-base-task mean, SCBench 27-configuration mean, RULER 4K, RULER 8K, RULER selected-64K, all 33 evaluated RULER configurations, and all 60 configurations. The 11-base-task mean excludes length variants; the 27-configuration mean gives each variant equal weight. The RULER total weights the three lengths by 13/33, 13/33, and 7/33; it is not an equal mean of length averages. These are descriptive prepared-split summaries, not official complete-benchmark scores or pooled-question accuracy. No average over retention ratios is reported. Display values are rounded to two decimal places only after aggregation.

Mixed SCBench tasks retain the existing combined configuration score. Summary + needles also retains the deployed 48-token output limit for both summary and retrieval questions. These settings are unchanged across the three methods, rather than retroactively replaced with a different benchmark protocol. The three logical tables use standard `longtable`, with repeated headers and a page break between SCBench and RULER.

Each table now additionally reports 5% and 10% retention, sourced from later history on the same three W&B runs (`merge_extreme_scores.py`, provenance recorded under each method's `config.evaluation_extensions.scbench-extreme-ratios` in the JSON snapshot). These two columns cover the 11 SCBench base-task rows and the SCBench 11-base-task mean only; the 16 length variants, all 33 RULER configurations, the 27-configuration mean, RULER's four aggregate rows, and the overall 60-configuration mean show `—` in both columns, since they were not evaluated at these ratios. `RATIOS` in `render_result_tables.py` (the six original columns, still required for all 60 configurations) is unchanged; `DISPLAY_RATIOS` is the full eight-column display order. `render_result_tables.py` regenerates both `sections/appendix_results.tex` and `results_tables.md` from the merged snapshot; re-running it after the merge reproduces the current files exactly.

## Benchmark guide

Appendix C covers every configuration in the existing 107-item evaluation inventory. Of these, 60 appear in the matched three-method report and 47 remain outside it: 45 RULER, SQuAD, and filtered GSM8K. This does not assert that an excluded configuration has never been run in a previous GraphKV-only experiment, and it does not schedule any new evaluation.

The guide is a single appendix section with two task-description tables. RULER uses the standard 4K through 128K labels; its separate average-length table has been removed. SCBench's description table includes one average-token column, listing all 27 reported variants rounded to the nearest thousand tokens (K). These are raw-context Qwen token counts excluding system/query/chat formatting. Full-precision measurements remain archived in the JSON snapshot, not displayed as an additional table. SQuAD's current adapter uses the training split; GSM is the filtered 148-example test subset, not the full official test set.

The short context-length interpretation uses the same seven RULER tasks at all three evaluated lengths, avoiding the unequal 13/13/7 task mix. Its conclusion is deliberately limited: relative performance improves against KVzip in some longer-context comparisons, but the trend is not uniform across retention ratios or against FastKVzip. This RULER comparison is unchanged by the SCBench extension.

The context-length plot pools every context example from all 60 reported configurations after normalizing for task difficulty. For each configuration, length bucket, and method, divide the mean compressed score by that method's mean full-cache score on the same contexts. Average these ratios across configurations in the bucket using their context counts as weights, then multiply by 100. This is equivalent to giving each context equal weight after dividing its score by its configuration/bucket/method's full-cache mean. It avoids individual-example division, since some examples have a zero full-cache score, and avoids normalizing a pooled raw mean, which implicitly upweights tasks with higher full-cache scores. All actual configuration/bucket denominators are positive (minimum 20.139 on the raw 0-100 scale); the plotting script raises an error for a nonpositive denominator rather than dropping data or adding an epsilon. Full cache is 100% by construction, and scores above 100% remain visible.

Each context's raw score uses the native evaluator. In particular, RepoQA + KV averages its KV score and RepoQA pass@1 rather than weighting all questions equally. This is not a benchmark-macro average or question-weighted accuracy, and retention ratios are never averaged together. All three methods use the identical 18,531 contexts (28,236 questions). Raw scores and table values are preserved. The number and mix of configurations in each bucket are saved in both snapshots; normalization adjusts the baseline score scale, but varying task composition and small configuration/bucket cells mean the curves remain descriptive rather than a controlled context-length ablation.

For this plot, lengths are the exact raw-context token counts printed by the baseline prefill code and joined to GraphKV by the same dataset revision, example index, and references. All 18,531 lengths are available from FastKVzip; all 18,521 available KVzip counts match exactly. KVzip's resumed Summary log omits indices 0–9, whose lengths therefore have one log source rather than two; their evaluation outputs are complete. All 1,719 newly added context lengths have both baseline log sources. Nominal RULER labels and dataset-average lengths are not used to assign examples. Buckets are `[0,4096)`, `[4096,8192)`, and so on through `[131072,262144)`; thus K means 1,024 here. This does not change the rounded decimal-thousand SCBench averages in the task guide.

Plotting dependencies are isolated under the ruler worktree's `.slurm/plot-venv`, not added to the model runtime. The existing scorer is reused with NumPy (1.26.4), Rouge (1.0.1), NLTK (3.10.3), tree-sitter (0.21.3), tree-sitter-languages (1.10.2), PyArrow (25.0.1), and Matplotlib (3.11.1). The original answer parser reads eight cached metadata parquets at SCBench revision `a079be919d3131822c202180ffd4dc322968de45`: the three RepoQA variants, RepoQA + KV, the three Many-shot variants, and Summary + needles. Native reference matching and mixed-task aggregation are preserved. All 1,080 task/ratio scores reconcile within 3e-13; the original 16,812 per-context score rows are unchanged. Updating the paper performs no new evaluations or W&B writes.

## Retention-score plot appendix

Appendix D supplements the tables and relative context-length Figure 2 with absolute score versus retention curves. The 23 plot pages are ordered as benchmark summaries (2), SCBench task families (2), RULER task families by length (3), RULER retrieval subtypes by length (3), SCBench individual configurations (5), RULER 4K configurations (3), RULER 8K configurations (3), and selected RULER 64K configurations (2). A plot guide in the manuscript, PDF bookmarks in the standalone atlas, numbered filenames, and `RETENTION_PLOTS.md` make the collection navigable. Each page has at most six panels, one method legend, common retention coordinates, and a clear note that y-axis ranges vary by panel.

Every panel plots the six measured retention ratios (20%, 30%, 40%, 50%, 75%, 100%) at their actual numeric spacing. The y-axis shows the raw native-evaluator score using a separate zoomed linear range for each panel. Limits enclose every method score and full-cache baseline, with 8% data padding, a minimum two-point window, simple ticks within 0-100, and 5% frame padding beyond the outer ticks so markers remain visible. Tick labels show actual scores without numeric offsets or scientific notation. Constant coincident curves are labeled explicitly. The axis limits and ticks are archived per panel in the plot snapshot. Three solid curves use the same method colors as Figure 2. Horizontal dashed references retain each method's own full-cache score; when all three values coincide, one gray dashed line represents the shared baseline. Otherwise, colored dash segments distinguish even closely spaced baselines. Coincident curves can overlap. Of the 60 individual configurations, 36 share exactly equal full-cache values and 24 differ; no shared reference is fabricated by averaging different baselines.

The seven benchmark aggregate panels exactly reproduce the tables' unrounded configuration means. SCBench's 11-base-task and 27-configuration summaries are separate. RULER 4K and 8K each contain 13 tasks; 64K contains seven selected tasks; the combined RULER panel includes all 33 configurations. The combined suite includes all 60, with equal configuration weighting, not equal benchmark weighting or pooled-context weighting. Retention ratios are never averaged together. These are summaries of the matched prepared suite rather than complete official benchmark scores.

The official [SCBench categories](https://arxiv.org/html/2412.10319v2#S3) are string retrieval, semantic retrieval, global information processing, and multi-tasking. Category summaries are provided both for base tasks and for all prepared variants. Chinese QA, one official semantic-retrieval task, is absent from the evaluated suite. The official [RULER categories and retrieval subtypes](https://arxiv.org/html/2404.06654v3#S3) are plotted separately at 4K, 8K, and 64K, with evaluated/official counts in each panel. Unevaluated 64K multi-query retrieval has no curve. Single-task family panels deliberately repeat their task curve to preserve the comparison grid. The snapshot has no separate Many-shot or mixed-task component scores, so these remain the same combined task scores as the tables.

17 of the 98 panels additionally plot 5% and 10% retention (marked † in `RETENTION_PLOTS.md`): the 11 individual SCBench base-task panels, their aggregate mean, and their four family panels — including the "all variants" Multi-tasking panel, which happens to have identical membership to its base-task counterpart since neither of its two tasks has a prepared length variant. `plot_retention_scores.py` decides this per panel from data (every task in the panel must be one of `suite.extreme_ratio_tasks` in the snapshot), not from a hardcoded list, so it stays correct if the underlying data ever changes. Every other panel is unaffected: same 6 points, same values, same rendering. `x` in `draw_page` and the y-axis zoom are computed per panel rather than once per page for this reason.

Regenerate from the repository root with:

```sh
.worktrees/ruler-evaluation/.slurm/plot-venv/bin/python output/paper/graphkv_draft_2026-09-08/plot_retention_scores.py
```

The standalone atlas uses pypdf 6.18.1 in the existing plotting environment to combine vector pages and add PDF bookmarks. Figure 2 (the relative context-length plot) is unchanged and out of scope for the 5%/10% extension above: it pools raw per-context scores, which are not available for the two new ratios (only W&B's per-task aggregates are); extending it would need re-parsing raw per-context result files from the cluster. The 81 of 98 panels and 55 of 67 table rows unaffected by the extension are content-identical after regeneration (verified by stripping PDF creation-timestamp metadata before comparing); only the 17 extended panels, one duplicated by the multi-tasking family having no length variants, and the 12 corresponding table rows (11 base tasks plus their aggregate mean) changed.

## Working choices

The descriptive title “Graph-Based KV Cache Compression with Answer Supervision” avoids a collision with Li et al.'s existing EMNLP 2025 GraphKV paper. The manuscript explicitly cites and distinguishes that work. The abstract has only the minimal naming substitutions needed for this descriptive title; its substantive content is preserved.

The method describes the core graph mixer and the deployed blockwise answer-gradient replay, rather than the optional GraNoLa normalization variant. GIN is cited for its aggregation design, without claiming that this weighted graph mixer is the standard GIN architecture or inherits its expressivity guarantee. Retention is described as query-agnostic context-cache selection, with question-dependent supervision during training.

Context length is denoted by `T` throughout; a graph block has `T_b = |b|` tokens. Answer-position indices and reconstruction targets retain their distinct meanings. BatchNorm is named directly in the mixer, with current-block statistics at both training and inference.

## Page budget and formatting

The [ICLR 2027 author guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines), checked on 8 September 2026, set a nine-page main-text limit for the initial submission. References and appendices are excluded; the required AI-use and optional ethics/reproducibility statements are also excluded. The guide allows ten main-text pages later in the discussion/camera-ready stages; this draft budgets against the initial nine-page limit.

The abstract, introduction, related work, method, and architecture/training overview occupy the first five pages and the opening lines of page six. Detailed discrete-selection, factorization, and replay material is consolidated in Appendix A.2--A.3. Figure 1 appears on page five. The experimental section follows, with the completed comparison protocol in Main Results and the remaining experimental placeholders preserved. The three full result tables are in the appendix rather than consuming the main-text budget. The reserved introduction paragraph, final experimental discussion, and any additional figures still need to fit within the main-text limit.

No margin, font-size, line-spacing, or template-file changes were used to meet the budget. Hyperlink borders are hidden for readability, and the descriptive title has a manual line break to avoid splitting a word. The appendix begins on a new page.

## Compile

From this directory, with a current TeX Live installation:

```sh
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The review PDF was built with the detected TeX Live 2026 runtime and visually checked after rendering with Poppler. The final packaged source includes the generated `.bbl` as a convenience; the `.bib` remains the editable reference source.
