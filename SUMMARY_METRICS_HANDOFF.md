**Summarization evaluation: handoff for the next session**

Written 12 September 2026. This document is intended to restore context without rereading the long conversation. The user is conserving their weekly token quota. Work efficiently, reuse completed analysis, and give concise updates.

**Prompt to give the assistant after the quota resets**

> Read `SUMMARY_METRICS_HANDOFF.md` in `/Users/dani/Documents/school/GNN Course/FastKVzip` and continue the approved standalone analysis. First verify that all summary jobs finished successfully and collect the final missing outputs. Then calculate the four requested metrics on the full common cohort, using the saved summaries and the full-cache self-agreement normalization described here. Run locally on my Mac, preserve progress for resumption, and make no evaluation implementation changes in the repository. Do not generate new summaries or submit cluster jobs without my approval.

The current request is only to write this handoff. Do not interpret its creation as starting the deferred scoring run.

**User intent and authorization**

The user approved a local, standalone analysis of saved summaries using:

1. BERTScore F1 for semantic/content similarity.
2. ROUGE-2 F1 for local phrasing.
3. Complete-summary ROUGE-L F1 for wording and word order.
4. Cosine similarity of StyleDistance embeddings for writing style.

The user explicitly wants this **without implementing the metrics in the repository**, and wanted to wait until all summarization jobs completed. Use isolated analysis scripts, environments and artifacts; do not modify production evaluators, open another feature PR, or rerun generation for this task. The user already approved the analysis; routine local work does not need another confirmation. New cluster submissions still require approval before submission. Do not send messages to other people.

The latest user message says all summary runs appear finished. **That has not been independently verified in this handoff-writing turn.** Verify completion first rather than treating that observation as an audited result. No automatic watcher or deferred scoring process has been scheduled.

**Where everything is**

All paths below are local unless explicitly marked remote. Define these names for reading this document:

```text
ROOT = /Users/dani/Documents/school/GNN Course/FastKVzip
GRID = ROOT/.slurm/grids/summary-three-methods-n16
SNAP = GRID/results-snapshot-20260912T084629Z
PAIR = SNAP/pairwise-normalization-20260912
PILOT = SNAP/neural-metric-feasibility-20260912
```

These are explanatory abbreviations, not preconfigured shell variables. Quote real paths because ROOT contains spaces.

| Artifact | Purpose |
|---|---|
| `GRID/manifest.json` | Original immutable approved 36-shard manifest; exact methods, arguments, account aliases, output paths, selected indices and document IDs |
| `GRID/active-manifest.json` | Same experiment arguments, updated runtime paths for the resume fix |
| `GRID/submission-receipt.json` | Logical shard identities and all recorded attempts; one row has a replacement attempt |
| `GRID/submission-receipt-original-36.json` | Original submission history |
| `GRID/dashboard-projects.json` | Dashboard projects and attempt IDs |
| `GRID/deferred-analysis-intent.json` | Approved four-metric scope and completion prerequisite |
| `GRID/odedshah-current-progress-20260912.json` | Last independently verified progress of the final job |
| `SNAP/{user}.json` | Saved metrics and complete run/sample manifests for each account |
| `SNAP/interim-report.md` and `interim-results.json` | Existing matched-ROUGE results and exact interim cohorts |
| `SNAP/aggregate.py` | Validated document-weighted aggregation of completed shard means |
| `SNAP/audit_saved_outputs.py` | Read-only remote output integrity audit; contains old cohort assumptions |
| `SNAP/output-integrity-audit-20260912T085105Z/` | Audit receipts, input fingerprints and raw-file SHA256s for the 33 common completed shards |
| `PAIR/{user}.jsonl.gz` | Local, hash-verified text copies of the 149,760 summaries already analyzed |
| `PAIR/collection-receipt.json` | Successful seven-account text collection, checksums and counts |
| `PAIR/retrieve_texts.py` | Text retrieval script, tied to the old audit/cohort |
| `PAIR/analyze_pairwise.py` | Tested standalone normalized pairwise ROUGE-L analysis |
| `PAIR/per-document.jsonl` | Reusable per-document raw baseline, cross similarity, normalized score, pruned self-similarity and matched ROUGE-L |
| `PAIR/aggregate.json`, `report.md` | Normalized interim results, all four length bins and protocol details |
| `PAIR/matched-verification.json` | 198 successful checks against original saved matched-ROUGE means |
| `PILOT/roberta-mps-pilot.json` | Actual local GPU encoding benchmark |

The feature worktree is `ROOT/.worktrees/summary-evaluation`, branch `feature/summary-evaluation`. The known feature PR is https://github.com/d4nieldev/FastKVzip/pull/23. Last verified branch head was `ed9acd0aae6e1d1cd3ae03377e4ebbb1e45dc96f`. Its current merge status is not known. Do not pull/reset/rebase unrelated work just to perform this standalone analysis. Preserve the user's existing uncommitted root-worktree changes.

**What was generated**

All runs use base LM `Qwen/Qwen2.5-7B-Instruct-1M`, model/tokenizer revision `e28526f7bb80e2a9c8af03b831a9af3812f18fba`. Three pruning methods were evaluated: GraphKV with whole-context graph pruning, FastKVzip with chunked pruning, and KVzip. Exact implementation paths and settings are in the manifest.

The GraphKV checkpoint is `graph_checkpoints/answer/q25a-s40n40-n200e2-uniform-s0/best.pt`, SHA256 `1f3e7c5354a966aa4409af9e609268e796fbec813e7b7d3c683843d9ba177ad0`. This is the previously approved evaluation-only checkpoint; do not launch training.

```text
N = 16 samples per document per condition
retention ratios = [1, 0.75, 0.5, 0.4, 0.3, 0.2]
temperature = 0.7
top_p = 0.9
top_k = 0
max_new_tokens = 1024
```

Each method/document therefore has 96 active summaries. The full-cache pool is reused as the reference across all five pruning ratios. Generation reused prepared KV caches within each condition. There is no N/2 split: every pool contains all 16 samples.

The fixed request was:

> Summarize the supplied document in approximately 400–500 words. Cover its main developments, central ideas, and important conclusions. Base the summary on the supplied text.

Actual outputs can exceed that approximate word target. Do not truncate them to 500 words for scoring.

| Dataset | Source test documents | Eligible full documents per method | Original exclusions |
|---|---:|---:|---|
| GovReport | 973 | 493 | 480 below 8,192 document tokens |
| PG-19 | 100 | 77 | 5 below 8,192; 18 at or above 131,072 document tokens |

Documents were selected reproducibly using actual Qwen tokenizer lengths, with no document truncation. The complete formatted prompt plus output allowance was checked against the model limit. No additional model-limit exclusions occurred.

Report datasets separately and use bins `[8192,16384)`, `[16384,32768)`, `[32768,65536)`, `[65536,131072)` (k=1024). Final available bin counts are GovReport **392/97/4/0** and PG-19 **3/15/25/34**.

The final target is **570 documents per method**, **1,710 method/document records**, and **164,160 active summaries**. There are 36 logical one-GPU shards: 30 GovReport shards and 6 PG-19 shards. Most process 50 documents; tails have 43 GovReport or 27 PG-19 documents.

**Cluster access and job identity**

Apply the available `running-fastkvzip-experiments` and `using-bgu-slurm` skills when reading cluster outputs. The configured endpoint is `slurm.bgu.ac.il:22`; use existing SSH aliases. Follow the skill's effective-config/DNS/port-22 connection gate. If VPN is unavailable, ask the user to connect it and stop network attempts. Do not change credentials, SSH configuration, host keys or endpoints. Do not run heavy metric computation on a login node.

| User | SSH alias | Remote project |
|---|---|---|
| danieloh | `bgu-slurm` | `/home/danieloh/FastKVzip-ruler-evaluation` |
| dulbergg | `bgu-slurm-dulbergg` | `/home/dulbergg/FastKVzip-ruler-evaluation` |
| guyzagor | `bgu-slurm-guyzagor` | `/home/guyzagor/danieloh/FastKVzip-ruler-evaluation` |
| liranatt | `bgu-slurm-liranatt` | `/home/liranatt/FastKVzip-ruler-evaluation` |
| odedshah | `bgu-slurm-odedshah` | `/home/odedshah/FastKVzip-ruler-evaluation` |
| shkabatu | `bgu-slurm-shkabatu` | `/home/shkabatu/FastKVzip-ruler-evaluation` |
| josefye | `bgu-slurm-josefye` | `/home/josefye/FastKVzip-ruler-evaluation` |

Use bounded, noninteractive SSH. Recent reads used `BatchMode=yes`, `ConnectTimeout=10`, `ServerAliveInterval=15`, `ServerAliveCountMax=2`, `ControlMaster=no`, `ControlPath=none`, and an outer subprocess timeout. SSH authentication/file reads can be slow. One automatic approval review also timed out; a shorter, read-only retry succeeded. A timeout is not evidence of experiment failure. Only retry read operations in a bounded way; never blindly repeat a submission.

| Array | Original ID | Task indices |
|---|---|---|
| danieloh RTX | 21174855 | 0–4 |
| dulbergg RTX | 21174856 | 0–4 |
| guyzagor RTX | 21174857 | 0–4 |
| liranatt RTX | 21174858 | 0–4 |
| odedshah RTX | 21174859 | 0–4 |
| shkabatu RTX | 21174860 | 0–4 |
| danieloh PRO | 21174861 | 0 |
| dulbergg PRO | 21174862 | 0 |
| guyzagor PRO | 21174863 | 0 |
| liranatt PRO | 21174864 | 0 |
| josefye PRO | 21174865 | 0–1 |

**Replacement:** failed `21174856_0` was replaced by **`21183487_0`**, using the same logical output directory. The replacement was subsequently verified complete. Keep the failed attempt as history; it does not make the logical experiment incomplete. There are 37 attempt records for 36 logical jobs.

Dashboard: https://graphfastkvzip.onrender.com. Projects are `summary-graphkv-n16`, `summary-fastkvzip-n16`, and `summary-kvzip-n16`. KVzip includes the failed historical attempt and its replacement. Do not create retry projects or new projects for this continuation.

At the last direct check, **12 September 12:56 Israel time**, only `21174859_3` remained running. It is GraphKV GovReport indices 300–349, named `sum-odedshah-rtx6000`. It had 45/50 documents complete and 4,336/4,800 summaries, actively writing document 46. Its log was:

```text
/home/odedshah/FastKVzip-ruler-evaluation/.slurm/logs/odedshah-rtx6000-21174859_3.log
```

The user now reports it appears finished. First verify this job's final accepted state and complete outputs. For accounting, a query through one Unix user did not return other users' jobs even with broad flags; do not interpret invisible accounting records as failed/missing jobs. Use each owning alias where necessary, saved outputs and dashboard evidence. A complete Slurm state alone does not replace output validation.

**Saved output schema and recovery history**

Each remote logical run is under its manifest's `run_dir`. Relevant paths beneath it are:

```text
manifest.json
metrics.json
samples/<dataset>/manifest.json
samples/<dataset>/examples/<selected-index>.json
```

An example JSON contains `identity`, `metadata`, `num_generations`, and `ratios`. Each ratio contains `actual_retention` and `samples`. Samples preserve `index`, full `text`, `token_ids`, `token_count`, `seed`, and `finish_reason`. Input/document hashes, runtime, prompt/prefix, decoding and pruning identities are recorded. Duplicates must remain distinct samples.

The resume fix in commit `ed9acd0` archives an incomplete condition's pool before regenerating all 16 samples using a rebuilt cache. Completed pools remain untouched. Inline `superseded_pools` and separate `recovery/` artifacts are historical, **not active samples**. A manual dulbergg recovery archived three partial summaries; the common-cohort audit later found 31 inline superseded samples excluded from scoring. Never merge historical partial pools back into current outputs.

The fixed runtime was deployed to all seven accounts. Remote `GRID/manifest.json` points to the fixed runtime, while the local original manifest intentionally remains historical. **Do not copy the local historical manifest over remote active manifests.**

Full-reference RNG seeds exclude pruning identity. Full pools from different methods can therefore share seeds and identical outputs; do not treat them as independent additional reference generations.

**The distinction between the two scoring protocols already used**

Original saved metrics use a 16×16 ROUGE-L F1 matrix and a Hungarian assignment maximizing total similarity. ROUGE-1/2 are evaluated on those same matched pairs. Standard ROUGE tokenization, complete-summary `rougeL`, no stemming. Ratio 1 matches a pool with itself and is 100 by construction. The original `metrics.json` files contain **ROUGE-1, ROUGE-2 and ROUGE-L; no BLEU**.

For summarization shards, `cohort_complete=true` means every selected document has all required N=16 pools. `complete=false` can still be expected for a 50-document shard of a 493-document dataset. Do not mistake that for an unfinished shard. Existing completed-shard means may be combined by actual document-count weighting, using bin-specific counts for bins. Never average shard means without their weights.

The user subsequently requested normalization against natural full-cache variation. The implemented standalone ROUGE-L analysis uses **ordinary pairwise means, not Hungarian matching**:

```text
f[1..16] = full-cache summaries; p[1..16] = summaries at one pruned ratio
R(a,b) = complete-summary ROUGE-L F1 in [0,1]
       = 2*LCS(tokens(a),tokens(b)) / (len(tokens(a)) + len(tokens(b)))

B_d = mean R(f_i,f_j) over i<j                 # 120 distinct-index full/full pairs
C_d = mean R(f_i,p_j) over all i,j             # 256 full/pruned pairs
normalized_d = 100*C_d/B_d
dataset score = mean_d(normalized_d)          # equal document weights
```

Compute the ratio **per document before averaging**. This differs from dividing dataset-average C by dataset-average B. Do not divide the old Hungarian score by B: numerator and denominator would use different estimators. Do not introduce an N/2 split or additional generations.

Also retain raw B, C, `100*(C-B)`, and pruned/pruned agreement over 120 distinct-index pairs. Duplicate texts remain. Values above 100 are allowed; the metric measures relative lexical agreement, not accuracy or equality of distributions. B=0 is undefined; do not add an epsilon or silently drop inconsistent cohorts. No zero baselines occurred in the completed interim ROUGE-L analysis. For cosine or rescaled neural scores, explicitly handle nonpositive/near-zero baselines rather than assuming the ROUGE behavior applies.

The four-metric continuation should retain raw scores and baselines alongside normalized results. The user's normalization discussion concerned this pairwise protocol. BERTScore's optional built-in baseline rescaling is a different operation; do not silently substitute it for full-cache normalization.

**What is already available locally, and what is missing**

The interim common cohort was **443 GovReport reports + 77 PG-19 books per method**: 33 shards, 1,560 method/document records, 149,760 active summaries. All 50 GovReport indices 300–349 were excluded from **all three methods** because the GraphKV shard was unfinished. That kept method/ratio comparisons on identical documents.

The full current text is locally stored in seven `PAIR/{user}.jsonl.gz` files, total about167 MB compressed. Each record has `name`, `method`, `dataset`, `index`, `metadata`, raw-source `file_sha256`, and `ratios` mapping to 16 `{index,text,seed}` samples. They omit token IDs to reduce transfer size; full original JSONs remain on the cluster.

The existing audit verified exact input/runtime/metadata agreement across methods, all active sample indices 0–15, six conditions, and no errors. Retrieval verified every source file hash against that audit. Local checksums are in `collection-receipt.json`.

For final coverage, add **exactly these three omitted 50-document shards** (14,400 summaries):

| Method | Account / accepted job | Run directory beneath that account's remote project |
|---|---|---|
| GraphKV | odedshah / `21174859_3` | `results/q25-graphkv-summary-n16-govreport_summary-i0300-n50` |
| FastKVzip | odedshah / `21174859_0` | `results/q25-fastkvzip-summary-n16-govreport_summary-i0300-n50` |
| KVzip | guyzagor / `21174857_3` | `results/q25-kvzip-summary-n16-govreport_summary-i0300-n50` |

Verify these identities against `manifest.json` before retrieval. Reuse the existing 33-shard copies when their immutable completed outputs remain compatible. Fetch into a new final snapshot; preserve all interim artifacts.

**Important:** the current audit, retrieval and analysis scripts deliberately hardcode the interim exclusion, 33 shards, and/or 443 GovReport documents. Do not run them unchanged and label their output final. Make standalone copies or adapt orchestration outside production source to cover all36 shards/493 GovReport documents. The generic repository merger can combine complementary partial sample indices; do not feed it pre-recovery archives or overlapping historical copies. One current accepted copy per logical document is sufficient.

Existing normalized pairwise ROUGE-L results, for reference:

| Dataset | Method | 75% | 50% | 40% | 30% | 20% |
|---|---|---:|---:|---:|---:|---:|
| GovReport,443 | GraphKV | 100.13 | 99.32 | 99.05 | 97.41 | 94.61 |
| GovReport,443 | FastKVzip | 99.43 | 98.91 | 99.02 | 97.99 | 93.52 |
| GovReport,443 | KVzip | 99.94 | 99.26 | 98.73 | 97.52 | 92.98 |
| PG-19,77 | GraphKV | 100.08 | 99.86 | 99.40 | 99.14 | 98.24 |
| PG-19,77 | FastKVzip | 99.59 | 99.84 | 100.00 | 99.52 | 97.73 |
| PG-19,77 | KVzip | 100.04 | 100.02 | 99.84 | 99.42 | 98.36 |

Mean full/full pairwise ROUGE-L was31.86–31.88 on GovReport and26.41 on PG-19. These are not the older optimal-matching scores. See the linked artifact paths above for exact values and all bins.

The optimized ROUGE-L uses official `rouge_score.tokenize.tokenize(text,None)` and RapidFuzz `LCSseq.similarity`, with F1 calculated explicitly; RapidFuzz's `normalized_similarity` has a different denominator and must not be substituted. It was checked against official ROUGE on synthetic and real pairs. All198 original shard/ratio matched-score comparisons passed, maximum error2.84e-14. The four-worker full interim analysis took47.43seconds. Reuse its per-document results; only150 new records need additional ROUGE-L work for final coverage.

**Mac, environment and neural metric decisions**

Verified Mac: Apple M1 Max,32 GPU cores,10 CPU cores,64 GiB unified memory, approximately408 GiB free disk at inspection. PyTorch MPS was built and available. Recheck availability rather than falling back silently to CPU.

Existing temporary Python environment: `/private/tmp/fastkvzip-summary-venv/bin/python`, Python3.12.10. Known packages: torch2.7.0, transformers4.51.3, numpy1.26.4, scipy1.14.1, rouge-score0.1.2, rapidfuzz3.14.3. `bert-score` and `sentence-transformers` were **not installed** at the last check. A temporary environment may disappear; inspect it and recreate an isolated environment if needed. Avoid modifying the user's global environment.

Suggested scoring configuration, not yet a completed neural evaluation:

- BERTScore: English `roberta-large`, layer17, `idf=False`, built-in `rescale_with_baseline=False`; record tokenizer choice, precision and exact versions. Full weights/tokenizer are already cached at `~/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59`. The safetensors file is about1.42GB. Use the cached pinned model where compatible. This is token-level contextual matching, not cosine of one mean summary embedding.
- StyleDistance: `StyleDistance/styledistance`, verified revision `b7df5f0b0480773c097ba3121d83ca32b71015ca`. RoBERTa-base,124,645,632 parameters,768-dimensional mean-pooled embeddings, about249MB BF16 weights. These weights were not downloaded during feasibility checks. Use cosine **similarity** (higher means closer style), not `1-cosine` unless explicitly labeled distance.
- Both usual encoders have a512-token window including special tokens, ordinarily510 content tokens. In a deterministic420-summary sample,414 exceeded510 tokens; median content length695. Standard encoding would truncate almost all of that sample. Merely increasing the configured max length is not a fix.
- Full-text handling must be explicit. Broad full-text chunking was discussed and the user agreed to proceed, but exact chunk boundaries/overlap/aggregation were not fixed. Choose and record a deterministic policy before full scoring; clearly label it a chunked adaptation. A simple starting option is nonoverlapping token-bounded windows covering every token. For BERTScore, reconstruct full token representations and apply token-level matching while handling boundary/special tokens consistently. For StyleDistance, token-weighted averaging of chunk vectors followed by L2 normalization is a proposed adaptation. Do not silently describe either as the untouched single-window metric.
- Encode each summary once per scoring model/protocol and reuse it across pair comparisons. For BERTScore, process bounded document groups rather than retaining every token embedding for all164k summaries in RAM. StyleDistance's complete vector set is only hundreds of MB. Preserve repeated sample indices even if identical text embeddings are cached once.

Actual feasibility measurements:

| Work | Evidence / estimate |
|---|---|
| Pairwise ROUGE-L interim analysis |47.43s with4 local workers, including original matched-score checks |
| ROUGE-2 pilot |20 documents/28,000 pairs in3.37s; projects to263s serial for the interim cohort, including tokenization but excluding input loading |
| RoBERTa-large encoder pilot |17 layers, FP32, MPS, batch4/8:26.24/27.89 padded windows/sec; only32 timed chunks, not a sustained endurance test |
| BERTScore encoding extrapolation |About3.0–3.2h for149,760 summaries at roughly2 windows each; final corpus is about9.6% larger |
| Synthetic MPS token-alignment pilot |About3,133 pairs/sec at700×700 tokens,1,939 at1000×1000; projects to12–19min for2.184million interim pairs, excluding transfer/masking/other overhead |
| Practical BERTScore estimate previously given |Roughly3–5h for the interim cohort, assuming embedding reuse; temperature, sustained load, chunking and I/O can change this |
| StyleDistance |Memory feasible, expected lighter than BERTScore, but no model-specific throughput benchmark yet |

Only short timing pilots ran. **No full BERTScore or StyleDistance evaluation has started.** Neural metrics and full-final-cohort pairwise ROUGE-2 remain to be done. No BLEU computation was requested as part of these four metrics.

Primary references if implementation details need checking:

- https://github.com/Tiiiger/bert_score — BERTScore interface and input-length limitations.
- https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/utils.py — model/layer defaults and token matching.
- https://huggingface.co/StyleDistance/styledistance — official style model.
- https://huggingface.co/StyleDistance/styledistance/blob/main/sentence_bert_config.json — model window.
- https://huggingface.co/StyleDistance/styledistance/blob/main/1_Pooling/config.json — pooling configuration.

**Continuation sequence and completion criteria**

1. Read this handoff and the small identity/receipt files. Verify the final accepted job and outputs. Preserve historical failures; do not resubmit or restart anything just because an original attempt failed.
2. Audit and retrieve the three missing GovReport shards into a new final snapshot. Require complete active16-sample pools for all6 ratios, correct source identities and the same final documents across methods. Expected final counts:36 logical shards,1,710 records,164,160 samples. Exclude archives. Reuse existing verified local texts.
3. Record the four-metric protocol, model revisions, tokenizers, precision, chunking, baseline handling and aggregation. Validate optimized calculations against official implementations on small inputs. Compare newly recomputed matched ROUGE where useful, without changing the requested pairwise normalization.
4. Run a small StyleDistance/MPS compatibility and throughput check. Use bounded batches. Then run the standalone local analysis with durable per-document completion records, so interruption or a quota reset does not require recomputing completed embeddings/scores. Save raw B,C,normalized values and pruned self-agreement where calculated.
5. Reuse old per-document ROUGE-L and calculate only the missing150 records before final aggregation. Compute the other requested metrics on the final common cohort. Preserve all16 samples; no new generation and no N/2 split.
6. Produce a concise report and machine-readable per-document/aggregate results for every method, ratio, dataset and four document-length bins. Include cohort counts, raw baselines, raw cross scores and normalized scores. Explain that these measure semantic/lexical/style similarity rather than factual accuracy or proof of identical distributions. Record any undefined normalizations and preserve consistent comparison cohorts.
7. Report final artifact paths, runtime and material limitations. Do not publish, merge, change dashboard projects, or submit cluster work unless separately authorized. Keep progress messages useful and brief; once the requested work is done, finish rather than continuing optional checks indefinitely.

If the session is short on quota again, prioritize a durable progress record and exact resume commands over repeating research, re-reading every source, re-downloading validated artifacts, or recomputing completed metrics.
