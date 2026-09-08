# Uniform evaluation: approved retry execution

Submission completed **2026-09-08 20:56 UTC** following the user's approval of
the 54-row replacement manifest. **51 attempts were submitted**: 26 failed
originals resumed and 25 still-pending originals cancelled individually and
replaced. Three originals had started and were left untouched:
`21114647` (SCBench VT), `21114652` (RULER multikey-3 32K), and `21114654`
(RULER VT 32K). No additional pilots were submitted.

## Preserved protocol and artifacts

- Same uniform `best.pt`; all six accounts verified checkpoint SHA-256
  `1f3e7c5354a966aa4409af9e609268e796fbec813e7b7d3c683843d9ba177ad0`.
- Fixed runtime `cea2400b6fd67c75503f99311d6874692675905d` staged separately
  from every live checkout. Existing running jobs/code were not changed.
- Window 0, pair pruning, GraphKV prompts, complete datasets, ratios
  0.75/0.50/0.40/0.30/0.20, full-cache baseline and resume mode. No Agentic jobs.
- Fifty RTX 6000 attempts use graph/token microbatches 8/16000; one Shkabatu
  reserved PRO attempt uses 16/16000 and the verified shared `eliasof` QoS.
- Five partial evaluations remain in their original Daniel directories, with
  1,817 already complete examples and two partial examples preserved.
- All workers disable direct W&B uploads and bind evaluation-only run
  [hdo2z4x4](https://wandb.ai/danielohayon2016-ben-gurion-university-of-the-negev/graphkv-answer-qwen25-7b1m-s40n40-grid-v1/runs/hdo2z4x4).
  Source training run `61ewyfcm` is not used for uploads.
- All 51 new IDs were assigned to the existing dashboard project
  `graphkv-answer-qwen25-7b1m-s40n40-grid-v1`; API response confirmed
  `assigned: 51`. Earlier attempts remain recorded.

Preflight caught one stale proposed environment path for Oded. The actual
commands use his existing, validated `/home/odedshah/FastKVzip/.venv`, confirmed
from running production job `21114607`, instead of the nonexistent
`/home/odedshah/.venvs/fastkvzip`. This narrow correction is explicitly recorded
in the receipt; resources, packages, model and evaluation identities did not
change. The originally approved manifest was not rewritten.

## Submission and startup verification

Every source was rechecked before replacement. Cancellation used installed
`scancel --ctld --state=PENDING` with one exact approved ID, followed by terminal
state and zero-output checks. Every submission intent preceded `sbatch`, and
its genuine returned ID was saved before the next submission. There were no
ambiguous outcomes. An independent read-only audit checked the effective
scheduler request for all 51 replacements.

At **20:59:30 UTC**, the 107 logical benchmarks comprised **46 completed,
24 running, and 37 pending**. Of the 51 replacements, 14 were running and 37
pending. No replacement had failed in this snapshot; completion is not yet
claimed. Queued jobs remain subject to physical availability, priority, and
normal per-user/shared-account limits. Unrelated users' jobs were not changed.

Startup evidence confirms real RTX 6000 Ada hardware (47.4 GiB) and successful
partial resumes. The resumed 128K FWE job finished its previously partial
example at 30.3 GiB peak allocated GPU memory; completed 64K examples peaked
around 27.5 GiB and a 112K SCBench example reached scoring at 30.0 GiB. These are
observations from production, not full-benchmark memory guarantees or scores.

The existing serial five-minute collector now uses the retry-aware tracked
collector, with one process lock and the existing collection/upload lock. It
collects only compatible successful terminal attempts, uploads only to
`hdo2z4x4`, and stops on failures or connection errors without resubmitting.
It runs locally, so uploads require this computer and BGU VPN to remain
available; Slurm evaluation jobs do not depend on that local process.

## Attempt inventory

| Account | New attempts |
| --- | ---: |
| danieloh | 14 |
| shkabatu | 7 |
| dulbergg | 8 |
| liranatt | 5 |
| odedshah | 8 |
| guyzagor | 9 |

| Benchmark | Destination account | Original job | Replacement job |
| --- | --- | ---: | ---: |
| ruler_niah_multikey_2_128k | danieloh | 21114623 | 21143608 |
| ruler_fwe_128k | danieloh | 21114626 | 21143609 |
| ruler_qa_1_128k | danieloh | 21114625 | 21143610 |
| ruler_niah_multiquery_64k | danieloh | 21114628 | 21143612 |
| ruler_cwe_64k | danieloh | 21114629 | 21143614 |
| ruler_niah_multivalue_64k | shkabatu | 21114631 | 21143615 |
| ruler_niah_multikey_3_64k | dulbergg | 21114632 | 21143618 |
| ruler_qa_2_64k | liranatt | 21114641 | 21143621 |
| scbench_mf | shkabatu | 21114644 | 21143623 |
| scbench_prefix_suffix | dulbergg | 21114648 | 21143626 |
| ruler_cwe_32k | shkabatu | 21114649 | 21143628 |
| ruler_niah_single_3_32k | dulbergg | 21114653 | 21143629 |
| ruler_niah_single_1_32k | dulbergg | 21114655 | 21143631 |
| ruler_niah_single_2_32k | shkabatu | 21114657 | 21143632 |
| ruler_niah_multikey_1_32k | liranatt | 21114658 | 21143633 |
| ruler_niah_multikey_2_32k | dulbergg | 21114659 | 21143640 |
| ruler_fwe_32k | odedshah | 21114660 | 21143641 |
| ruler_niah_multiquery_16k | guyzagor | 21114661 | 21143642 |
| scbench_mf_mid | guyzagor | 21114662 | 21143643 |
| ruler_qa_1_32k | odedshah | 21114663 | 21143644 |
| ruler_qa_2_32k | guyzagor | 21114664 | 21143645 |
| scbench_prefix_suffix_mid | guyzagor | 21114665 | 21143646 |
| scbench_kv_short | odedshah | 21114666 | 21143647 |
| scbench_repoqa_short | danieloh | 21114667 | 21143650 |
| ruler_cwe_16k | danieloh | 21114668 | 21143651 |
| ruler_niah_multivalue_16k | odedshah | 21114669 | 21143652 |
| ruler_niah_single_3_16k | danieloh | 21114670 | 21143653 |
| ruler_niah_multikey_3_16k | guyzagor | 21114672 | 21143654 |
| scbench_kv_tiny | danieloh | 21114673 | 21143655 |
| gsm | dulbergg | 21114674 | 21143656 |
| ruler_vt_16k | liranatt | 21114675 | 21143657 |
| scbench_mf_short | shkabatu | 21114676 | 21143659 |
| ruler_niah_single_2_16k | dulbergg | 21114677 | 21143661 |
| ruler_niah_multikey_1_16k | odedshah | 21114678 | 21143664 |
| ruler_niah_single_1_16k | guyzagor | 21114679 | 21143665 |
| ruler_niah_multikey_2_16k | liranatt | 21114680 | 21143667 |
| ruler_fwe_16k | guyzagor | 21114681 | 21143677 |
| scbench_prefix_suffix_short | danieloh | 21114682 | 21143678 |
| ruler_qa_2_16k | danieloh | 21114683 | 21143680 |
| ruler_qa_1_16k | guyzagor | 21114688 | 21143681 |
| scbench_mf_tiny | dulbergg | 21114689 | 21143682 |
| scbench_prefix_suffix_tiny | guyzagor | 21114690 | 21143684 |
| scbench_repoqa_tiny | odedshah | 21114691 | 21143685 |
| scbench_summary_tiny | liranatt | 21114692 | 21143688 |
| scbench_summary_mid | odedshah | 21114693 | 21143689 |
| scbench_summary_short | danieloh | 21114694 | 21143692 |
| scbench_qa_eng | odedshah | 21114695 | 21143696 |
| scbench_choice_eng | danieloh | 21114699 | 21143697 |
| scbench_many_shot | shkabatu | 21114700 | 21143698 |
| scbench_many_shot_short | shkabatu | 21114701 | 21143700 |
| scbench_many_shot_tiny | danieloh | 21114702 | 21143702 |

## Durable receipts

The local grid directory is `.slurm/grids/uniform-eval-all-w0/`. Original
`manifest.json` and `receipt.json` remain unchanged. New `retry-manifest.json`,
`retry-receipt.json`, and `dashboard-retry-assignment.json` preserve every
attempt, exact invocation, skip, cancellation and dashboard acknowledgment.
All three files were copied to every account's isolated project under
`.slurm/grids/uniform-eval-all-w0/retry-execution-20260908T205629Z/` and their
file SHA-256 values matched on all six accounts.

Approved canonical retry-manifest SHA-256:
`1aa757d0a32fd40e365b1d523b74beb29ec20e291170c28d0ec34090e7c0a121`.

The collector's local-only validation accepted the original 107 attempts plus
51 replacements (158 total), excluding the three explicitly reconciled skips.
Submission/watcher compilation and command syntax checks passed; the fixed
runtime's [426-test verification](ruler-review-fixes.md) preceded deployment.
