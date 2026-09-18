# Pilot progress log

Append one dated line per wakeup. Newest at the bottom.

## 2026-09-18

- 12:00 UTC — Preflight green: cluster reachable, 13 free rtx_pro_6000, fresh
  clone at d9f6e82, 189 graph tests pass in the cluster venv (torch 2.7.0).
- 12:07 — Submitted then cancelled 21450032: wrong model (Qwen3-8B) and
  whole-context, neither of which production has used since Grid 6.
- 12:14 — Phase 0 (21450072) COMPLETED in 6m05s. Scores-only cache complete at
  6 files / 64 MB. Train BCE 0.162, mean alpha 0.0995. `last.pt` written.
- 12:20 — Submitted phases 1 and 2: 16 training jobs, 21450082-21450097. Plus
  evaluation 21450098 on the phase-0 checkpoint.
- 12:26 — GPS at graph microbatch 16 COMPLETED in 3m06s, half the implicit
  mixer's 6m05s at the same setting.
- 12:26 — Recorded the parameter-count finding: GPS is 26.9M against the
  implicit mixer's 39.3M at graph width 32. Smaller, not larger.
- 12:26 — First ceiling bracket: implicit + batchnorm OOMs at graph microbatch
  56 (width 448) and completes at 28 (width 224).
- 12:27 — Armed cron a2a9d84a, every 20 minutes, to keep the pilot advancing
  unattended.

- 13:0x UTC — All of phases 0-8 terminal. Phase 8 resolved the non-monotonic
  batchnorm peaks: a clean gm16 repeat gives 45.2 GiB, not 77.9, and the warm
  series 45.2/51.6/64.4 at gm 16/20/28 is exactly linear. The 77.9 came from
  p0's cold cache plus resumed runs inheriting their parent's W&B run.
  Corrected the earlier claim that 38.3 GiB was a dead-node artifact: the
  requeued runs reproduce it exactly. All three arms share a ~19.6 GiB
  intercept; slopes are granola 1.18, gps 1.49, batchnorm 1.60.
- Submitted phase 9 (21450518-19): stage-2 ladder at width 448, the last
  planned measurement. No findings; findings.md still records zero defects.
