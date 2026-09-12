# Implementation decisions

Legend: 🟠 Plan gap identifies a choice the approved requirement left open.

## D1 — 🟠 Plan gap — Preserve resume when existing history is ahead

- **Background:** W&B history is append-only. Its next writable Step can be ahead of the checkpoint's next optimizer update, either because an older trainer logged validation separately or because the checkpoint predates already-uploaded training rows. Explicitly logging a smaller Step would discard metrics. The original K=1 run has separate validation rows at Step 180 and Step 361, despite one optimizer update per training question.
- **Decision:** New runs and aligned resumes explicitly commit each optimizer count as native W&B Step, starting at 1. If existing W&B history is already ahead, retain its previous append numbering and emit a warning. The existing `train/optimizer_step` metric continues to carry the actual optimizer count in that compatibility case. No additional counter is saved.
- **Plan gap or deviation:** The requirement specifies the desired axis and meaning but does not specify how to handle immutable incompatible history. The user was asked about compatibility and replied that K=1 historically meant one update per training example. The separate validation rows were then explained; this entry does not treat that reply as explicit approval of a fallback policy.
- **Reason and tradeoff:** Preserving the existing resume behavior avoids rejecting otherwise valid checkpoints, dropping resumed metrics, rewriting historical data, or creating a different run identity. Native Step is not the optimizer count for this exceptional continuation. The warning and documentation make that limitation explicit.
- **Status:** 🟠 Plan gap. Implemented as a compatibility choice within the logging fix; no separate user approval is claimed.

| Code reference | What this code does |
| --- | --- |
| [History compatibility check](https://github.com/d4nieldev/FastKVzip/blob/3f53ccb417b21a79b90882577496cd005be68984/prefill/train_graph_answer.py#L1303) | Compares W&B's next writable Step with the next optimizer count and warns when history must retain its numbering. |
| [Update logging](https://github.com/d4nieldev/FastKVzip/blob/3f53ccb417b21a79b90882577496cd005be68984/prefill/train_graph_answer.py#L1432) | Commits an explicit optimizer Step for aligned history, or appends for the compatibility case. |
| [Resume history test](https://github.com/d4nieldev/FastKVzip/blob/3f53ccb417b21a79b90882577496cd005be68984/prefill/tests/test_answer_batching.py#L410) | Exercises ahead-of-checkpoint history while checking that resumed parameters, optimizer state, and counters remain identical. |

## D2 — 🟠 Plan gap — Include prefill size in W&B configuration

- **Background:** The accuracy investigation compared runs with different prefill chunk sizes. That size was saved at the checkpoint's top level and in launch manifests, but omitted from W&B configuration, making the run comparison incomplete.
- **Decision:** Add the resolved `prefill_chunk` value to the W&B configuration update. Leave the checkpoint configuration and strict resume compatibility checks unchanged.
- **Plan gap or deviation:** The approved code requirement focuses on the Step axis. This small additional provenance fix arose directly from investigating the user's question about apparently identical run settings.
- **Reason and tradeoff:** Exposing the actual prefill size makes future comparisons accurate without changing training behavior or checkpoint compatibility. The active run receives this metadata at startup or resume; this change does not backfill other completed runs.
- **Status:** 🟠 Plan gap. Implemented as a related logging fix; no separate user approval is claimed.

| Code reference | What this code does |
| --- | --- |
| [W&B configuration update](https://github.com/d4nieldev/FastKVzip/blob/3f53ccb417b21a79b90882577496cd005be68984/prefill/train_graph_answer.py#L1295) | Adds the resolved prefill size only to the dictionary sent to W&B. |
| [Configuration provenance assertion](https://github.com/d4nieldev/FastKVzip/blob/3f53ccb417b21a79b90882577496cd005be68984/prefill/tests/test_answer_batching.py#L275) | Checks the logged value against the actual checkpoint prefill setting. |
