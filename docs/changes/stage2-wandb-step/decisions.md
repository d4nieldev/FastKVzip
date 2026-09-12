# Implementation decisions

Legend: 🟢 User-approved amendment records an explicit change requested after the original plan. 🟠 Plan gap identifies a choice the approved requirement left open.

## D1 — 🟢 User-approved amendment — Remove compatibility renumbering

- **Background:** The initial implementation used separate W&B numbering when existing history was ahead of a resumed checkpoint. That made Step differ from the optimizer count for those continuations.
- **Decision:** Remove the fallback, history check, and warning. Always pass the completed optimizer count as W&B Step, including on resume.
- **Plan gap or deviation:** The original requirement did not specify how to handle history ahead of a checkpoint. The user explicitly rejected special handling: "this is a wild edge case I dont want this in my code".
- **Reason and tradeoff:** This keeps one step meaning and removes the compatibility branch. W&B's normal history rules apply without trainer-specific recovery.
- **Status:** 🟢 User-approved amendment. Implemented at the user's request; replaces the original D1 fallback.

| Code reference | What this code does |
| --- | --- |
| [Update logging](https://github.com/d4nieldev/FastKVzip/blob/a7ea573e81506af18b8f4bddac7faf75b23cffd4/prefill/train_graph_answer.py#L1423) | Always commits the completed optimizer count as W&B Step. |

## D2 — 🟠 Plan gap — Include prefill size in W&B configuration

- **Background:** The accuracy investigation compared runs with different prefill chunk sizes. That size was saved at the checkpoint's top level and in launch manifests, but omitted from W&B configuration, making the run comparison incomplete.
- **Decision:** Add the resolved `prefill_chunk` value to the W&B configuration update. Leave the checkpoint configuration and strict resume compatibility checks unchanged.
- **Plan gap or deviation:** The approved code requirement focuses on the Step axis. This small additional provenance fix arose directly from investigating the user's question about apparently identical run settings.
- **Reason and tradeoff:** Exposing the actual prefill size makes future comparisons accurate without changing training behavior or checkpoint compatibility. The active run receives this metadata at startup or resume; this change does not backfill other completed runs.
- **Status:** 🟠 Plan gap. Implemented as a related logging fix; no separate user approval is claimed.

| Code reference | What this code does |
| --- | --- |
| [W&B configuration update](https://github.com/d4nieldev/FastKVzip/blob/a7ea573e81506af18b8f4bddac7faf75b23cffd4/prefill/train_graph_answer.py#L1296) | Adds the resolved prefill size only to the dictionary sent to W&B. |
| [Configuration provenance assertion](https://github.com/d4nieldev/FastKVzip/blob/a7ea573e81506af18b8f4bddac7faf75b23cffd4/prefill/tests/test_answer_batching.py#L273) | Checks the logged value against the actual checkpoint prefill setting. |
