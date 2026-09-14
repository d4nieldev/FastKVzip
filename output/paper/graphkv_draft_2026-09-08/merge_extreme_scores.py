"""Merge SCBench extreme-ratio (5%, 10%) scores into data/method-scores.json.

Adds two new ratio keys to the score curves of the 11 SCBench *base* tasks
only (no length variants, no RULER) for all 3 methods, sourced from the same
three W&B runs already cited by the paper (later history steps, not new
runs). Everything else in the file -- the other 49 tasks, the original 6
ratios everywhere, all provenance for the base grid -- is left untouched.

Run with --check to verify an already-merged file without writing anything.
"""

import base64
import copy
import datetime as dt
import json
import math
import netrc
import sys
import urllib.request
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SNAPSHOT = PAPER / "data" / "method-scores.json"

ENTITY = "danielohayon2016-ben-gurion-university-of-the-negev"
PROJECT = "graphkv-answer-qwen25-7b1m-s40n40-grid-v1"
RUN_IDS = {"graphkv": "c0s997un", "fastkvzip": "vi31uf3h", "kvzip": "p2wsdvv5"}
NEW_RATIOS = ("0.05", "0.1")

BASE_TASK_KEYS = ("kv", "mf", "qa_eng", "repoqa", "summary", "choice_eng",
                  "many_shot", "prefix_suffix", "vt", "summary_with_needles",
                  "repoqa_and_kv")
TASKS = tuple(f"scbench_{key}" for key in BASE_TASK_KEYS)

RUNTIME_COMMITS = {"graphkv": "cea2400b6fd67c75503f99311d6874692675905d",
                   "fastkvzip": "265fde8fddba638630e735bd46f266ee46745373",
                   "kvzip": "265fde8fddba638630e735bd46f266ee46745373"}
GRID_MANIFEST_SHA256 = "1496d9dabe1b85ca4386de7e2f3b36a8f950ab6df87de660a268b1ae9a41ef9d"

# Current (post-recovery) job IDs per method, from the grid's receipt.json
# `current_jobs`, with the four replacement jobs submitted after the original
# receipt (superseding 21225517, 21225813, 21224549, 21224550 respectively).
JOB_IDS = {
    "graphkv": sorted(["21224535", "21224536", "21224542", "21228249",
                       "21224551", "21224554", "21224556", "21224557",
                       "21224559", "21224562", "21224566"]),
    "fastkvzip": sorted(["21224533", "21224534", "21224537", "21224540",
                         "21224544", "21224545", "21224546", "21224547",
                         "21224548", "21228248", "21228261"]),
    "kvzip": sorted(["21224543", "21224552", "21224553", "21224555",
                     "21224558", "21224560", "21224561", "21224563",
                     "21224564", "21224565", "21228245"]),
}


def wandb_key():
    return netrc.netrc().authenticators("api.wandb.ai")[2]


def gql(query, variables):
    key = wandb_key()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()},
    )
    return json.load(urllib.request.urlopen(req, timeout=120))


def fetch(method, run_id):
    q = """query($e:String!,$p:String!,$n:String!){
      project(name:$p, entityName:$e){ run(name:$n){ history(samples:5000) } }
    }"""
    d = gql(q, {"e": ENTITY, "p": PROJECT, "n": run_id})
    rows = d["data"]["project"]["run"]["history"]
    rows = [json.loads(r) if isinstance(r, str) else r for r in rows]
    values = {}
    for row in rows:
        ratio = row.get("test/retention_ratio")
        if ratio not in (0.05, 0.1):
            continue
        key = "0.05" if ratio == 0.05 else "0.1"
        for task in TASKS:
            v = row.get(f"test/{task}")
            if v is not None:
                values.setdefault(task, {})[key] = v
    missing = [t for t in TASKS if set(values.get(t, {})) != set(NEW_RATIOS)]
    assert not missing, f"{method}: incomplete coverage for {missing}"
    return values


def merge(snapshot):
    snapshot = copy.deepcopy(snapshot)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    added = 0
    for method, run_id in RUN_IDS.items():
        record = snapshot["methods"][method]
        assert record["run_id"] == run_id
        new_values = fetch(method, run_id)
        for task, curve in new_values.items():
            existing = record["scores"][task]
            for ratio, value in curve.items():
                assert math.isfinite(value) and 0 <= value <= 100, (method, task, ratio, value)
                if ratio in existing:
                    assert existing[ratio] == value, f"{method}/{task}/{ratio} conflicts with existing value"
                else:
                    existing[ratio] = value
                    added += 1
        record["score_points"] = sum(len(c) for c in record["scores"].values())
        record.setdefault("config", {}).setdefault("evaluation_extensions", {})["scbench-extreme-ratios"] = {
            "jobs": JOB_IDS[method],
            "protocol": "same frozen method runtime and checkpoint as the base grid; "
                        "isolated result directories; ratios 0.1 and 0.05 only, "
                        "11 SCBench base tasks only, no length variants, no RULER",
            "manifest_sha256": GRID_MANIFEST_SHA256,
            "runtime_commits": [RUNTIME_COMMITS[method]],
            "added_ratios": list(NEW_RATIOS),
            "added_benchmarks": list(TASKS),
        }
    snapshot["suite"]["extreme_ratio_tasks"] = list(TASKS)
    snapshot["extreme_ratio_extension"] = {
        "added_at": now,
        "ratios": list(NEW_RATIOS),
        "tasks": list(TASKS),
        "manifest_sha256": GRID_MANIFEST_SHA256,
        "history_source": "Read-only public W&B history, rows filtered to "
                           "test/retention_ratio in {0.05, 0.1}",
        "note": "The other 49 configurations (16 SCBench length variants, all "
                "33 RULER configurations) were not evaluated at these ratios "
                "and are unchanged.",
    }
    return snapshot, added


def check_merged(after):
    """Validate an already-merged file's internal consistency (no prior snapshot needed)."""
    assert "extreme_ratio_extension" in after
    assert set(after["extreme_ratio_extension"]["tasks"]) == set(TASKS)
    for method in RUN_IDS:
        for task in TASKS:
            assert set(after["methods"][method]["scores"][task]) == \
                {"0.2", "0.3", "0.4", "0.5", "0.75", "1.0"} | set(NEW_RATIOS)
        for task, curve in after["methods"][method]["scores"].items():
            if task not in TASKS:
                assert set(curve) == {"0.2", "0.3", "0.4", "0.5", "0.75", "1.0"}, task
        assert "scbench-extreme-ratios" in after["methods"][method]["config"]["evaluation_extensions"]


def check(before, after):
    # Everything outside the 11 tasks' two new keys must be byte-identical.
    b, a = copy.deepcopy(before), copy.deepcopy(after)
    for method in RUN_IDS:
        for task in TASKS:
            for ratio in NEW_RATIOS:
                a["methods"][method]["scores"][task].pop(ratio, None)
        a["methods"][method]["config"].get("evaluation_extensions", {}).pop("scbench-extreme-ratios", None)
        a["methods"][method].pop("score_points", None)
        b["methods"][method].pop("score_points", None)
    a["suite"].pop("extreme_ratio_tasks", None)
    a.pop("extreme_ratio_extension", None)
    assert a == b, "merge touched something outside the 11 base tasks' new ratio keys"
    for method in RUN_IDS:
        for task in TASKS:
            assert set(after["methods"][method]["scores"][task]) == {"0.2", "0.3", "0.4", "0.5", "0.75", "1.0"} | set(NEW_RATIOS)
        for task in after["methods"][method]["scores"]:
            if task not in TASKS:
                assert set(after["methods"][method]["scores"][task]) == {"0.2", "0.3", "0.4", "0.5", "0.75", "1.0"}


if __name__ == "__main__":
    before = json.loads(SNAPSHOT.read_text())
    if "--check" in sys.argv:
        if "extreme_ratio_extension" not in before:
            print("OK: no merge present yet")
            sys.exit(0)
        check_merged(before)
        print("OK: file already merged and consistent")
        sys.exit(0)
    after, added = merge(before)
    check(before, after)
    SNAPSHOT.write_text(json.dumps(after, indent=1, sort_keys=True) + "\n")
    print(f"Added {added} new score values across {len(RUN_IDS)} methods.")
