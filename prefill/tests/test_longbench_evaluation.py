import importlib
import json
from types import SimpleNamespace

import pytest

from args import parse_args
from data.benchmarks import BenchmarkDataset, dataset_revisions
from data.longbench import LONGBENCH_PROTOCOL, LONGBENCH_REVISION
from data.ruler import RULER_REVISIONS
from eval import run_evaluation
from results.evaluation_run import EvaluationRun
from results.parse import build_run_metrics
from test_graph_eval import _FakeCuda, _FakeProgress
from test_longbench_runtime import CPUModel, row


@pytest.mark.parametrize("runner", ["graph", "graph_chunked", "kvzip", "fastkvzip"])
def test_all_entrypoints_use_real_longbench_prompts_and_resumable_results(monkeypatch, tmp_path, runner):
    model = CPUModel()
    data_name = "longbench_repobench-p"
    rows = [row(), row(context="a second context\n", question=["\nNext line of code:\n"])]
    loader_calls = []

    def load(name, tokenizer, n_data, **kwargs):
        loader_calls.append((name, n_data))
        return BenchmarkDataset(rows if n_data is None else rows[:n_data], full_size=2)

    common = ["--data", data_name, "--run-dir", str(tmp_path / "run"),
              "--window-size", "0", "--ratios", "0.5", "0.2", "--num", "2"]
    if runner.startswith("graph"):
        module = importlib.import_module("eval_" + runner)
        checkpoint_file = tmp_path / "checkpoint.pt"
        checkpoint_file.write_bytes(b"unit checkpoint")
        checkpoint = SimpleNamespace(
            config={"num_layers": 1, "num_kv_heads": 1},
            graph_microbatch_size=1, token_microbatch_size=4, subgraph_size=None,
            prefill_chunk=16000, prefix_ids=model.encode("exact checkpoint prefix|"),
            payload={"wandb_run_id": "training-run"},
        )
        monkeypatch.setattr(module, "load_evaluation_checkpoint", lambda *a, **k: checkpoint)
        monkeypatch.setattr(module, "build_evaluation_runtime", lambda *a, **k: (model, SimpleNamespace(device=model.device)))
        score_name = "score_context_chunk_cache" if runner == "graph_chunked" else "score_context_cache"
        monkeypatch.setattr(module, score_name, lambda kv, *a, **k: kv.hidden_cache.clear())
        args = module.build_parser().parse_args(common + ["--graph-checkpoint", str(checkpoint_file)])

        def execute():
            module.run_evaluation(
                args, dataset_loader=load, cuda=_FakeCuda(),
                progress_factory=lambda **kwargs: _FakeProgress([], data_name, **kwargs),
            )

        expected_prefix = "exact checkpoint prefix|"
    else:
        monkeypatch.setattr("utils.TimeStamp", lambda *_: lambda *_: None)
        args = parse_args(common + ["--gate_path_or_name", "fastkvzip" if runner == "fastkvzip" else ""], num_default=None)

        def execute():
            run_evaluation(args, chunked=runner == "fastkvzip", dataset_loader=load, model_factory=lambda *a: model)

        expected_prefix = "chat prefix|"

    execute()
    run = EvaluationRun.load(args.run_dir)
    assert run.manifest["dataset_revisions"] == {
        "longbench": LONGBENCH_REVISION, "longbench_protocol": LONGBENCH_PROTOCOL,
    }
    assert run.manifest["prefill_mode"] == ("chunked" if runner in {"fastkvzip", "graph_chunked"} else "post-prefill")
    assert run.dataset_sizes == {data_name: 2}
    assert loader_calls == [(data_name, 2)]
    assert len(model.generations) == 6  # full cache and two retention ratios per row
    for query, cache, settings in model.generations:
        data = next(data for data in rows if data["context"] == model.decode(cache.ctx_ids))
        protected = expected_prefix + data["context_prefix"]
        assert model.decode(cache.prefill_ids) == protected + data["context"]
        assert cache.start_idx == len(protected)
        assert model.decode(query) == data["question"][0] + model.decode(model.postfix_ids)
        assert settings["max_new_tokens"] == 64
        assert settings["eos_token_id"] == [2, 3]
    assert model.decode(model.sys_prompt_ids) == expected_prefix

    for index, data in enumerate(rows):
        example = run.load_example(data_name, index)
        assert example.requested_ratios == (0.5, 0.2)
        assert example.has_full_answers
        for _, output in example.payload["qa"]:
            assert output == {"pruned": "reference", "full__": "reference", "answer": data["answers"][0]}
    online_metrics = json.loads(run.metrics_path.read_text())
    offline_metrics = build_run_metrics(run, dataset_sizes=run.dataset_sizes)
    assert online_metrics == offline_metrics
    metrics = online_metrics["tasks"][data_name]
    assert metrics["complete"]
    assert set(metrics["ratios"]) == {"0.2", "0.5", "1.0"}
    assert all(result["score"] == 100 for result in metrics["ratios"].values())

    before = len(model.prefills), len(model.generations)
    args.existing_results = "resume"
    execute()
    assert (len(model.prefills), len(model.generations)) == before
    assert json.loads(run.metrics_path.read_text()) == online_metrics


def test_longbench_revisions_do_not_change_existing_dataset_identities():
    assert dataset_revisions(["squad", "gsm", "scbench_kv"]) == {}
    ruler = {"ruler_4k": RULER_REVISIONS["4k"], "ruler_8k": RULER_REVISIONS["8k"]}
    assert dataset_revisions(["ruler_niah_single_1_4k", "ruler_vt_4k", "ruler_qa_2_8k"]) == ruler
    assert dataset_revisions(["ruler_vt_4k", "longbench_qasper", "longbench_lcc"]) == {
        "ruler_4k": RULER_REVISIONS["4k"], "longbench": LONGBENCH_REVISION,
        "longbench_protocol": LONGBENCH_PROTOCOL,
    }


@pytest.mark.parametrize("key", ["longbench", "longbench_protocol"])
def test_dataset_and_protocol_revision_changes_reject_resume(tmp_path, key):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"unit checkpoint")
    revisions = dataset_revisions(["longbench_qasper"])
    kwargs = dict(checkpoint_path=checkpoint, wandb_run_id=None, window_size=0.02,
                  level="pair", dataset_revisions=revisions)
    with EvaluationRun.open(tmp_path, "run", **kwargs):
        pass
    with EvaluationRun.open(tmp_path, "run", existing_results="resume", **kwargs):
        pass
    with pytest.raises(ValueError, match="dataset_revisions"):
        EvaluationRun.open(tmp_path, "run", existing_results="resume",
                           **(kwargs | {"dataset_revisions": revisions | {key: "changed"}}))
