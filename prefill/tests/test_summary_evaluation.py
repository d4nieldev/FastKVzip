"""Integration of dataset-driven sampling with the existing pruning entry points."""

import json
from types import SimpleNamespace

import pytest
import torch

from data.benchmarks import BenchmarkDataset
from data.wrapper import DataWrapper
from generation import GenerationSettings
from results.evaluation_run import EvaluationRun
from utils.summary_evaluation import evaluate_summary_dataset, validate_generation_args


class Tokenizer:
    name_or_path = "test/tokenizer"
    _fastkvzip_revision = "b" * 40
    model_max_length = 262144
    chat_template = "test template"

    def encode(self, text, **kwargs):
        ids = [3] * {"document a": 8192, "document b": 16384}.get(text, 5)
        return torch.tensor([ids]) if kwargs.get("return_tensors") else ids


class Cache:
    n_layers = n_heads_kv = 1
    sink = 2

    def __init__(self, size, ratio=1):
        self.ctx_len = size
        self.valid = torch.arange(size)[None, None, :] < int(size * ratio)
        self.key_cache = [torch.zeros(1)]

    def _mem(self):
        return 0

    def prune(self, ratio, level):
        self.valid = torch.arange(self.ctx_len)[None, None, :] < int(self.ctx_len * ratio)
        return 0.0, self.valid.float().mean().item()


class Model:
    name = "test-model"
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.config = SimpleNamespace(max_position_embeddings=262144, _commit_hash="a" * 40)
        self.model = SimpleNamespace(config=self.config, name_or_path="test/model",
                                     _fastkvzip_revision="a" * 40)
        self.gen_kwargs = {}
        self.gates = None
        self.calls = []
        self.fail_after = None
        self.set_chat_template("qa")

    def set_chat_template(self, task):
        self.sys_prompt_ids = torch.tensor([[1, 2]])
        self.postfix_ids = torch.tensor([[4]])

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def apply_template(self, query):
        return torch.cat([self.encode(query), self.postfix_ids], dim=1)

    def prefill(self, ids, **kwargs):
        self.calls.append(("prefill", kwargs))
        return Cache(ids.size(1), kwargs.get("chunk_ratio", 1))

    def sample_responses(self, query, kv, settings, *, sample_indices, seed, on_sample):
        self.calls.append(("sample", tuple(sample_indices), settings))
        for index in sample_indices:
            if self.fail_after == index:
                raise RuntimeError("generation interrupted")
            on_sample({"index": index, "seed": seed + index, "text": f"summary number {index}",
                       "token_count": 3, "finish_reason": "length"})


def rows():
    return BenchmarkDataset([
        {"context": context, "question": ["Summarize this document."], "answers": [""],
         "task": "summarization", "id": str(i), "n_tokens": count,
         "length_bin": band, "source": "test"}
        for i, (context, count, band) in enumerate([
            ("document a", 8192, "8k-16k"), ("document b", 16384, "16k-32k")
        ])
    ], full_size=2)


def arguments(tmp_path, **overrides):
    return SimpleNamespace(**dict(
        data="govreport_summary", idx=0, num=1, ratios=[1, .5, .5, .25],
        temperature=.7, top_p=.9, top_k=0, max_new_tokens=10, num_generations=2,
        run_dir=tmp_path / "run", existing_results="fail", answer_cache_dir=None,
    ) | overrides)


def open_run(args):
    return EvaluationRun.open(args.run_dir.parent, args.run_dir.name,
                              checkpoint_path=None, model_identity={"model_id": "test/model"},
                              wandb_run_id=None, window_size=0, level="pair",
                              existing_results=args.existing_results)


def provider(model, dataset):
    def prepare(index, ratios):
        kv = dataset.prefill_context(index)
        for ratio in ratios:
            actual = 1.0 if ratio == 1 else kv.prune(ratio, "pair")[1]
            yield ratio, kv, actual
    return prepare


def test_shared_sampling_has_one_full_pool_and_resumes_missing_samples(tmp_path):
    args = arguments(tmp_path)
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    samples = model.calls[1:]
    assert len(samples) == 3
    assert sum(len(call[1]) for call in samples) == 6
    assert all(call[2] == GenerationSettings(.7, .9, 0, 10, 2) for call in samples)
    args.existing_results = "resume"
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert model.calls == []


def test_interrupted_generation_restarts_the_incomplete_pool(tmp_path):
    args = arguments(tmp_path, ratios=[.5])
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    model.fail_after = 1
    with open_run(args) as run, pytest.raises(RuntimeError, match="interrupted"):
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    args.existing_results = "resume"
    model.fail_after = None
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert [call[1] for call in model.calls if call[0] == "sample"] == [(0, 1), (0, 1)]


@pytest.mark.parametrize("rebuilt_retention", [.5, .625])
def test_interrupted_pruned_pool_never_mixes_rebuilt_masks(tmp_path, rebuilt_retention):
    args = arguments(tmp_path, ratios=[.5])

    class InterruptedModel(Model):
        attempt = 0
        interrupt = True

        def sample_responses(self, query, kv, settings, **kwargs):
            pruned = not kv.valid.all()
            self.fail_after = 1 if pruned and self.interrupt else None
            save = kwargs["on_sample"]
            kwargs["on_sample"] = lambda sample: save({
                **sample, "text": f"mask {self.attempt}: {sample['text']}"})
            super().sample_responses(query, kv, settings, **kwargs)

    model = InterruptedModel()
    dataset = DataWrapper(args.data, rows(), model)
    masks = []

    def prepare(index, ratios):
        kv = dataset.prefill_context(index)
        for ratio in ratios:
            if ratio != 1:
                retention = .5 if model.attempt == 0 else rebuilt_retention
                kv.prune(retention, "pair")
                kv.valid = kv.valid.roll(model.attempt, dims=-1)
                masks.append(kv.valid.clone())
            yield ratio, kv, kv.valid.float().mean().item()

    path = args.run_dir / "samples/govreport_summary/examples/0.json"
    with open_run(args) as run, pytest.raises(RuntimeError, match="interrupted"):
        evaluate_summary_dataset(dataset, args, run, prepare)
    first = json.loads(path.read_text())
    args.existing_results = "resume"
    model.attempt = 1
    model.calls.clear()
    with open_run(args) as run, pytest.raises(RuntimeError, match="interrupted"):
        evaluate_summary_dataset(dataset, args, run, prepare)
    second = json.loads(path.read_text())
    assert second["ratios"]["1.0"] == first["ratios"]["1.0"]
    assert not torch.equal(masks[0], masks[1])
    assert second["ratios"]["0.5"]["samples"][0]["text"].startswith("mask 1:")
    assert second["ratios"]["0.5"]["superseded_pools"][0]["samples"] == first["ratios"]["0.5"]["samples"]
    model.attempt = 2
    model.interrupt = False
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, prepare)
    final = json.loads(path.read_text())
    pool = final["ratios"]["0.5"]
    assert [s["index"] for s in pool["samples"]] == [0, 1]
    assert all(s["text"].startswith("mask 2:") for s in pool["samples"])
    assert [p["actual_retention"] for p in pool["superseded_pools"]] == [.5, rebuilt_retention]
    assert pool["superseded_pools"][1]["samples"] == second["ratios"]["0.5"]["samples"]
    assert final["identity"] == first["identity"]
    assert final["ratios"]["1.0"] == first["ratios"]["1.0"]
    assert [call[1] for call in model.calls if call[0] == "sample"] == [(0, 1)]
    assert len([call for call in model.calls if call[0] == "prefill"]) == 1


def test_shared_partial_reference_is_archived_before_it_can_return(tmp_path):
    args = arguments(tmp_path, ratios=[.5], answer_cache_dir=tmp_path / "references")
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    model.fail_after = 1
    with open_run(args) as run, pytest.raises(RuntimeError, match="interrupted"):
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    path = args.run_dir / "samples/govreport_summary/examples/0.json"
    reference_path = next((args.answer_cache_dir / "summaries").glob("*.json"))
    previous = json.loads(path.read_text())["ratios"]["1.0"]["samples"]
    # Simulate interruption between the atomic local reset and shared-reference reset.
    data = json.loads(path.read_text())
    data["ratios"]["1.0"]["samples"] = []
    data["ratios"]["1.0"]["actual_retention"] = None
    path.write_text(json.dumps(data))
    args.existing_results = "resume"
    model.fail_after = None
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert [call[1] for call in model.calls if call[0] == "sample"] == [(0, 1), (0, 1)]
    reference = json.loads(reference_path.read_text())["ratios"]["1.0"]
    assert reference["superseded_pools"][0]["samples"] == previous
    assert len(reference["samples"]) == 2


def test_context_limit_accounts_for_prefix_query_and_output_budget(tmp_path):
    args = arguments(tmp_path, num=None)
    model = Model()
    model.config.max_position_embeddings = 8209  # 8192 + 2 + 6 + 10 exceeds this.
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert model.calls == []
    manifest = json.loads((args.run_dir / "samples/govreport_summary/manifest.json").read_text())
    assert manifest["selected_indices"] == []
    assert len(manifest["excluded"]) == 2


def test_changed_prompt_cannot_reuse_saved_samples(tmp_path):
    args = arguments(tmp_path)
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    args.existing_results = "resume"
    model.sys_prompt_ids = torch.tensor([[11, 22]])
    with open_run(args) as run, pytest.raises(ValueError, match="identity|protocol|changed"):
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))


def test_changed_runtime_config_cannot_reuse_saved_samples(tmp_path):
    args = arguments(tmp_path)
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    args.existing_results = "resume"
    model.config.rope_scaling = {"rope_type": "yarn", "factor": 4}
    with open_run(args) as run, pytest.raises(ValueError, match="identity|protocol|changed"):
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))


def test_full_reference_reused_across_pruners_but_not_decoding_changes(tmp_path):
    args = arguments(tmp_path, ratios=[.5], answer_cache_dir=tmp_path / "references")
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert len(list((args.answer_cache_dir / "summaries").glob("*.json"))) == 1
    args.run_dir = tmp_path / "other-pruner"
    args.prefill_chunk = 4096
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert len([call for call in model.calls if call[0] == "sample"]) == 1
    args.run_dir = tmp_path / "other-decoding"
    args.top_p = .8
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert len([call for call in model.calls if call[0] == "sample"]) == 2
    assert len(list((args.answer_cache_dir / "summaries").glob("*.json"))) == 2


def test_mutable_model_revision_is_rejected(tmp_path):
    args = arguments(tmp_path)
    model = Model()
    model.model._fastkvzip_revision = "main"
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run, pytest.raises(ValueError, match="immutable revisions"):
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert model.calls == []


def test_completed_run_can_populate_a_shared_reference_cache_on_resume(tmp_path):
    args = arguments(tmp_path, ratios=[.5])
    model = Model()
    dataset = DataWrapper(args.data, rows(), model)
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    args.answer_cache_dir = tmp_path / "references"
    args.existing_results = "resume"
    model.calls.clear()
    with open_run(args) as run:
        evaluate_summary_dataset(dataset, args, run, provider(model, dataset))
    assert model.calls == []
    reference_paths = list((args.answer_cache_dir / "summaries").glob("*.json"))
    assert len(reference_paths) == 1
    saved = json.loads(reference_paths[0].read_text())
    assert len(saved["ratios"]["1.0"]["samples"]) == 2


@pytest.mark.parametrize("temperature,top_k", [(0, 0), (.7, 1)])
def test_invalid_repeated_greedy_stops_before_model_work(tmp_path, temperature, top_k):
    with pytest.raises(ValueError, match="sampling|greedy"):
        validate_generation_args(arguments(tmp_path, temperature=temperature, top_k=top_k))


@pytest.mark.parametrize("pruner", ["graph", "graph_chunked", "fastkvzip", "kvzip",
                                  "fastkvzip_chunked", "kvzip_chunked"])
def test_existing_entrypoints_share_sample_pools_and_metrics(tmp_path, monkeypatch, pruner):
    import eval as baseline
    import eval_graph
    import eval_graph_chunked
    from args import parse_args

    model = Model()
    cli = ["--data", "govreport_summary", "--num", "1", "--run-dir", str(tmp_path / "run"),
           "--ratios", "1", ".75", ".5", ".3", ".2", ".5", "--temperature", ".7",
           "--top-p", ".9", "--top-k", "0", "--max-new-tokens", "10", "--num-generations", "2"]
    if pruner.startswith("graph"):
        module = eval_graph if pruner == "graph" else eval_graph_chunked
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint_path.write_bytes(b"checkpoint fixture")
        args = module.build_parser().parse_args(cli + ["--graph-checkpoint", str(checkpoint_path)])
        checkpoint = SimpleNamespace(config={"num_layers": 1, "num_kv_heads": 1},
            graph_microbatch_size=1, token_microbatch_size=4, prefill_chunk=8,
            prefix_ids=torch.tensor([[1, 2]]), payload={}, subgraph_size=None)
        monkeypatch.setattr(module, "load_evaluation_checkpoint", lambda *a, **kw: checkpoint)
        monkeypatch.setattr(module, "build_evaluation_runtime", lambda *a, **kw: (model, SimpleNamespace(device="cpu")))
        score_name = "score_context_cache" if pruner == "graph" else "score_context_chunk_cache"
        monkeypatch.setattr(module, score_name, lambda *a, **kw: None)
        cuda = SimpleNamespace(get_device_properties=lambda *_: SimpleNamespace(total_memory=1024))
        module.run_evaluation(args, dataset_loader=lambda *a, **kw: rows(), cuda=cuda)
    else:
        args = parse_args(cli + ["--gate_path_or_name", "fastkvzip" if pruner.startswith("fastkvzip") else ""])
        baseline.run_evaluation(args, model_factory=lambda *a: model,
                                dataset_loader=lambda *a, **kw: rows(),
                                chunked=pruner.endswith("_chunked"))
    samples = [call for call in model.calls if call[0] == "sample"]
    assert len(samples) == 5
    assert sum(len(call[1]) for call in samples) == 10
    assert all(call[2] == GenerationSettings(.7, .9, 0, 10, 2) for call in samples)
    prefills = [call for call in model.calls if call[0] == "prefill"]
    assert len(prefills) == (5 if pruner.endswith("_chunked") else 1)
    metrics = json.loads((args.run_dir / "metrics.json").read_text())["tasks"]["govreport_summary"]
    assert set(metrics["ratios"]) == {"1.0", "0.75", "0.5", "0.3", "0.2"}
    assert metrics["ratios"]["0.5"]["rougeL"] == 100
    assert metrics["length_bins"]["8k-16k"]["ratios"]["0.5"]["rouge1"] == 100


@pytest.mark.parametrize("module_name", ["eval", "eval_graph", "eval_graph_chunked"])
def test_cli_rejects_greedy_repetition_before_loading_checkpoint_or_model(tmp_path, monkeypatch, module_name):
    import importlib
    from args import parse_args

    module = importlib.import_module(module_name)
    argv = ["--data", "govreport_summary", "--run-dir", str(tmp_path / "run"), "--num-generations", "2"]
    if module_name != "eval":
        argv += ["--graph-checkpoint", str(tmp_path / "absent.pt")]
        args = module.build_parser().parse_args(argv)
        monkeypatch.setattr(module, "load_evaluation_checkpoint", lambda *a, **kw: pytest.fail("checkpoint loaded"))
    else:
        args = parse_args(argv)
    with pytest.raises(ValueError, match="sampling|greedy"):
        module.run_evaluation(args, model_factory=lambda *a, **kw: pytest.fail("model loaded"))
