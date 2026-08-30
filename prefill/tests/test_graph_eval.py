import io
import math
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import eval_graph
from attention.score import KVScore
from data import DataWrapper
from graph import ImplicitGraphScorer, save_checkpoint
from graph.evaluation import (
    _clear_hidden_cache,
    load_evaluation_checkpoint,
    protect_local_window,
    score_context_cache,
    score_hidden_cache,
    score_seen_context_cache,
)
from model import ModelKVzip
from utils import Evaluator


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead, self.ngroup, self.output_dim, self.sink, self.d = 1, 1, 1, 1, 1.0
        self.q_proj = nn.Linear(2, 1)
        self.k_proj = nn.Linear(2, 1, bias=False)
        self.q_norm = _ScaleNorm(1)
        self.k_norm = _ScaleNorm(1)
        self.k_base = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1))


class _ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value):
        return value * self.weight


def _config():
    return SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=2,
    )


def _scorer():
    return ImplicitGraphScorer(
        [Gate().double()],
        _config(),
        graph_dim=2,
        graph_microbatch_size=1,
        compute_dtype=torch.float64,
    )


def _checkpoint_config():
    return {
        "model_id": "unit",
        "compute_dtype": "float64",
        "gate_dim": 1,
        "gate_sink": 1,
        "hidden_dim": 2,
        "num_layers": 1,
        "num_kv_heads": 1,
        "query_groups": 1,
        "graph_dim": 2,
        "graph_microbatch_size": 1,
        "token_microbatch_size": 2,
        "gram_normalization": "token-count",
        "leaky_relu_slope": 0.01,
        "activation_order": "batchnorm-leaky-relu",
        "alpha_init": 0.1,
    }


def test_score_hidden_cache_matches_full_scorer_across_token_chunks():
    torch.manual_seed(8)
    scorer = _scorer()
    context = torch.randn(5, 2, dtype=torch.float64)
    hidden_cache = [torch.cat((torch.zeros(1, 2, 2, dtype=torch.float64), context.unsqueeze(0)), dim=1)]
    actual = score_hidden_cache(
        scorer, hidden_cache, start_idx=2, end_idx=7, token_microbatch_size=2
    )
    expected = scorer(context.unsqueeze(0), token_microbatch_size=5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_current_checkpoint_validation_has_only_implicit_mixer_state(tmp_path):
    scorer = _scorer()
    path = save_checkpoint(
        tmp_path,
        "last",
        scorer=scorer,
        config=_checkpoint_config(),
        model_id="unit",
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prefill_chunk=4,
        data_cursor={"epoch": 0},
        wandb_run_id=None,
    )
    checkpoint = load_evaluation_checkpoint(path)
    assert checkpoint.config["gram_normalization"] == "token-count"
    payload = torch.load(path, weights_only=False)
    assert set(payload["mixer"]) == {
        "mixer.in_proj.weight",
        "mixer.out_proj.weight",
        "mixer.gamma",
        "mixer.beta",
        "mixer.alpha",
    }
    payload["mixer"].pop("mixer.alpha")
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="missing tensor"):
        load_evaluation_checkpoint(bad)
    payload = torch.load(path, weights_only=False)
    payload["config"]["activation_order"] = "leaky-relu-batchnorm"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="activation order"):
        load_evaluation_checkpoint(bad)


def test_protection_only_changes_the_context_local_window_and_hidden_cache_is_cleared():
    scores = torch.arange(5.0).view(1, 1, 1, 5)
    window = protect_local_window(scores, token_count=5, prefill_chunk=4, window_size=2)
    assert window == 2
    assert scores[..., :3].tolist() == [[[[0.0, 1.0, 2.0]]]]
    assert scores[..., -2:].eq(scores.max()).all()
    assert (
        protect_local_window(
            scores, token_count=5, prefill_chunk=5, window_size=10
        )
        == 5
    )
    kv = SimpleNamespace(hidden_cache=[torch.zeros(1)])
    _clear_hidden_cache(kv)
    assert kv.hidden_cache == []


@pytest.mark.parametrize(
    "level", ("pair", "pair-head", "pair-layer", "adakv-layer")
)
def test_protected_window_survives_when_larger_than_pruning_budget(level):
    scores = torch.arange(10.0).view(1, 1, 1, 10)
    window = protect_local_window(
        scores, token_count=10, prefill_chunk=10, window_size=4
    )
    cache = KVScore()
    cache.protected_window = window

    valid, _ = cache.threshold(scores, ratio=0.2, level=level)

    assert valid[..., -window:].all()
    assert valid.float().mean().item() >= window / scores.size(-1)


def test_score_context_cache_preserves_non_context_indices_and_releases_hidden():
    scorer = _scorer()
    kv = SimpleNamespace(
        start_idx=2,
        end_idx=5,
        ctx_len=3,
        device=scorer.device,
        hidden_cache=[torch.randn(1, 5, 2, dtype=torch.float64)],
        score=None,
    )
    scores = score_context_cache(
        kv, scorer, prefill_chunk=3, window_size=1, token_microbatch_size=2
    )
    assert scores.shape == (1, 1, 1, 3)
    assert kv.protected_window == 1
    assert kv.hidden_cache == []


def test_seen_context_scoring_excludes_prefix_and_replaces_old_scores():
    torch.manual_seed(9)
    scorer = _scorer()
    hidden = torch.randn(1, 5, 2, dtype=torch.float64)
    kv = SimpleNamespace(start_idx=2, hidden_cache=[hidden], score=None)

    first = score_seen_context_cache(
        kv, scorer, token_microbatch_size=2
    )
    assert kv.score[0].shape == (1, 1, 5)
    assert kv.score[0][..., :2].eq(0).all()
    torch.testing.assert_close(kv.score[0][..., 2:], first[0])

    kv.hidden_cache[0] = torch.cat(
        (hidden, torch.randn(1, 2, 2, dtype=torch.float64)), dim=1
    )
    second = score_seen_context_cache(
        kv, scorer, token_microbatch_size=3
    )
    assert kv.score[0].shape == (1, 1, 7)
    assert kv.score[0][..., :2].eq(0).all()
    torch.testing.assert_close(kv.score[0][..., 2:], second[0])


def test_model_prefill_uses_graph_callback_before_official_chunk_pruning():
    events = []

    class Cache:
        def __init__(self, evict_range):
            self.start_idx, self.end_idx = evict_range
            self.ctx_len = self.end_idx - self.start_idx
            self.hidden_cache = []
            self.valid = None

        def init_score(self, get_score=False):
            self.compute_gate = not get_score
            self.score = [torch.zeros(1, 1, 0)]

        def prune_chunk(self, ratio, evict_range, level):
            events.append(("prune", evict_range, ratio, level))
            valid = torch.ones(1, 1, evict_range[1] - evict_range[0], dtype=torch.bool)
            self.valid = valid if self.valid is None else torch.cat((self.valid, valid), -1)
            return 0.0, ratio

    class Model(ModelKVzip):
        def __init__(self):
            self.sys_prompt_ids = torch.tensor([[90, 91]])
            self.gates = None
            self.kv_type = "retain"

        def _init_kv(self, kv=None, evict_range=(0, 0)):
            return Cache(evict_range)

        def __call__(self, input_ids, kv, update_cache=False, **_kwargs):
            hidden = torch.ones(1, input_ids.size(1), 1)
            if kv.hidden_cache:
                kv.hidden_cache[0] = torch.cat((kv.hidden_cache[0], hidden), dim=1)
            else:
                kv.hidden_cache.append(hidden)
            events.append(("forward", kv.hidden_cache[0].size(1)))

    def score_chunk(kv):
        seen = kv.hidden_cache[0].size(1)
        events.append(("score", seen))
        kv.score = [torch.zeros(1, 1, seen)]

    kv = Model().prefill(
        torch.arange(8).view(1, -1),
        prefill_chunk_size=6,
        window_size=2,
        chunk_ratio=0.5,
        level="pair",
        chunk_scorer=score_chunk,
    )

    assert events == [
        ("forward", 6),
        ("score", 6),
        ("prune", (2, 4), pytest.approx(1 / 3), "pair"),
        ("forward", 10),
        ("score", 10),
        ("prune", (4, 8), pytest.approx(1 / 3), "pair"),
    ]
    assert kv.valid.shape[-1] == kv.ctx_len == 8
    assert kv.hidden_cache == []


def test_full_cache_answer_can_be_disabled_without_skipping_pruned_generation():
    class Model:
        name = "unit"

        def __init__(self):
            self.generated = 0

        def set_chat_template(self, _task):
            pass

        def apply_template(self, _query):
            return torch.tensor([[1]])

        def generate(self, _query, *, kv):
            self.generated += 1
            return "pruned"

        def encode(self, text):
            return torch.tensor([[2 if text == "gold" else 3]])

        def decode(self, ids):
            return "gold" if ids.item() == 2 else "pruned"

    parser = eval_graph.build_parser()
    required = [
        "--graph-checkpoint",
        "checkpoint.pt",
        "--run-dir",
        "results/test",
    ]
    assert parser.parse_args(required).full_cache_answer
    assert parser.parse_args(
        [*required, "--ratios", "0.1", "0.2", "0.3"]
    ).ratios == [0.1, 0.2, 0.3]
    maximum = parser.parse_args(
        [
            *required,
            "--token-microbatch-size",
            "full",
            "--graph-microbatch-size",
            "all",
        ]
    )
    assert maximum.token_microbatch_size == "full"
    assert maximum.graph_microbatch_size == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*required, "--ratios", "0.0"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*required, "--token-microbatch-size", "0"]
        )
    options = parser.parse_args(
        [*required, "--no-full-cache-answer"]
    )
    assert not options.full_cache_answer

    model = Model()
    dataset = DataWrapper(
        "unit",
        [{"question": ["question"], "answers": ["gold"]}],
        model,
    )
    inputs, info = dataset.generate_answer(
        0, object(), prob=False, full_cache_answer=False
    )
    assert model.generated == 0

    result = Evaluator(model, inputs, info)(object())
    assert model.generated == 1
    assert result["qa"] == {
        "pruned": "pruned",
        "full__": None,
        "answer": "gold",
    }
    with pytest.raises(ValueError, match="probabilities require"):
        dataset.generate_answer(0, object(), prob=True, full_cache_answer=False)

    model = Model()
    dataset = DataWrapper(
        "unit",
        [{"question": ["question"], "answers": ["gold"]}],
        model,
    )
    inputs, info = dataset.generate_answer(0, object(), prob=False)
    assert model.generated == 1
    result = Evaluator(model, inputs, info)(object())
    assert model.generated == 2
    assert result["qa"]["full__"] == "pruned"


def test_graph_eval_parser_supports_quiet_default_and_verbose_output():
    parser = eval_graph.build_parser()
    required = [
        "--graph-checkpoint",
        "checkpoint.pt",
        "--run-dir",
        "results/test",
    ]
    quiet = parser.parse_args(required)
    verbose = parser.parse_args([*required, "--verbose"])

    assert not quiet.verbose
    assert verbose.verbose
    assert quiet.run_dir.as_posix() == "results/test"
    assert quiet.existing_results == "fail"
    resumed = parser.parse_args(
        [
            *required,
            "--existing-results",
            "resume",
        ]
    )
    assert resumed.run_dir.as_posix() == "results/test"
    assert resumed.existing_results == "resume"

    with pytest.raises(SystemExit):
        parser.parse_args(["--graph-checkpoint", "checkpoint.pt"])


def test_wandb_requires_checkpoint_run_id_before_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eval_graph,
        "load_evaluation_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(payload={}),
    )
    monkeypatch.setattr(
        eval_graph,
        "build_evaluation_runtime",
        lambda *_args, **_kwargs: pytest.fail("runtime should not load"),
    )
    args = eval_graph.build_parser().parse_args(
        [
            "--graph-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--run-dir",
            str(tmp_path / "results" / "run"),
            "--log-to-wandb",
            "--wandb-project",
            "project",
        ]
    )

    with pytest.raises(ValueError, match="checkpoint with a W&B run ID"):
        eval_graph.run_evaluation(args)


def test_real_tqdm_keeps_quiet_progress_on_one_terminal_line(capsys):
    stream = io.StringIO()
    percentages = [[], [], []]
    eval_graph._record_phase_percentages(
        percentages, (2.8, 21.6, 8.5), 32.9
    )
    eval_graph._record_phase_percentages(
        percentages, (3.5, 14.9, 56.0), 74.4
    )
    with eval_graph._example_output(False):
        progress = eval_graph.tqdm(
            file=stream, total=1, mininterval=0, desc="[1/1] task"
        )
        progress.set_postfix(
            eval_graph._postfix(10, percentages, 1 << 30, 2 << 30)
        )
        print("noisy stdout")
        print("noisy stderr", file=sys.stderr)
        progress.update(1)
        progress.close()

    captured = capsys.readouterr()
    assert "noisy" not in captured.out + captured.err
    rendered = stream.getvalue()
    assert rendered.count("\n") == 1
    assert "[1/1] task" in rendered
    assert "1/1" in rendered
    assert "max_tokens=10" in rendered
    assert "prefill=6.6±1.9%" in rendered
    assert "mixer=42.8±22.8%" in rendered
    assert "gen=50.6±24.7%" in rendered
    assert "max_gpu=1.0/2.0GiB" in rendered
    assert "00:00<00:00" in rendered
    assert "it/s" in rendered or "s/it" in rendered


def test_run_evaluation_uses_one_in_place_progress_bar_per_task(
    monkeypatch, tmp_path, capsys
):
    run = _run_fake_evaluation(monkeypatch, tmp_path, tasks=("first", "second"))

    assert len(run.progresses) == 2
    for index, progress in enumerate(run.progresses, 1):
        assert progress.kwargs["total"] == 1
        task = ("first", "second")[index - 1]
        assert progress.kwargs["desc"] == f"[{index}/2] {task}"
        assert "bar_format" not in progress.kwargs
        assert "unit" not in progress.kwargs
        assert progress.updates == [1]
        assert progress.closed

        assert len(progress.postfixes) == 4
        prefill, mixer, generation, done = progress.postfixes
        assert "phase" not in prefill
        assert {**prefill, "max_gpu": None} == {
            "max_tokens": "--",
            "prefill": "--",
            "mixer": "--",
            "gen": "--",
            "max_gpu": None,
        }
        assert prefill["max_gpu"].endswith("/2.0GiB")
        assert mixer["max_tokens"] == 10
        assert mixer["prefill"] == "--"
        assert mixer["mixer"] == "--"
        assert mixer["gen"] == "--"
        assert generation["mixer"] == "--"
        assert generation["gen"] == "--"
        assert done == {
            "max_tokens": 10,
            "prefill": "27.3±0.0%",
            "mixer": "9.1±0.0%",
            "gen": "18.2±0.0%",
            "max_gpu": "1.0/2.0GiB",
        }
        assert progress.postfix_refresh[-1] is False

    assert run.events == [
        "update:first",
        "finalize:first",
        "update:second",
        "finalize:second",
    ]
    assert [call[0][:2] for call in run.merges] == [
        ("first", 0),
        ("second", 0),
    ]
    assert run.cuda.resets == 2
    assert run.cuda.synchronizations == 14
    assert run.cuda.peak_reads >= 4

    captured = capsys.readouterr()
    assert "prefill detail" not in captured.out
    assert "mixer detail" not in captured.out
    assert "generation detail" not in captured.out
    assert "stderr detail" not in captured.err


def test_run_evaluation_verbose_restores_per_example_output(
    monkeypatch, tmp_path, capsys
):
    _run_fake_evaluation(monkeypatch, tmp_path, verbose=True)

    captured = capsys.readouterr()
    assert "prefill detail" in captured.out
    assert "mixer detail" in captured.out
    assert "generation detail" in captured.out
    assert "stderr detail" in captured.err
    assert "Start evaluation with 0~1 samples" in captured.out
    assert "## Time: 11.0s. Task peak GPU: 1.0/2.0GiB" in captured.out
    assert "Finished." in captured.out


def test_run_evaluation_replays_quiet_diagnostics_on_failure(
    monkeypatch, tmp_path, capsys
):
    with pytest.raises(RuntimeError, match="prefill failed"):
        _run_fake_evaluation(monkeypatch, tmp_path, fail=True)

    captured = capsys.readouterr()
    assert "prefill detail" in captured.err
    assert "stderr detail" in captured.err


def test_resumable_evaluation_skips_complete_example_before_prefill(
    monkeypatch, tmp_path
):
    stored = SimpleNamespace(
        requested_ratios=(0.2,),
        has_full_answers=True,
        formats=("qa",),
    )

    run = _run_fake_evaluation(
        monkeypatch,
        tmp_path,
        resumable_result=stored,
    )

    assert run.prefills == []
    assert run.merges == []
    assert run.events == ["update:task", "finalize:task"]
    assert run.cuda.resets == 1
    assert run.progresses[0].postfixes[-1]["max_tokens"] == "--"


def test_resumable_evaluation_backfills_only_full_answer_without_mixer(
    monkeypatch, tmp_path
):
    stored = SimpleNamespace(
        requested_ratios=(0.2,),
        has_full_answers=False,
        formats=("qa",),
    )

    run = _run_fake_evaluation(
        monkeypatch,
        tmp_path,
        resumable_result=stored,
    )

    assert run.prefills == [
        {"prefill_chunk": 8, "save_hidden": False, "do_score": False}
    ]
    assert run.score_calls == []
    assert run.generate_full_flags == [True]
    assert len(run.merges) == 1
    _, merge = run.merges[0]
    assert merge["outputs"] is None
    assert merge["full_answers"] == {"qa": "full"}


def test_resumable_evaluation_generates_only_missing_ratio_and_reuses_full(
    monkeypatch, tmp_path
):
    stored = SimpleNamespace(
        requested_ratios=(0.2,),
        has_full_answers=True,
        formats=("qa",),
    )

    run = _run_fake_evaluation(
        monkeypatch,
        tmp_path,
        resumable_result=stored,
        ratios=("0.2", "0.3"),
    )

    assert len(run.prefills) == 1
    ratio_prefill = run.prefills[0]
    assert ratio_prefill["save_hidden"] is True
    assert ratio_prefill["chunk_ratio"] == 0.3
    assert callable(ratio_prefill["chunk_scorer"])
    assert run.score_calls == [True]
    assert run.generate_full_flags == [False]
    assert len(run.merges) == 1
    ratio_outputs = run.merges[0][1]["outputs"]
    assert ratio_outputs["qa"][0][0][0] == 0.3


def test_requested_ratios_are_deduplicated_before_generation(monkeypatch, tmp_path):
    run = _run_fake_evaluation(
        monkeypatch,
        tmp_path,
        ratios=("0.2", "0.2", "0.3"),
    )

    assert [
        merge[1]["outputs"]["qa"][0][0][0]
        for merge in run.merges
    ] == [0.2, 0.3]
    assert [prefill.get("chunk_ratio") for prefill in run.prefills] == [
        None,
        0.2,
        0.3,
    ]
    assert run.score_calls == [True, True]
    for _, merge in run.merges:
        assert merge["outputs"]["qa"][0][0][1:] == [0.25, 0.0]


def test_metrics_are_finalized_after_each_concrete_task(monkeypatch, tmp_path):
    run = _run_fake_evaluation(
        monkeypatch,
        tmp_path,
        tasks=("first", "second"),
    )

    assert run.events == [
        "update:first",
        "finalize:first",
        "update:second",
        "finalize:second",
    ]
    assert [call[1:3] for call in run.finalizations] == [
        ("first", 1),
        ("second", 1),
    ]


class _FakeProgress:
    def __init__(self, events, task, **kwargs):
        self.events = events
        self.task = task
        self.kwargs = kwargs
        self.postfixes = []
        self.postfix_refresh = []
        self.updates = []
        self.closed = False

    def set_postfix(self, values=None, *, refresh=True, **kwargs):
        self.postfixes.append(dict(values or kwargs))
        self.postfix_refresh.append(refresh)

    def refresh(self):
        pass

    def update(self, value):
        self.events.append(f"update:{self.task}")
        self.updates.append(value)

    def close(self):
        self.closed = True


class _FakeCuda:
    def __init__(self):
        self.resets = 0
        self.synchronizations = 0
        self.peak_reads = 0

    def is_available(self):
        return True

    def synchronize(self, *_args, **_kwargs):
        self.synchronizations += 1

    def reset_peak_memory_stats(self, *_args, **_kwargs):
        self.resets += 1

    def max_memory_allocated(self, *_args, **_kwargs):
        self.peak_reads += 1
        return 1 << 30

    def get_device_properties(self, *_args, **_kwargs):
        return SimpleNamespace(total_memory=2 << 30)


class _PhaseClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        value = float(self.calls)
        self.calls += 1
        return value


def _run_fake_evaluation(
    monkeypatch,
    tmp_path,
    *,
    tasks=("task",),
    verbose=False,
    fail=False,
    resumable_result=None,
    ratios=("0.2",),
):
    events, progresses, prefills, merges = [], [], [], []
    generate_full_flags, score_calls, finalizations = [], [], []
    cuda = _FakeCuda()
    checkpoint = SimpleNamespace(
        config={"num_layers": 1, "num_kv_heads": 1},
        graph_microbatch_size=1,
        token_microbatch_size=4,
        prefill_chunk=8,
        prefix_ids=torch.tensor([[1, 2]]),
        payload={"wandb_run_id": "training-run"},
    )
    model = SimpleNamespace(name="unit", tokenizer=object())
    scorer = SimpleNamespace(device=torch.device("cuda"))

    class Cache:
        start_idx = 2
        end_idx = 12
        ctx_len = 10
        hidden_cache = [torch.zeros(1, 12, 2)]
        valid = torch.tensor([[[True, False, False, False]]])

    class Dataset:
        def __init__(self, name):
            self.name = name
            self.dataset = [
                {
                    "context": "context",
                    "question": ["question"],
                    "answers": ["gold"],
                }
            ]

        def __len__(self):
            return 1

        def prefill_context(self, *_args, **kwargs):
            prefills.append(kwargs)
            print("prefill detail")
            print("stderr detail", file=sys.stderr)
            if fail:
                raise RuntimeError("prefill failed")
            cache = Cache()
            if "chunk_scorer" in kwargs:
                kwargs["chunk_scorer"](cache)
            return cache

        def generate_answer(self, *_args, **kwargs):
            generate_full_flags.append(kwargs["full_cache_answer"])
            print("generation detail")
            if resumable_result is not None:
                full_ids = (
                    torch.tensor([[1]]) if kwargs["full_cache_answer"] else None
                )
                return {
                    "qa": {"a": full_ids, "gt": torch.tensor([[2]])}
                }, {"qa": {}}
            return object(), object()

    class Evaluator:
        def __init__(self, inputs=None, info=None):
            self.inputs = inputs
            self.info = info

        def decode(self, ids):
            return "full" if ids.item() == 1 else "gold"

        def __call__(self, *_args, **_kwargs):
            print("generation detail")
            return {"qa": {"answer": "unchanged"}}

    def progress_factory(**kwargs):
        task = kwargs["desc"].split(" ", 1)[1]
        progress = _FakeProgress(events, task, **kwargs)
        progresses.append(progress)
        return progress

    def score_context(*_args, **_kwargs):
        score_calls.append(True)
        print("mixer detail")

    monkeypatch.setattr(
        eval_graph,
        "load_evaluation_checkpoint",
        lambda *_a, **_k: checkpoint,
    )
    monkeypatch.setattr(
        eval_graph, "build_evaluation_runtime", lambda *_a, **_k: (model, scorer)
    )
    monkeypatch.setattr(eval_graph, "get_data_list", lambda *_a, **_k: list(tasks))
    monkeypatch.setattr(
        eval_graph, "restore_checkpoint_prefix", lambda *_a, **_k: None
    )
    monkeypatch.setattr(eval_graph, "score_seen_context_cache", score_context)

    store = SimpleNamespace(
        load_example=lambda *_a, **_k: resumable_result,
        merge_example=lambda *args, **kwargs: merges.append((args, kwargs)),
    )

    class RunFactory:
        @staticmethod
        def open(*_args, **_kwargs):
            return nullcontext(store)

    monkeypatch.setattr(eval_graph, "EvaluationRun", RunFactory)

    argv = [
        "--graph-checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--ratios",
        *ratios,
        "--num",
        "1",
        "--run-dir",
        str(tmp_path / "results" / "run"),
        "--existing-results",
        "resume",
    ]
    if verbose:
        argv.append("--verbose")
    args = eval_graph.build_parser().parse_args(argv)

    def finalize(*args, **kwargs):
        finalizations.append((*args, kwargs))
        events.append(f"finalize:{args[1]}")

    eval_graph.run_evaluation(
        args,
        dataset_loader=lambda *_a, **_k: [],
        wrapper_factory=lambda name, *_a, **_k: Dataset(name),
        evaluator_factory=lambda _model, inputs, info: Evaluator(inputs, info),
        generation_length_setter=lambda *_a, **_k: None,
        progress_factory=progress_factory,
        clock=_PhaseClock(),
        cuda=cuda,
        metrics_finalizer=finalize,
    )
    return SimpleNamespace(
        events=events,
        progresses=progresses,
        cuda=cuda,
        prefills=prefills,
        merges=merges,
        generate_full_flags=generate_full_flags,
        score_calls=score_calls,
        finalizations=finalizations,
    )
