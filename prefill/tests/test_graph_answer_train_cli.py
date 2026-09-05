import copy
import importlib
import math
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from attention.kvcache import RetainCache
from graph import ImplicitGraphScorer, save_checkpoint
from graph.evaluation import load_evaluation_checkpoint
from model.wrapper import ModelKVzip


def _trainer():
    try:
        return importlib.import_module("train_graph_answer")
    except ModuleNotFoundError:
        pytest.fail("prefill/train_graph_answer.py is missing")


def _argv(*extra):
    return ["--validation-retention-ratio", "0.2", *extra]


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = self.ngroup = self.output_dim = self.sink = 1
        self.d = 1.0
        self.q_proj = nn.Linear(2, 1, bias=True, dtype=torch.float64)
        self.k_proj = nn.Linear(2, 1, bias=False, dtype=torch.float64)
        self.q_norm = nn.RMSNorm(1, dtype=torch.float64)
        self.k_norm = nn.RMSNorm(1, dtype=torch.float64)
        self.k_base = nn.Parameter(torch.ones(1, 1, 1, 1, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, dtype=torch.float64))


def _scorer():
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=2,
    )
    return ImplicitGraphScorer(
        [Gate()],
        config,
        graph_dim=2,
        graph_microbatch_size=1,
        compute_dtype=torch.float64,
    )


def _base_config():
    return {
        "model_id": "Qwen/unit",
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


def test_parser_has_exact_answer_training_flags_defaults_and_forbidden_flags():
    module = _trainer()
    parser = module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    args = parser.parse_args(_argv())
    assert args.model is None
    assert args.data == "agentic"
    assert args.retention_scheduler == "linear"
    assert args.retention_min == pytest.approx(0.10)
    assert args.retention_max == pytest.approx(0.30)
    assert args.validation_retention_ratio == pytest.approx(0.2)
    assert args.ste_temperature == pytest.approx(1.0)
    assert "--resume" in option_strings
    assert "--graph-checkpoint" in option_strings
    assert "--answer-cache-dir" in option_strings
    assert "--subgraph-size" in option_strings
    assert "--generate-only" not in option_strings
    assert "--subgraphs-per-step" not in option_strings

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(_argv("--resume", "a.pt", "--graph-checkpoint", "b.pt"))
    for forbidden in ("--generate-only", "--subgraphs-per-step"):
        with pytest.raises(SystemExit):
            parser.parse_args(_argv(forbidden, "1"))


def test_options_validate_required_model_ranges_temperature_counts_and_cadence():
    module = _trainer()
    parser = module.build_parser()
    with pytest.raises(ValueError, match="--model"):
        module.resolve_options(parser.parse_args(_argv()))

    options = module.resolve_options(
        parser.parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--train-context-start",
                "3",
                "--train-context-count",
                "12",
            )
        )
    )
    assert options.model_id == "Qwen/unit"
    assert options.train_context_start == 3
    assert options.train_context_count == 12

    invalid = (
        ("--retention-min", "0"),
        ("--retention-min", "0.4", "--retention-max", "0.3"),
        ("--retention-max", "1.1"),
        ("--validation-retention-ratio", "0"),
        ("--ste-temperature", "0"),
        ("--train-context-start", "-1"),
        ("--train-context-count", "0"),
        ("--epochs", "0"),
        ("--max-contexts", "0"),
        ("--save-every", "0"),
        ("--eval-every", "0"),
    )
    for extra in invalid:
        argv = ["--model", "Qwen/unit", *_argv(*extra)]
        with pytest.raises(ValueError):
            module.resolve_options(parser.parse_args(argv))


def test_checkpoint_selects_model_architecture_and_rejects_mismatches():
    module = _trainer()
    parser = module.build_parser()
    payload = {
        "model_id": "meta-llama/unit",
        "config": {
            **_base_config(),
            "model_id": "meta-llama/unit",
            "dataset": "source-data",
            "epochs": 9,
            "train_context_start": 123,
            "train_context_count": 456,
            "seed": 88,
            "retention_scheduler": "uniform",
            "retention_min": 0.4,
            "retention_max": 0.8,
            "gate_dim": 7,
            "gate_sink": 5,
            "graph_dim": 9,
            "graph_microbatch_size": 1,
            "token_microbatch_size": 8,
            "subgraph_size": 4,
        },
        "prefill_chunk": 16,
    }
    args = parser.parse_args(
        _argv("--graph-checkpoint", "source.pt")
    )
    options = module.resolve_options(args, payload)
    assert options.model_id == "meta-llama/unit"
    assert (options.gate_dim, options.gate_sink, options.graph_dim) == (7, 5, 9)
    assert options.compute_dtype == "float64"
    assert options.subgraph_size == 4
    assert options.epochs == 1
    assert options.train_context_start == 0
    assert options.train_context_count == 29
    assert options.seed == 0
    assert options.graph_microbatch_size == "auto"
    assert options.token_microbatch_size == 1000
    assert options.prefill_chunk == 16000
    assert options.data == "agentic"
    assert options.retention_scheduler == "linear"
    assert options.retention_min == pytest.approx(0.1)
    assert options.retention_max == pytest.approx(0.3)

    with pytest.raises(ValueError, match="model"):
        module.resolve_options(
            parser.parse_args(
                _argv(
                    "--model",
                    "Qwen/other",
                    "--graph-checkpoint",
                    "source.pt",
                )
            ),
            payload,
        )
    with pytest.raises(ValueError, match="graph_dim"):
        module.resolve_options(
            parser.parse_args(
                _argv(
                    "--graph-checkpoint",
                    "source.pt",
                    "--graph-dim",
                    "10",
                )
            ),
            payload,
        )


def test_resume_inherits_saved_training_configuration_and_rejects_overrides():
    module = _trainer()
    parser = module.build_parser()
    original = module.resolve_options(
        parser.parse_args(
            [
                "--validation-retention-ratio",
                "0.25",
                "--model",
                "Qwen/unit",
                "--data",
                "unit-data",
                "--epochs",
                "3",
                "--seed",
                "9",
                "--retention-scheduler",
                "uniform",
                "--retention-min",
                "0.2",
                "--retention-max",
                "0.4",
                "--gate-lr",
                "0.002",
            ]
        )
    )
    config = module.answer_checkpoint_config(
        {
            **_base_config(),
            "gate_lr": original.gate_lr,
            "mixer_lr": original.mixer_lr,
            "adamw_eps": original.adamw_eps,
            "amsgrad": original.amsgrad,
            "gate_lr_scheduler": None,
            "mixer_lr_scheduler": None,
            "freeze_gate": False,
            "training_mode": "joint",
            "train_context_count": original.train_context_count,
            "train_context_start": original.train_context_start,
        },
        options=original,
        total_steps=60,
    )
    payload = {
        "model_id": "Qwen/unit",
        "config": config,
        "prefill_chunk": original.prefill_chunk,
    }
    resumed = module.resolve_options(
        parser.parse_args(
            ["--validation-retention-ratio", "0.25", "--resume", "last.pt"]
        ),
        payload,
    )
    assert resumed.data == "unit-data"
    assert resumed.epochs == 3
    assert resumed.seed == 9
    assert resumed.retention_scheduler == "uniform"
    assert resumed.retention_min == pytest.approx(0.2)
    assert resumed.retention_max == pytest.approx(0.4)
    assert resumed.gate_lr == pytest.approx(0.002)

    for override in (
        ("--model", "meta-llama/other"),
        ("--data", "other"),
        ("--epochs", "4"),
        ("--seed", "10"),
        ("--retention-scheduler", "linear"),
        ("--gate-lr", "0.003"),
    ):
        with pytest.raises(ValueError, match="model|conflicts"):
            module.resolve_options(
                parser.parse_args(
                    [
                        "--validation-retention-ratio",
                        "0.25",
                        "--resume",
                        "last.pt",
                        *override,
                    ]
                ),
                payload,
            )


def test_resume_preserves_step_cadence_and_plateau_scheduler_timing():
    module = _trainer()
    parser = module.build_parser()
    original = module.resolve_options(
        parser.parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--eval-strategy",
                "steps",
                "--eval-every",
                "7",
                "--save-strategy",
                "steps",
                "--save-every",
                "11",
                "--no-save-best",
                "--gate-lr-scheduler",
                "ReduceLROnPlateau",
                "--gate-lr-scheduler-kwargs",
                '{"mode":"min","factor":0.5,"patience":0}',
            )
        )
    )
    config = module.answer_checkpoint_config(
        {
            **_base_config(),
            "gate_lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "kwargs": {"mode": "min", "factor": 0.5, "patience": 0},
            },
            "mixer_lr_scheduler": None,
        },
        options=original,
        total_steps=30,
    )
    payload = {
        "model_id": "Qwen/unit",
        "config": config,
        "prefill_chunk": original.prefill_chunk,
    }

    resumed = module.resolve_options(
        parser.parse_args(_argv("--resume", "last.pt")), payload
    )
    assert (
        resumed.eval_strategy,
        resumed.eval_every,
        resumed.save_strategy,
        resumed.save_every,
        resumed.save_best,
    ) == ("steps", 7, "steps", 11, False)
    assert resumed.gate_scheduler.name == "ReduceLROnPlateau"

    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(()))], lr=1.0)
    scheduler = module.build_scheduler(optimizer, resumed.gate_scheduler)
    due_steps = []
    for step in range(1, 16):
        if module.train_graph.cadence_due(
            resumed.eval_strategy,
            resumed.eval_every,
            train_steps=step,
            completed_epoch=False,
            contexts_per_epoch=30,
        ):
            due_steps.append(step)
            module._step_plateau((scheduler,), float(step))
    assert due_steps == [7, 14]
    assert scheduler.last_epoch == 2

    for override in (
        ("--eval-strategy", "epochs"),
        ("--eval-every", "1"),
        ("--save-strategy", "epochs"),
        ("--save-every", "1"),
        ("--save-best",),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            module.resolve_options(
                parser.parse_args(_argv("--resume", "last.pt", *override)), payload
            )

    graph = module.resolve_options(
        parser.parse_args(_argv("--graph-checkpoint", "source.pt")), payload
    )
    assert (
        graph.eval_strategy,
        graph.eval_every,
        graph.save_strategy,
        graph.save_every,
        graph.save_best,
    ) == ("epochs", 1, "epochs", 1, True)


def test_data_split_uses_native_validation_or_one_deduplicated_fallback_load():
    module = _trainer()
    options = module.resolve_options(
        module.build_parser().parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--data",
                "unit",
                "--train-context-start",
                "2",
                "--train-context-count",
                "11",
            )
        )
    )
    teacher = SimpleNamespace(tokenizer=object())
    calls = []

    def native_loader(name, tokenizer, **kwargs):
        calls.append((name, tokenizer, kwargs))
        count = kwargs["count"]
        return [
            {"context": f"{kwargs['split']}-{index}", "question": ["q"]}
            for index in range(count)
        ]

    native = module.load_data_splits(
        options,
        teacher,
        dataset_loader=native_loader,
        splits_loader=lambda _name: frozenset({"train", "validation"}),
    )
    assert calls == [
        (
            "unit",
            teacher.tokenizer,
            {
                "split": "train",
                "teacher": teacher,
                "answer_cache_dir": None,
                "start": 2,
                "count": 11,
            },
        ),
        (
            "unit",
            teacher.tokenizer,
            {
                "split": "validation",
                "teacher": teacher,
                "answer_cache_dir": None,
                "start": 0,
                "count": 2,
            },
        ),
    ]
    assert native.train_indices == tuple(range(11))
    assert native.validation_indices == (0, 1)

    calls.clear()
    rows = [
        {"context": "a", "question": ["q"]},
        {"context": "b", "question": ["q"]},
        {"context": "a", "question": ["q"]},
        {"context": "c", "question": ["q"]},
        {"context": "d", "question": ["q"]},
        {"context": "e", "question": ["q"]},
        {"context": "f", "question": ["q"]},
        {"context": "g", "question": ["q"]},
        {"context": "h", "question": ["q"]},
        {"context": "i", "question": ["q"]},
        {"context": "j", "question": ["q"]},
        {"context": "k", "question": ["q"]},
    ]

    def fallback_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return rows

    fallback = module.load_data_splits(
        options,
        teacher,
        dataset_loader=fallback_loader,
        splits_loader=lambda _name: frozenset({"train", "test"}),
    )
    assert len(calls) == 1
    assert fallback.train_dataset is fallback.validation_dataset is rows
    assert fallback.train_indices == (0, 1, 3, 4, 5, 6, 7, 8, 9)
    assert fallback.validation_indices == (10, 11)


def test_global_horizon_and_uniform_rng_resume_are_flattened_and_deterministic():
    module = _trainer()
    linear = module.resolve_options(
        module.build_parser().parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--epochs",
                "2",
                "--retention-min",
                "0.1",
                "--retention-max",
                "0.6",
            )
        )
    )
    rng = random.Random(7)
    cursor = module.initial_cursor(total_steps=6, retention_rng=rng)
    ratios = []
    for _ in range(6):
        ratios.append(module.scheduled_retention_ratio(linear, cursor, rng))
        cursor, _ = module.advance_cursor(
            cursor,
            token_count=3,
            contexts_per_epoch=3,
            retention_rng=rng,
        )
    assert ratios == pytest.approx([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert cursor["global_step"] == cursor["retention_horizon"] == 6
    assert cursor["epoch"] == 2

    uniform = copy.copy(linear)
    object.__setattr__(uniform, "retention_scheduler", "uniform")
    first_rng = random.Random(19)
    first_cursor = module.initial_cursor(total_steps=6, retention_rng=first_rng)
    module.scheduled_retention_ratio(uniform, first_cursor, first_rng)
    first_cursor, _ = module.advance_cursor(
        first_cursor,
        token_count=1,
        contexts_per_epoch=3,
        retention_rng=first_rng,
    )
    expected = module.scheduled_retention_ratio(uniform, first_cursor, first_rng)
    resumed = random.Random()
    resumed.setstate(first_cursor["retention_rng_state"])
    assert module.scheduled_retention_ratio(uniform, first_cursor, resumed) == expected


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr=lr)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def test_resume_restores_full_state_while_graph_checkpoint_restores_only_weights(
    tmp_path,
):
    module = _trainer()
    source = _scorer()
    source_gate = torch.optim.AdamW(source.gates.parameters(), lr=0.01)
    source_mixer = torch.optim.AdamW(source.mixer.parameters(), lr=0.02)
    source_gate_scheduler = torch.optim.lr_scheduler.StepLR(source_gate, step_size=1)
    source_mixer_scheduler = torch.optim.lr_scheduler.StepLR(source_mixer, step_size=1)
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_gate.step()
    source_mixer.step()
    source_gate_scheduler.step()
    source_mixer_scheduler.step()
    retention_rng = random.Random(31)
    retention_rng.random()
    cursor = module.initial_cursor(total_steps=6, retention_rng=retention_rng)
    cursor["global_step"] = 2
    cursor["offset"] = 2
    cursor["retention_rng_state"] = retention_rng.getstate()
    random.seed(101)
    payload_path = save_checkpoint(
        tmp_path,
        "last",
        scorer=source,
        config=_base_config(),
        model_id="Qwen/unit",
        prefix_ids=torch.tensor([[91, 92]], dtype=torch.long),
        prefill_chunk=4,
        data_cursor=cursor,
        wandb_run_id="run-1",
        gate_optimizer=source_gate,
        mixer_optimizer=source_mixer,
        gate_scheduler=source_gate_scheduler,
        mixer_scheduler=source_mixer_scheduler,
    )
    payload = torch.load(payload_path, weights_only=False)
    expected_python = random.random()

    resumed_scorer = _scorer()
    resumed_gate = torch.optim.AdamW(resumed_scorer.gates.parameters(), lr=0.01)
    resumed_mixer = torch.optim.AdamW(resumed_scorer.mixer.parameters(), lr=0.02)
    resumed_gate_scheduler = torch.optim.lr_scheduler.StepLR(resumed_gate, step_size=1)
    resumed_mixer_scheduler = torch.optim.lr_scheduler.StepLR(resumed_mixer, step_size=1)
    resumed = module.restore_training_state(
        payload,
        initialization="resume",
        scorer=resumed_scorer,
        gate_optimizer=resumed_gate,
        mixer_optimizer=resumed_mixer,
        gate_scheduler=resumed_gate_scheduler,
        mixer_scheduler=resumed_mixer_scheduler,
        total_steps=6,
        seed=999,
    )
    assert random.random() == expected_python
    assert resumed.cursor == cursor
    assert torch.equal(resumed.prefix_ids, torch.tensor([[91, 92]]))
    assert resumed_gate.state_dict()["state"]
    assert resumed_mixer.state_dict()["state"]
    assert resumed_gate_scheduler.last_epoch == source_gate_scheduler.last_epoch
    expected_uniform = random.Random()
    expected_uniform.setstate(cursor["retention_rng_state"])
    assert resumed.retention_rng.random() == expected_uniform.random()

    graph_scorer = _scorer()
    graph_gate = torch.optim.AdamW(graph_scorer.gates.parameters(), lr=0.01)
    graph_mixer = torch.optim.AdamW(graph_scorer.mixer.parameters(), lr=0.02)
    random.seed(303)
    before_rng = random.getstate()
    graph = module.restore_training_state(
        payload,
        initialization="graph-checkpoint",
        scorer=graph_scorer,
        gate_optimizer=graph_gate,
        mixer_optimizer=graph_mixer,
        gate_scheduler=None,
        mixer_scheduler=None,
        total_steps=6,
        seed=17,
    )
    assert random.getstate() == before_rng
    assert graph_gate.state_dict()["state"] == {}
    assert graph_mixer.state_dict()["state"] == {}
    assert graph.cursor["global_step"] == 0
    assert graph.cursor["retention_horizon"] == 6
    for expected, actual in zip(source.parameters(), graph_scorer.parameters()):
        torch.testing.assert_close(actual, expected)


def test_one_answer_step_uses_answer_only_loss_and_updates_only_gate_and_mixer():
    module = _trainer()
    torch.manual_seed(41)
    scorer = _scorer()
    gate_optimizer = CountingSGD(scorer.gates.parameters(), lr=0.01)
    mixer_optimizer = CountingSGD(scorer.mixer.parameters(), lr=0.01)

    class FrozenLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    class Teacher:
        def __init__(self):
            self.model = FrozenLM()
            self.cache_metadata = None
            self.last_logits = None

        def apply_template(self, query):
            assert query == "Q: question"
            return torch.tensor([[1, 1]], dtype=torch.long)

        def encode(self, answer):
            assert answer == "teacher answer"
            return torch.tensor([[2, 3]], dtype=torch.long)

        def __call__(self, input_ids, kv, **kwargs):
            cache_position = kwargs.pop("cache_position")
            assert torch.equal(cache_position, torch.arange(5, 9))
            assert kwargs == {
                "update_cache": True,
                "return_logits": True,
                "use_cache": True,
            }
            self.cache_metadata = (
                kv._seen_tokens,
                kv.key_cache[0].size(2),
                kv.start_idx,
                kv.end_idx,
                kv.ctx_len,
                kv.prefill_ids,
            )
            signal = kv.value_cache[0].sum() + self.model.anchor * 0
            logits = signal.new_zeros((1, input_ids.size(1), 4))
            logits[0, 0, 0] = 100.0  # ignored non-answer target
            logits[0, 1, 2] = signal
            logits[0, 2, 3] = signal
            self.last_logits = logits.detach().clone()
            return SimpleNamespace(logits=logits)

    teacher = Teacher()

    class DeferredDataset:
        def __init__(self):
            self.rows = [{"question": ["question"], "answers": None}]
            self.resolved = 0

        def __getitem__(self, index):
            return self.rows[index]

        def resolve_answers(self, index, full_kv):
            assert full_kv.key_cache[0].size(2) == 5
            self.resolved += 1
            return ["teacher answer"]

    dataset = DeferredDataset()
    full_kv = SimpleNamespace(
        start_idx=1,
        end_idx=5,
        ctx_len=4,
        key_cache=[torch.randn(1, 1, 5, 2, dtype=torch.float64)],
        value_cache=[torch.rand(1, 1, 5, 2, dtype=torch.float64) + 0.5],
        hidden_cache=[torch.randn(1, 5, 2, dtype=torch.float64)],
        prefill_ids=torch.tensor([[9, 10, 11, 12, 13]], dtype=torch.long),
        ctx_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        _seen_tokens=5,
    )
    wrapper = SimpleNamespace(
        dataset=dataset,
        model=teacher,
        prefill_context=lambda index, **kwargs: full_kv,
    )
    options = module.resolve_options(
        module.build_parser().parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--token-microbatch-size",
                "2",
                "--subgraph-size",
                "2",
            )
        )
    )
    module.freeze_llm(teacher.model)
    scorer_before = [parameter.detach().clone() for parameter in scorer.parameters()]
    llm_before = [parameter.detach().clone() for parameter in teacher.model.parameters()]

    result = module.train_answer_example(
        wrapper,
        0,
        scorer=scorer,
        gate_optimizer=gate_optimizer,
        mixer_optimizer=mixer_optimizer,
        gate_scheduler=None,
        mixer_scheduler=None,
        options=options,
        ratio=0.5,
        expected_prefix=None,
    )

    expected = F.cross_entropy(
        teacher.last_logits[:, 1:3].reshape(-1, 4),
        torch.tensor([2, 3]),
    )
    assert result.answer_nll == pytest.approx(expected.item())
    assert result.answer_tokens == 2
    assert dataset.resolved == 1
    assert teacher.cache_metadata == (5, 3, 1, 5, 2, None)
    assert gate_optimizer.steps == mixer_optimizer.steps == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(scorer_before, scorer.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(llm_before, teacher.model.parameters())
    )
    assert all(parameter.grad is None for parameter in teacher.model.parameters())
    assert result.score_grad_norm > 0
    assert result.retained_score_grad_norm > 0
    assert result.evicted_score_grad_norm > 0


def test_one_use_retain_cache_is_discarded_without_logical_slice_restoration():
    module = _trainer()
    torch.manual_seed(43)

    class AppendingCausalLM(nn.Module):
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
        )

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
            self.last_cache = None

        def forward(self, input_ids, *, past_key_values, **_kwargs):
            self.last_cache = past_key_values
            signal = past_key_values.value_cache[0].sum() + self.anchor * 0
            appended = signal.new_zeros((1, 1, input_ids.size(1), 2))
            past_key_values.update(appended, appended, 0)
            logits = signal.new_zeros((1, input_ids.size(1), 4))
            logits[0, 1, 2] = signal
            logits[0, 2, 3] = signal
            return SimpleNamespace(logits=logits)

    teacher = object.__new__(ModelKVzip)
    teacher.model = AppendingCausalLM()
    teacher.apply_template = lambda _query: torch.tensor([[1, 1]])
    teacher.encode = lambda _answer: torch.tensor([[2, 3]])
    full_kv = RetainCache(teacher.model, (1, 5))
    full_kv.key_cache = [torch.randn(1, 1, 5, 2, dtype=torch.float64)]
    full_kv.value_cache = [torch.rand(1, 1, 5, 2, dtype=torch.float64) + 0.5]
    full_kv.hidden_cache = [torch.randn(1, 5, 2, dtype=torch.float64)]
    full_kv.prefill_ids = torch.tensor([[9, 10, 11, 12, 13]])
    full_kv.ctx_ids = torch.tensor([[10, 11, 12, 13]])
    full_kv._seen_tokens = 5
    wrapper = SimpleNamespace(
        dataset=[{"question": ["question"], "answers": ["teacher answer"]}],
        model=teacher,
        prefill_context=lambda _index, **_kwargs: full_kv,
    )
    options = module.resolve_options(
        module.build_parser().parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--token-microbatch-size",
                "2",
                "--subgraph-size",
                "2",
            )
        )
    )
    scorer = _scorer()
    module.freeze_llm(teacher.model)

    module.train_answer_example(
        wrapper,
        0,
        scorer=scorer,
        gate_optimizer=CountingSGD(scorer.gates.parameters(), lr=0.01),
        mixer_optimizer=CountingSGD(scorer.mixer.parameters(), lr=0.01),
        gate_scheduler=None,
        mixer_scheduler=None,
        options=options,
        ratio=0.5,
        expected_prefix=None,
    )

    assert isinstance(teacher.model.last_cache, RetainCache)
    assert teacher.model.last_cache.key_cache[0].size(2) == 7
    assert teacher.model.last_cache._seen_tokens == 9


def test_validation_is_token_weighted_and_best_is_selected_by_nll():
    module = _trainer()
    results = [
        module.AnswerStepResult.validation(nll_sum=2.0, correct_tokens=1, answer_tokens=2),
        module.AnswerStepResult.validation(nll_sum=9.0, correct_tokens=2, answer_tokens=3),
    ]
    aggregate = module.aggregate_validation(results)
    assert aggregate.answer_nll == pytest.approx(11 / 5)
    assert aggregate.answer_token_accuracy == pytest.approx(3 / 5)

    cursor = module.initial_cursor(total_steps=4, retention_rng=random.Random(1))
    improved, is_best = module.update_validation_cursor(cursor, aggregate)
    assert is_best
    assert improved["best_validation_nll"] == pytest.approx(11 / 5)
    worse = SimpleNamespace(answer_nll=3.0, answer_token_accuracy=1.0)
    unchanged, is_best = module.update_validation_cursor(improved, worse)
    assert not is_best
    assert unchanged["best_validation_nll"] == pytest.approx(11 / 5)


def test_wandb_metric_helpers_emit_exact_allowlist():
    module = _trainer()
    scorer = SimpleNamespace(mixer=SimpleNamespace(alpha=torch.tensor([1.0, 3.0])))
    parameter = nn.Parameter(torch.zeros(()))
    gate = torch.optim.SGD([parameter], lr=0.01)
    mixer = torch.optim.SGD([parameter], lr=0.02)
    result = module.AnswerStepResult(
        answer_nll=1.0,
        answer_nll_sum=2.0,
        answer_token_accuracy=0.5,
        correct_tokens=1,
        answer_tokens=2,
        context_tokens=10,
        grad_norm=3.0,
        gate_grad_norm=1.0,
        mixer_grad_norm=2.0,
        score_grad_norm=4.0,
        retained_score_grad_norm=5.0,
        evicted_score_grad_norm=6.0,
        prefix_ids=torch.tensor([[1]]),
    )
    train_metrics = module.train_log_metrics(
        result,
        scorer=scorer,
        gate_optimizer=gate,
        mixer_optimizer=mixer,
        fractional_epoch=0.5,
        cumulative_tokens=10,
    )
    validation_metrics = module.validation_log_metrics(
        SimpleNamespace(answer_nll=1.5, answer_token_accuracy=0.25)
    )
    assert set(train_metrics) | set(validation_metrics) == {
        "train/answer_nll",
        "train/answer_token_accuracy",
        "train/grad_norm",
        "train/gate_grad_norm",
        "train/mixer_grad_norm",
        "train/score_grad_norm",
        "train/retained_score_grad_norm",
        "train/evicted_score_grad_norm",
        "train/mean_alpha",
        "train/gate_learning_rate",
        "train/mixer_learning_rate",
        "train/epoch",
        "train/tokens",
        "validation/answer_nll",
        "validation/answer_token_accuracy",
    }


def test_answer_checkpoint_keeps_evaluator_top_level_compatibility(tmp_path):
    module = _trainer()
    scorer = _scorer()
    options = module.resolve_options(
        module.build_parser().parse_args(
            _argv(
                "--model",
                "Qwen/unit",
                "--token-microbatch-size",
                "2",
            )
        )
    )
    config = module.answer_checkpoint_config(
        _base_config(), options=options, total_steps=7
    )
    path = save_checkpoint(
        tmp_path,
        "last",
        scorer=scorer,
        config=config,
        model_id="Qwen/unit",
        prefix_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prefill_chunk=4,
        data_cursor=module.initial_cursor(
            total_steps=7, retention_rng=random.Random(0)
        ),
        wandb_run_id="run-1",
    )
    checkpoint = load_evaluation_checkpoint(path)
    assert checkpoint.model_id == "Qwen/unit"
    assert checkpoint.config["objective"] == "answer-only-causal-ce-v1"
    assert checkpoint.config["dataset"] == "agentic"
    assert checkpoint.config["retention_horizon"] == 7
    assert checkpoint.config["validation_retention_ratio"] == pytest.approx(0.2)


def test_run_training_executes_train_validation_checkpoint_and_exact_logging(
    tmp_path,
):
    module = _trainer()
    events = []

    class FrozenLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    class Teacher:
        name = "qwen-unit"
        device = torch.device("cpu")
        dtype = torch.float64
        tokenizer = object()
        config = SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=2,
        )

        def __init__(self):
            self.model = FrozenLM()
            self.sys_prompt_ids = torch.tensor([[9]], dtype=torch.long)

        def apply_template(self, _query):
            return torch.tensor([[1, 1]], dtype=torch.long)

        def encode(self, _answer):
            return torch.tensor([[2, 3]], dtype=torch.long)

        def __call__(self, input_ids, kv, **_kwargs):
            signal = kv.value_cache[0].sum() + self.model.anchor * 0
            logits = signal.new_zeros((1, input_ids.size(1), 4))
            logits[0, 1, 2] = signal
            logits[0, 2, 3] = signal
            return SimpleNamespace(logits=logits)

    teacher = Teacher()
    llm_before = [parameter.detach().clone() for parameter in teacher.model.parameters()]
    rows = [
        {"context": "train", "question": ["q"], "answers": ["a"]},
        {"context": "validation", "question": ["q"], "answers": ["a"]},
    ]
    loader_calls = []

    def dataset_loader(name, tokenizer, **kwargs):
        loader_calls.append((name, tokenizer, kwargs))
        return rows

    class Wrapper:
        def __init__(self, _name, dataset, model):
            self.dataset = dataset
            self.model = model

        def prefill_context(self, index, **kwargs):
            events.append(("prefill", index, kwargs))
            return SimpleNamespace(
                start_idx=1,
                end_idx=5,
                ctx_len=4,
                key_cache=[torch.randn(1, 1, 5, 2, dtype=torch.float64)],
                value_cache=[torch.rand(1, 1, 5, 2, dtype=torch.float64) + 0.5],
                hidden_cache=[torch.randn(1, 5, 2, dtype=torch.float64)],
                prefill_ids=torch.tensor([[9, 10, 11, 12, 13]], dtype=torch.long),
                ctx_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
                _seen_tokens=5,
            )

    class Config:
        def __init__(self):
            self.value = None

        def update(self, value, **_kwargs):
            self.value = dict(value)

    class Run:
        id = "answer-run"

        def __init__(self):
            self.config = Config()
            self.logs = []
            self.exit_code = None

        def log(self, metrics, *, step):
            self.logs.append((dict(metrics), step))

        def finish(self, exit_code=None):
            self.exit_code = exit_code

    run = Run()
    wandb = SimpleNamespace(init=lambda **_kwargs: run)

    class Progress:
        def update(self, _value):
            pass

        def set_postfix(self, _value):
            pass

        def set_description(self, _value):
            pass

        def close(self):
            pass

    args = module.build_parser().parse_args(
        _argv(
            "--model",
            "Qwen/unit",
            "--train-context-count",
            "2",
            "--max-contexts",
            "1",
            "--output-dir",
            str(tmp_path),
            "--eval-strategy",
            "steps",
            "--eval-every",
            "1",
            "--save-strategy",
            "steps",
            "--save-every",
            "1",
            "--wandb-mode",
            "disabled",
        )
    )
    path = module.run_training(
        args,
        model_factory=lambda *_a, **_k: teacher,
        dataset_loader=dataset_loader,
        splits_loader=lambda _name: frozenset({"train", "test"}),
        wrapper_factory=Wrapper,
        wandb_module=wandb,
        progress_factory=lambda **_kwargs: Progress(),
    )

    assert path == tmp_path / "last.pt"
    assert loader_calls == [
        (
            "agentic",
            teacher.tokenizer,
            {
                "split": "train",
                "start": 0,
                "count": 2,
                "teacher": teacher,
                "answer_cache_dir": None,
            },
        )
    ]
    assert [event[:2] for event in events] == [
        ("prefill", 0),
        ("prefill", 1),
    ]
    assert all(
        event[2] == {"prefill_chunk": 16000, "save_hidden": True, "do_score": False}
        for event in events
    )
    assert set().union(*(metrics for metrics, _step in run.logs)) == (
        module.TRAIN_LOG_KEYS | module.VALIDATION_LOG_KEYS
    )
    assert run.exit_code == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(llm_before, teacher.model.parameters())
    )
    payload = torch.load(path, weights_only=False)
    assert payload["data_cursor"]["global_step"] == 1
    assert payload["data_cursor"]["retention_horizon"] == 1
    assert torch.equal(payload["prefix_ids"], torch.tensor([[9]]))
    assert payload["gate_optimizer"]["state"]
    assert payload["mixer_optimizer"]["state"]


@pytest.mark.parametrize("model", ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B"])
def test_qwen_and_llama_are_accepted(model):
    module = _trainer()
    args = module.build_parser().parse_args(_argv("--model", model))
    assert module.resolve_options(args).model_id == model


def test_gemma3_is_rejected_before_wandb_model_or_streaming():
    module = _trainer()
    events = []
    args = module.build_parser().parse_args(
        _argv("--model", "google/gemma-3-4b-it")
    )
    wandb = SimpleNamespace(
        login=lambda: events.append("wandb"),
        init=lambda **kwargs: events.append("wandb") or object(),
    )

    with pytest.raises(ValueError, match="Gemma3.*not supported"):
        module.run_training(
            args,
            model_factory=lambda *_a, **_k: events.append("model"),
            dataset_loader=lambda *_a, **_k: events.append("data"),
            wandb_module=wandb,
        )
    assert events == []
