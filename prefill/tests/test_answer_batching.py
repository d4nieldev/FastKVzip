"""Exercise question batching through real scorer gradients and training checkpoints."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from test_graph_answer_train_cli import _argv, _scorer, _trainer


class ToyTeacher:
    name = "Qwen/unit"
    device = torch.device("cpu")
    dtype = torch.float64
    tokenizer = object()
    config = SimpleNamespace(
        num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1, hidden_size=2
    )

    def __init__(self):
        self.model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.sys_prompt_ids = torch.tensor([[9]])

    def apply_template(self, _query):
        return torch.tensor([[1, 1]])

    def encode(self, answer):
        return torch.tensor([[int(token) for token in answer.split()]])

    def __call__(self, input_ids, cache, **_kwargs):
        signal = cache.value_cache[0].sum() / 4 + self.model.weight.sum() * 0
        logits = signal * signal.new_tensor([-0.4, 0.1, 0.6, -0.2])
        return SimpleNamespace(logits=logits.view(1, 1, 4).expand(1, input_ids.size(1), 4))


def toy_cache(index, *, context_tokens=None):
    # Consume global torch RNG so stop/resume must restore it correctly.
    tokens = 4 + index % 2 if context_tokens is None else context_tokens
    return SimpleNamespace(
        start_idx=1, end_idx=tokens + 1, ctx_len=tokens,
        key_cache=[torch.randn(1, 1, tokens + 1, 2, dtype=torch.float64)],
        value_cache=[torch.rand(1, 1, tokens + 1, 2, dtype=torch.float64) + 0.2],
        hidden_cache=[torch.randn(1, tokens + 1, 2, dtype=torch.float64)],
        prefill_ids=torch.tensor([[9] + list(range(tokens))]),
        ctx_ids=torch.arange(tokens).view(1, -1), _seen_tokens=tokens + 1,
    )


@pytest.fixture
def batch_run(tmp_path, monkeypatch):
    module = _trainer()
    active = {}
    train_example = module.train_answer_example
    save_checkpoint = module.save_checkpoint

    class TrainingInterrupted(Exception):
        pass

    def observe_example(wrapper, index, **kwargs):
        result = train_example(wrapper, index, **kwargs)
        active["examples"].append((index, kwargs["ratio"], result))
        return result

    def observe_save(output_dir, kind, **kwargs):
        active["saves"].append((kind, copy.deepcopy(kwargs["data_cursor"])))
        path = save_checkpoint(output_dir, kind, **kwargs)
        if kind == "last" and kwargs["data_cursor"]["optimizer_step"] == active["stop_after_update"]:
            raise TrainingInterrupted
        return path

    monkeypatch.setattr(module, "train_answer_example", observe_example)
    monkeypatch.setattr(module, "save_checkpoint", observe_save)

    class Wrapper:
        def __init__(self, _name, dataset, model):
            self.dataset, self.model = dataset, model

        def prefill_context(self, index, **_kwargs):
            active["prefills"].append(index)
            return toy_cache(index)

    class Progress:
        def __init__(self, **kwargs):
            active["progress"] = {"total": kwargs["total"], "value": kwargs["initial"]}

        def update(self, count):
            active["progress"]["value"] += count

        def set_postfix(self, _value):
            pass

        def set_description(self, _value):
            pass

        def close(self):
            pass

    wandb_steps = {}

    def run(name, *flags, stop_after_update=None):
        active.clear()
        active.update(examples=[], prefills=[], saves=[], logs=[], axes=[], stop_after_update=stop_after_update)
        config = SimpleNamespace(update=lambda value, **_kwargs: active.update(config=dict(value)))
        wandb_run = SimpleNamespace(
            id="batch-test", config=config,
            step=wandb_steps.get(name, 0),
            define_metric=lambda name, **kwargs: active["axes"].append((name, kwargs)),
            finish=lambda exit_code: active.update(exit_code=exit_code),
        )

        def log(metrics, *, step, commit):
            assert commit is True
            assert step >= wandb_run.step, "W&B rejects logging to past steps"
            active["logs"].append((dict(metrics), step))
            wandb_run.step = step + 1
            wandb_steps[name] = wandb_run.step

        wandb_run.log = log

        def teacher_factory(*_args, **_kwargs):
            teacher = ToyTeacher()
            active["teacher"] = teacher
            active["llm_before"] = copy.deepcopy(teacher.model.state_dict())
            return teacher

        def dataset_loader(_name, _tokenizer, **kwargs):
            return [
                {"context": f"document-{index // 3}", "question": [f"q{index}"],
                 "answers": ["2" if index % 2 == 0 else "3 1 2"]}
                for index in range(kwargs["count"])
            ]

        path = tmp_path / name
        args = module.build_parser().parse_args(_argv(
            "--model", "Qwen/unit", "--output-dir", str(path),
            "--epochs", "2", "--train-context-count", "6", "--seed", "7",
            "--gate-dim", "1", "--gate-sink", "1", "--graph-dim", "2",
            "--subgraph-size", "2", "--token-microbatch-size", "2",
            "--retention-scheduler", "uniform", "--retention-min", "0.25",
            "--retention-max", "0.75",
            "--gate-lr-scheduler", "LinearWarmupCosineLR",
            "--gate-lr-scheduler-kwargs", '{"warmup_fraction": 0.2}',
            "--mixer-lr-scheduler", "LinearWarmupCosineLR",
            "--mixer-lr-scheduler-kwargs", '{"warmup_fraction": 0.2}',
            "--wandb-mode", "disabled", *flags,
        ))
        interrupted = False
        try:
            checkpoint = module.run_training(
                args, model_factory=teacher_factory, dataset_loader=dataset_loader,
                splits_loader=lambda _name: frozenset({"train", "test"}),
                wrapper_factory=Wrapper, wandb_module=SimpleNamespace(init=lambda **_kw: wandb_run),
                progress_factory=Progress,
            )
        except TrainingInterrupted:
            interrupted = True
            checkpoint = path / "last.pt"
        assert interrupted == (stop_after_update is not None)
        result = SimpleNamespace(**active.copy(), path=checkpoint)
        result.payload = torch.load(checkpoint, weights_only=False)
        assert result.exit_code == int(interrupted)
        if not interrupted:
            assert result.progress["value"] == result.progress["total"]
        assert_nested_equal(result.teacher.model.state_dict(), result.llm_before)
        assert all(p.grad is None for p in result.teacher.model.parameters())
        return result

    return run


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def train_logs(run):
    return [metrics for metrics, _step in run.logs if "train/answer_nll" in metrics]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("token_microbatch_size", [2, 16])
def test_accumulated_update_matches_mean_question_losses_with_unequal_answers(
    batch_size, token_microbatch_size
):
    module = _trainer()
    torch.manual_seed(41)
    actual, teacher = _scorer(), ToyTeacher()
    direct = copy.deepcopy(actual)
    module.freeze_llm(teacher.model)
    llm_before = copy.deepcopy(teacher.model.state_dict())
    rows = [
        {"question": ["short"], "answers": ["2"]},
        {"question": ["long"], "answers": ["3 1 2"]},
    ]
    # Eight complete subgraphs, a partial group, and a short final subgraph.
    caches = [toy_cache(index, context_tokens=20 + index) for index in range(2)]
    wrapper = SimpleNamespace(
        dataset=rows, model=teacher, prefill_context=lambda index, **_kw: caches[index]
    )
    options = module.resolve_options(module.build_parser().parse_args(_argv(
        "--model", "Qwen/unit", "--token-microbatch-size", str(token_microbatch_size),
        "--subgraph-size", "2"
    )))
    optimizers = [torch.optim.SGD(s.gates.parameters(), lr=0.01) for s in (actual, direct)]
    mixers = [torch.optim.SGD(s.mixer.parameters(), lr=0.02) for s in (actual, direct)]
    schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.5)
                  for o in (optimizers[0], mixers[0])]

    losses, score_references = [], []
    for index in range(batch_size):
        full, hidden, ids, answer_start, _prefix = module._prepare_answer(wrapper, index, options, None)
        # Keep the reference independent of the new subgraph-packing helper.
        scores = torch.cat([
            direct(torch.stack([layer[:, start:start + 2] for layer in hidden]),
                   microbatch_size=1, token_microbatch_size=2)
            for start in range(0, full.ctx_len, 2)
        ], dim=-1)
        scores.retain_grad()
        compacted = module.compact_context_kv(
            full.key_cache, full.value_cache, scores, ratio=0.5,
            temperature=options.ste_temperature, context_range=(full.start_idx, full.end_idx),
        )
        logits = teacher(ids, module.install_compacted_cache(full, compacted)).logits
        losses.append(module.answer_objective(logits, ids, answer_start=answer_start).loss)
        score_references.append((scores, compacted.indices))
    torch.stack(losses).mean().backward()

    before = copy.deepcopy(actual.state_dict())
    results = [module.train_answer_example(
        wrapper, index, scorer=actual, options=options, ratio=0.5, expected_prefix=None
    ) for index in range(batch_size)]
    assert [r.answer_tokens for r in results] == [1, 3][:batch_size]
    for result, (scores, indices) in zip(results, score_references):
        # The reference backward averaged questions; diagnostics are per question.
        gradient = (scores.grad * batch_size).float()
        retained = torch.zeros_like(gradient, dtype=torch.bool).scatter_(-1, indices, True)
        assert (result.score_grad_norm, result.retained_score_grad_norm,
                result.evicted_score_grad_norm) == pytest.approx(
            tuple(values.norm().item() for values in
                  (gradient, gradient[retained], gradient[~retained]))
        )
    assert_nested_equal(actual.state_dict(), before)
    batch = module.finish_answer_batch(
        results, scorer=actual, gate_optimizer=optimizers[0], mixer_optimizer=mixers[0],
        gate_scheduler=schedulers[0], mixer_scheduler=schedulers[1],
    )
    for a, b in zip(actual.parameters(), direct.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)
    assert (batch.grad_norm, batch.gate_grad_norm, batch.mixer_grad_norm) == pytest.approx(
        module._gradient_norms(direct)
    )
    optimizers[1].step()
    mixers[1].step()
    for a, b in zip(actual.parameters(), direct.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    assert batch.answer_nll == pytest.approx(torch.stack(losses).mean().item())
    assert batch.answer_token_accuracy == pytest.approx(
        sum(r.answer_token_accuracy for r in results) / batch_size
    )
    assert optimizers[0].param_groups[0]["lr"] == 0.005
    assert mixers[0].param_groups[0]["lr"] == 0.01
    assert all(s.last_epoch == 1 for s in schedulers)
    assert_nested_equal(teacher.model.state_dict(), llm_before)
    assert all(parameter.grad is None for parameter in teacher.model.parameters())


@pytest.mark.parametrize("accumulation,windows", [(1, [1] * 10), (2, [2, 2, 1] * 2), (8, [5, 5])])
@pytest.mark.parametrize("token_microbatch_size", [2, 16])
def test_driver_batches_metrics_schedulers_and_epoch_cadence(
    batch_run, accumulation, windows, token_microbatch_size
):
    run = batch_run("batches", "--gradient-accumulation-steps", str(accumulation),
                    "--token-microbatch-size", str(token_microbatch_size))
    logs = train_logs(run)
    assert [m["train/batch_examples"] for m in logs] == windows
    assert [m["train/examples"] for m in logs] == list(np.cumsum(windows))
    assert [m["train/optimizer_step"] for m in logs] == list(range(1, len(windows) + 1))
    assert [step for _metrics, step in run.logs] == list(range(1, len(windows) + 1))
    assert dict(run.axes) == {
        key: {"overwrite": True}
        for key in {"*", *_trainer().TRAIN_LOG_KEYS, *_trainer().VALIDATION_LOG_KEYS}
    }
    assert run.config["prefill_chunk"] == run.payload["prefill_chunk"]
    assert run.config["token_microbatch_size"] == token_microbatch_size
    assert "global_step" not in run.payload["data_cursor"]
    assert "wandb_step" not in run.payload["data_cursor"]
    assert _trainer().processed_examples(run.payload["data_cursor"], 5) == 10
    assert run.payload["data_cursor"]["optimizer_step"] == len(windows)
    assert run.payload["config"]["retention_horizon"] == len(windows)
    assert run.payload["data_cursor"]["retention_horizon"] == len(windows)
    for name in ("gate", "mixer"):
        assert run.payload[f"{name}_scheduler"]["last_epoch"] == len(windows)
        assert all(int(s["step"]) == len(windows)
                   for s in run.payload[f"{name}_optimizer"]["state"].values())
    assert [(c["epoch"], c["offset"]) for kind, c in run.saves if kind == "last"] == [(1, 0), (2, 0)]
    offset = 0
    for log, size in zip(logs, windows):
        results = [r for _index, _ratio, r in run.examples[offset:offset + size]]
        for key in ("answer_nll", "answer_token_accuracy", "score_grad_norm",
                    "retained_score_grad_norm", "evicted_score_grad_norm"):
            assert log[f"train/{key}"] == pytest.approx(sum(getattr(r, key) for r in results) / size)
        offset += size


def test_global_question_shuffle_preserves_membership_and_retention_stream(batch_run):
    shuffled = batch_run("shuffle", "--gradient-accumulation-steps", "2")
    ordered = batch_run("ordered", "--gradient-accumulation-steps", "2", "--no-shuffle-data")
    expected = []
    for epoch in range(2):
        order = list(range(5))
        random.Random(7 + epoch).shuffle(order)
        expected.extend(order)
    assert [i for i, _ratio, _result in shuffled.examples] == expected
    assert expected[:5] != expected[5:]
    assert [i for i, _ratio, _result in ordered.examples] == list(range(5)) * 2
    assert shuffled.prefills.count(5) == ordered.prefills.count(5) == 2
    assert [r for _i, r, _result in shuffled.examples] == [r for _i, r, _result in ordered.examples]


@pytest.mark.parametrize("schedule", ["uniform", "linear"])
@pytest.mark.parametrize("update,examples", [(1, 2), (2, 4), (3, 5), (4, 7), (5, 9)])
@pytest.mark.parametrize("token_microbatch_size", [2, 16])
def test_checkpoint_resume_matches_uninterrupted_training(
    batch_run, schedule, update, examples, token_microbatch_size
):
    flags = ("--gradient-accumulation-steps", "2", "--retention-scheduler", schedule,
             "--save-strategy", "steps", "--save-every", "1",
             "--token-microbatch-size", str(token_microbatch_size))
    whole = batch_run("whole", *flags)
    first = batch_run("resume", *flags, stop_after_update=update)
    processed = _trainer().processed_examples(first.payload["data_cursor"], 5)
    assert processed == examples
    assert [m["train/batch_examples"] for m in train_logs(first)] in (
        [2], [2, 2], [2, 2, 1], [2, 2, 1, 2], [2, 2, 1, 2, 2]
    )
    resumed = batch_run("resume", *flags, "--resume", str(first.path))
    assert [(i, r) for i, r, _result in first.examples + resumed.examples] == [
        (i, r) for i, r, _result in whole.examples
    ]
    assert first.logs + resumed.logs == whole.logs
    assert first.prefills + resumed.prefills == whole.prefills
    assert [step for metrics, step in whole.logs if "validation/answer_nll" in metrics] == [3, 6]
    assert whole.prefills.count(5) == 2
    assert_nested_equal(resumed.payload, whole.payload)
    if schedule == "linear":
        assert [r for _i, r, _result in whole.examples] == pytest.approx(
            [0.75, 0.75, 0.65, 0.65, 0.55, 0.45, 0.45, 0.35, 0.35, 0.25]
        )


def test_step_cadence_uses_updates_and_never_saves_a_partial_batch(batch_run):
    run = batch_run(
        "cadence", "--gradient-accumulation-steps", "2",
        "--eval-strategy", "steps", "--eval-every", "2",
        "--save-strategy", "steps", "--save-every", "2",
    )
    assert [step for metrics, step in run.logs if "validation/answer_nll" in metrics] == [2, 4, 6]
    assert [(c["optimizer_step"], _trainer().processed_examples(c, 5)) for kind, c in run.saves if kind == "last"] == [
        (2, 4), (4, 7), (6, 10)
    ]


def test_epoch_end_flushes_two_remaining_questions(batch_run):
    run = batch_run(
        "partial", "--train-context-count", "12", "--epochs", "1",
        "--gradient-accumulation-steps", "8",
    )  # Ten training questions after the fallback validation holdout.
    assert len(run.examples) == run.progress["total"] == 10
    assert [m["train/batch_examples"] for m in train_logs(run)] == [8, 2]
    assert [m["train/examples"] for m in train_logs(run)] == [8, 10]
    assert [s for m, s in run.logs if "validation/answer_nll" in m] == [2]
    assert all(c["epoch"] == 1 and c["offset"] == 0 and c["optimizer_step"] == 2
               for _kind, c in run.saves)
    last = train_logs(run)[-1]
    for key in ("answer_nll", "answer_token_accuracy", "score_grad_norm"):
        assert last[f"train/{key}"] == pytest.approx(
            sum(getattr(result, key) for _i, _r, result in run.examples[-2:]) / 2
        )
    for name in ("gate", "mixer"):
        assert run.payload[f"{name}_scheduler"]["last_epoch"] == 2
        assert all(int(state["step"]) == 2
                   for state in run.payload[f"{name}_optimizer"]["state"].values())


def test_epoch_cadence_and_plateau_scheduler_follow_validation(batch_run):
    run = batch_run(
        "epochs", "--gradient-accumulation-steps", "2",
        "--eval-every", "2", "--save-every", "2",
        "--gate-lr-scheduler", "ReduceLROnPlateau",
        "--gate-lr-scheduler-kwargs", '{"patience": 0}',
    )
    assert [step for metrics, step in run.logs if "validation/answer_nll" in metrics] == [6]
    assert [(c["epoch"], c["optimizer_step"]) for kind, c in run.saves if kind == "last"] == [(2, 6)]
    assert run.payload["gate_scheduler"]["last_epoch"] == 1
    assert run.payload["mixer_scheduler"]["last_epoch"] == 6


def test_legacy_resume_normalizes_only_new_fields_and_keeps_original_order(batch_run):
    flags = ("--no-shuffle-data", "--save-strategy", "steps", "--save-every", "1")
    whole = batch_run("whole", *flags)
    first = batch_run("legacy", *flags, stop_after_update=3)
    old = copy.deepcopy(first.payload)
    del old["config"]["gradient_accumulation_steps"]
    del old["config"]["shuffle_data"]
    old["data_cursor"]["global_step"] = old["data_cursor"].pop("optimizer_step")
    old["data_cursor"]["wandb_step"] = 100  # Legacy logging rows are not a training clock.
    torch.save(old, first.path)
    legacy_bytes = first.path.read_bytes()
    module = _trainer()
    args = module.build_parser().parse_args(_argv("--resume", str(first.path)))
    options = module.resolve_options(args, old)
    assert options.gradient_accumulation_steps == 1
    assert options.shuffle_data is False
    assert "shuffle_data" not in old["config"]
    assert first.path.read_bytes() == legacy_bytes
    for flags in (("--shuffle-data",), ("--gradient-accumulation-steps", "2")):
        with pytest.raises(ValueError, match="conflicts"):
            module.resolve_options(module.build_parser().parse_args(
                _argv("--resume", str(first.path), *flags)
            ), old)
    resumed = batch_run("legacy", "--resume", str(first.path))
    assert first.logs + resumed.logs == whole.logs
    assert_nested_equal(resumed.payload, whole.payload)


def test_new_checkpoint_controls_are_inherited_and_weights_only_can_change_them(batch_run):
    first = batch_run("source", "--gradient-accumulation-steps", "2",
                      "--save-strategy", "steps", "--save-every", "1", stop_after_update=1)
    module = _trainer()
    for flags in (("--no-shuffle-data",), ("--gradient-accumulation-steps", "3")):
        with pytest.raises(ValueError, match="conflicts"):
            module.resolve_options(module.build_parser().parse_args(
                _argv("--resume", str(first.path), *flags)
            ), first.payload)
    resumed = batch_run("source", "--resume", str(first.path))
    assert resumed.config["gradient_accumulation_steps"] == 2
    assert resumed.config["shuffle_data"] is True
    assert train_logs(resumed)[0]["train/optimizer_step"] == 2
    fresh = batch_run("new", "--graph-checkpoint", str(first.path),
                      "--gradient-accumulation-steps", "3")
    assert fresh.config["gradient_accumulation_steps"] == 3
    assert fresh.config["shuffle_data"] is True
    assert train_logs(fresh)[0]["train/optimizer_step"] == 1


@pytest.mark.parametrize("value", ["0", "-2"])
def test_invalid_accumulation_is_rejected(value):
    module = _trainer()
    with pytest.raises(ValueError, match="gradient-accumulation-steps"):
        module.resolve_options(module.build_parser().parse_args(_argv(
            "--model", "Qwen/unit", "--gradient-accumulation-steps", value
        )))


def test_single_update_cosine_horizon_is_rejected(batch_run):
    with pytest.raises(ValueError, match="at least two total steps"):
        batch_run("invalid", "--epochs", "1", "--gradient-accumulation-steps", "8")


def test_removed_max_contexts_flag_is_rejected():
    module = _trainer()
    with pytest.raises(SystemExit) as error:
        module.build_parser().parse_args(_argv("--model", "Qwen/unit", "--max-contexts", "1"))
    assert error.value.code == 2
