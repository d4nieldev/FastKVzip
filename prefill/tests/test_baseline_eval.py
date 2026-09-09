from types import SimpleNamespace

import pytest
import torch

from args import parse_args
from data.benchmarks import BenchmarkDataset
from eval import run_evaluation
from results.evaluation_run import EvaluationRun


@pytest.fixture(autouse=True)
def cpu_timer(monkeypatch):
    monkeypatch.setattr('utils.TimeStamp', lambda *_: lambda *_: None)


def arguments(tmp_path, **changes):
    args = parse_args([
        '--data', 'ruler_niah_single_1_4k', '--run-dir', str(tmp_path / 'run'),
        '--gate_path_or_name', '', '--window-size', '0', '--ratios', '0.5', '0.2',
    ], num_default=None)
    return SimpleNamespace(**(vars(args) | changes))


class Cache:
    ctx_len = 100

    def __init__(self, ratio=1):
        self.valid = torch.arange(100)[None, None, :] < int(ratio * 100)
        self.n_layers = self.n_heads_kv = 1
        self.sink = 7
        self.info = {'len_k': [torch.tensor([int(ratio * 100) + self.sink])]}

    def prune(self, ratio, level):
        self.valid = torch.arange(100)[None, None, :] < int(ratio * 100)
        return 0.12345, self.valid.float().mean().item()


def runtime(size=3):
    calls = []
    model = SimpleNamespace(
        tokenizer=object(), name='unit', model=SimpleNamespace(_fastkvzip_revision='revision')
    )

    def loader(name, tokenizer, n_data):
        calls.append(('load', n_data))
        return BenchmarkDataset([{}] * (size if n_data is None else min(size, n_data)), full_size=size)

    class Wrapper:
        def __init__(self, name, dataset, model, **kwargs):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def prefill_context(self, idx, **kwargs):
            calls.append(('prefill', idx, kwargs))
            return Cache(kwargs.get('chunk_ratio', 1))

        def generate_answer(self, idx, kv, *, prob, full_cache_answer):
            calls.append(('answer', idx, full_cache_answer))
            return {'qa': {'a': 'teacher' if full_cache_answer else None}}, {'qa': {}}

    class Evaluator:
        def __init__(self, model, inputs, info):
            self.inputs, self.info = inputs, info

        def decode(self, tokens):
            return tokens

        def __call__(self, kv, *, generate):
            calls.append(('evaluate',))
            return {'qa': {'pruned': 'answer', 'full__': self.inputs['qa']['a'], 'answer': ['answer']}}

    return calls, {
        'model_factory': lambda *args: model,
        'dataset_loader': loader,
        'wrapper_factory': Wrapper,
        'evaluator_factory': Evaluator,
        'generation_length_setter': lambda *args: None,
    }


@pytest.mark.parametrize('chunked', [False, True])
def test_full_benchmark_resume_and_measured_retention(tmp_path, chunked):
    args = arguments(tmp_path, gate_path_or_name='fastkvzip' if chunked else '')
    calls, factories = runtime(size=500)
    run_evaluation(args, chunked=chunked, **factories)
    assert ('load', None) in calls
    run = EvaluationRun.load(args.run_dir)
    assert len(list(run.iter_examples())) == 500
    assert run.dataset_sizes == {args.data: 500}
    assert run.manifest['prefill_mode'] == ('chunked' if chunked else 'post-prefill')
    assert run.manifest['model_identity']['gate'] == args.gate_path_or_name
    assert run.load_example(args.data, 0).payload['qa'][0][0][1] == 0.5
    assert run.load_example(args.data, 0).has_full_answers
    prefills = [call[2] for call in calls if call[0] == 'prefill']
    if chunked:
        assert prefills[:3] == [
            {'do_score': False},
            {'prefill_chunk': 16000, 'window_size': 0, 'chunk_ratio': 0.5, 'level': 'pair'},
            {'prefill_chunk': 16000, 'window_size': 0, 'chunk_ratio': 0.2, 'level': 'pair'},
        ]
    else:
        assert prefills[0] == {'window_size': 0, 'do_score': True}
    calls.clear()
    run_evaluation(arguments(tmp_path, existing_results='resume', gate_path_or_name=args.gate_path_or_name), chunked=chunked, **factories)
    assert calls == [('load', None)]


@pytest.mark.parametrize('chunked', [False, True])
def test_limit_and_partial_ratio_resume_reuses_full_answer(tmp_path, chunked):
    calls, factories = runtime(size=7)
    args = arguments(tmp_path, idx=2, num=2, ratios=[0.5])
    run_evaluation(args, chunked=chunked, **factories)
    assert ('load', 4) in calls
    before = EvaluationRun.load(args.run_dir).load_example(args.data, 2).payload['qa'][0]
    calls.clear()
    args.ratios = [0.5, 0.2]
    args.existing_results = 'resume'
    run_evaluation(args, chunked=chunked, **factories)
    assert sum(call[0] == 'evaluate' for call in calls) == 2
    assert not any(call[0] == 'answer' and call[2] for call in calls)
    example = EvaluationRun.load(args.run_dir).load_example(args.data, 2)
    assert example.payload['qa'][0] == before
    assert example.requested_ratios == (0.5, 0.2)
    assert example.full_answers == {'qa': 'teacher'}


def test_wandb_binding_and_finalizer_reuse(tmp_path):
    calls, factories = runtime(size=1)
    finalized = []
    args = arguments(tmp_path, wandb_run_id='baseline-only', log_to_wandb=True, wandb_project='project')
    run_evaluation(args, **factories, metrics_finalizer=lambda *a, **kw: finalized.append((a, kw)))
    assert EvaluationRun.load(args.run_dir).manifest['wandb_run_id'] == 'baseline-only'
    assert finalized[0][0][1:] == (args.data, 1)
    assert finalized[0][1] == {'log_to_wandb': True, 'wandb_project': 'project', 'wandb_entity': None}


def test_cli_default_preserves_legacy_consumers_and_opt_in_complete_baselines():
    assert parse_args([]).num == 100
    assert parse_args([], num_default=None).num is None


def test_evict_cache_retention_excludes_system_prefix(tmp_path):
    _, factories = runtime(size=1)
    args = arguments(tmp_path, kv_type='evict')
    run_evaluation(args, chunked=True, **factories)
    row = EvaluationRun.load(args.run_dir).load_example(args.data, 0)
    assert [entry[0][1] for entry in row.payload['qa']] == [0.5, 0.2]


@pytest.mark.parametrize('chunked', [False, True])
def test_resume_adds_missing_full_answers_without_repeating_ratios(tmp_path, chunked):
    calls, factories = runtime(size=1)
    args = arguments(tmp_path, full_cache_answer=False)
    run_evaluation(args, chunked=chunked, **factories)
    calls.clear()
    args.full_cache_answer = True
    args.existing_results = 'resume'
    run_evaluation(args, chunked=chunked, **factories)
    assert not any(call[0] == 'evaluate' for call in calls)
    assert EvaluationRun.load(args.run_dir).load_example(args.data, 0).has_full_answers
