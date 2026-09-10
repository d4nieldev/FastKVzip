import pytest
import torch

from graph.evaluation import protect_local_window
from window import resolve_window_size


@pytest.mark.parametrize("context_length", [100, 15999, 16000, 20000])
def test_explicit_zero_never_protects_tokens_or_changes_scores(context_length):
    scores = torch.arange(context_length, 0, -1, dtype=torch.float32).view(1, 1, 1, -1)
    original = scores.clone()

    assert resolve_window_size(0, context_length, prefill_chunk=16000) == 0
    assert (
        protect_local_window(
            scores,
            token_count=context_length,
            prefill_chunk=16000,
            window_size=0,
        )
        == 0
    )
    assert torch.equal(scores, original)


@pytest.mark.parametrize(
    "window_size,context_length,expected",
    [
        (4096, 100, 2),
        (4096, 15999, 319),
        (4096, 16000, 4096),
        (4096, 20000, 4096),
        (30000, 20000, 20000),
        (0.1, 100, 10),
        (0.1, 16000, 1600),
        (0.1, 20000, 2000),
    ],
)
def test_nonzero_windows_keep_existing_fallback_ratio_and_clamp_behavior(
    window_size, context_length, expected
):
    assert (
        resolve_window_size(window_size, context_length, prefill_chunk=16000)
        == expected
    )
