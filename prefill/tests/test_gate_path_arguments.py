"""A checkpoint path passed to -g must not be read as a released gate name."""

import pytest
import torch

from args import parse_args
from attention.gate import is_gate_path, load_fastkvzip


def test_released_gate_names_keep_their_eviction_structure_and_tag():
    for name, level in (("", "pair"), ("fastkvzip", "pair"), ("head", "pair"),
                        ("expect", "adakv-layer"), ("snap", "pair-head")):
        args = parse_args(["-g", name])
        assert args.level == level, name
        assert args.tag == (f"_{name}" if name else ""), name


@pytest.mark.parametrize(
    "path",
    [
        # Every Hugging Face cache lays checkpoints out under "snapshots", and
        # the cluster points HF_HOME at scratch, so this is the realistic case.
        "/scratch/models--Q--M/snapshots/abc123/best.pt",
        "/runs/expected-baseline/best.pt",
    ],
)
def test_a_checkpoint_path_never_selects_an_eviction_structure(path):
    assert is_gate_path(path)
    assert parse_args(["-g", path]).level == "pair"


def test_result_tags_distinguish_runs_whose_checkpoints_share_a_name():
    first = parse_args(["-g", "/runs/gate-only-seed0/best.pt"]).tag
    second = parse_args(["-g", "/runs/gate-only-seed1/best.pt"]).tag
    last = parse_args(["-g", "/runs/gate-only-seed0/last.pt"]).tag
    assert first != second and first != last
    assert first == "_gate-only-seed0-best"


def test_an_explicit_level_still_wins_over_the_default():
    assert parse_args(["-g", "/runs/x/best.pt", "--level", "pair-head"]).level == "pair-head"


def test_a_missing_gate_path_fails_loudly_instead_of_reaching_the_hub(tmp_path):
    missing = tmp_path / "absent" / "best.pt"
    with pytest.raises(FileNotFoundError, match="gate checkpoint not found"):
        load_fastkvzip("Qwen/unit", str(missing), device="cpu")


def test_a_gate_file_without_a_sink_is_rejected(tmp_path):
    """An empty sink axis makes every score 1, so eviction becomes tie-break order."""

    path = tmp_path / "sinkless.pt"
    torch.save(
        {
            "module": [
                {
                    "q_proj.weight": torch.zeros(4, 8),
                    "q_proj.bias": torch.zeros(4),
                    "k_proj.weight": torch.zeros(2, 8),
                    "q_norm.weight": torch.ones(2),
                    "k_norm.weight": torch.ones(2),
                    "k_base": torch.zeros(1, 1, 0, 2),
                    "b": torch.zeros(1, 1, 2),
                }
            ]
        },
        path,
    )
    with pytest.raises(ValueError, match="no sink"):
        load_fastkvzip("Qwen/unit", str(path), device="cpu")
