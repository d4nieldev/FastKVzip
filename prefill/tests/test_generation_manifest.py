import json

import pytest

from results.evaluation_run import EvaluationRun


SETTINGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "max_new_tokens": 768,
    "num_generations": 3,
}


def _open(tmp_path, *, existing_results="fail", generation_settings=None):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    return EvaluationRun.open(
        tmp_path,
        "run",
        checkpoint_path=checkpoint,
        wandb_run_id=None,
        window_size=0,
        level="pair-head",
        existing_results=existing_results,
        generation_settings=generation_settings,
    )


def test_default_manifest_omits_generation_settings(tmp_path):
    with _open(tmp_path) as run:
        assert "generation_settings" not in run.manifest
        assert "generation_settings" not in json.loads(run.manifest_path.read_text())


def test_generation_settings_round_trip_through_manifest_load(tmp_path):
    with _open(tmp_path, generation_settings=SETTINGS) as run:
        run_dir = run.run_dir
        assert run.manifest["generation_settings"] == SETTINGS

    assert EvaluationRun.load(run_dir).manifest["generation_settings"] == SETTINGS


def test_generation_settings_support_baseline_model_identity(tmp_path):
    with EvaluationRun.open(
        tmp_path,
        "baseline",
        checkpoint_path=None,
        model_identity={"model_id": "org/model", "revision": "abc"},
        wandb_run_id=None,
        window_size=0,
        level="pair-head",
        generation_settings=SETTINGS,
    ) as run:
        assert run.manifest["model_identity"]["model_id"] == "org/model"
        assert run.manifest["generation_settings"] == SETTINGS


def test_resume_rejects_changed_generation_settings(tmp_path):
    with _open(tmp_path, generation_settings=SETTINGS):
        pass

    changed = {**SETTINGS, "temperature": 0.8}
    with pytest.raises(ValueError, match="manifest mismatch.*generation_settings"):
        _open(
            tmp_path,
            existing_results="resume",
            generation_settings=changed,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {**SETTINGS, "top_p": 2.0},
        {key: value for key, value in SETTINGS.items() if key != "top_k"},
        {**SETTINGS, "unknown": 1},
    ],
)
def test_generation_settings_require_the_validated_five_field_shape(
    tmp_path, settings
):
    with pytest.raises(ValueError, match="generation settings"):
        _open(tmp_path, generation_settings=settings)
