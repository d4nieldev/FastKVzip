"""Durable, resumable storage for graph evaluation runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from window import WINDOW_REVISION, parse_window_size
from generation import GENERATION_REVISION

LEVELS = {"pair", "pair-head", "pair-layer", "adakv-layer"}
EXISTING_RESULTS_MODES = {"fail", "resume", "overwrite"}
_LEGACY_MANIFEST_KEYS = {"checkpoint_path", "wandb_run_id", "window_size", "level"}
_PREVIOUS_MANIFEST_KEYS = _LEGACY_MANIFEST_KEYS | {"prefill_mode"}
_PRE_WINDOW_MANIFEST_KEYS = _PREVIOUS_MANIFEST_KEYS | {
    "generation_revision",
    "ruler_prompt_mode",
    "dataset_revisions",
}
_MANIFEST_KEYS = _PRE_WINDOW_MANIFEST_KEYS | {"window_revision"}


def atomic_write_json(path: str | Path, payload) -> None:
    """Replace a JSON file only after its new contents are complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_component(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty path component")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{field} must not contain path separators")
    return value


def _load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error


def _validate_manifest(payload, *, allow_legacy=False) -> dict:
    keys = set(payload) if isinstance(payload, dict) else set()
    legacy = allow_legacy and keys in (
        _LEGACY_MANIFEST_KEYS,
        _PREVIOUS_MANIFEST_KEYS,
        _PRE_WINDOW_MANIFEST_KEYS,
    )
    if not isinstance(payload, dict) or (keys != _MANIFEST_KEYS and not legacy):
        raise ValueError(f"manifest must contain exactly {sorted(_MANIFEST_KEYS)}")
    checkpoint_path = payload["checkpoint_path"]
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise ValueError("manifest checkpoint_path must be absolute")
    run_id = payload["wandb_run_id"]
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("manifest wandb_run_id must be a non-empty string or null")
    window_size = payload["window_size"]
    if isinstance(window_size, bool) or not isinstance(window_size, (int, float)):
        raise ValueError("manifest window_size must be numeric")
    try:
        window_size = parse_window_size(window_size)
    except ValueError as error:
        raise ValueError("manifest window_size is invalid") from error
    window_revision = payload.get("window_revision", 0)
    if type(window_revision) is not int or window_revision < 0:
        raise ValueError("manifest window_revision must be a non-negative integer")
    level = payload["level"]
    if level not in LEVELS:
        raise ValueError(f"manifest level must be one of {sorted(LEVELS)}")
    prefill_mode = payload.get("prefill_mode", "post-prefill")
    if prefill_mode not in {"chunked", "post-prefill"}:
        raise ValueError("manifest prefill_mode must be chunked or post-prefill")
    generation_revision = payload.get("generation_revision", 1)
    if type(generation_revision) is not int or generation_revision < 1:
        raise ValueError("manifest generation_revision must be a positive integer")
    ruler_prompt_mode = payload.get("ruler_prompt_mode", "graphkv")
    if ruler_prompt_mode not in {"graphkv", "official"}:
        raise ValueError("manifest ruler_prompt_mode must be graphkv or official")
    dataset_revisions = payload.get("dataset_revisions", {})
    if not isinstance(dataset_revisions, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in dataset_revisions.items()
    ):
        raise ValueError("manifest dataset_revisions must map names to revisions")
    return {
        "checkpoint_path": checkpoint_path,
        "wandb_run_id": run_id,
        "window_size": window_size,
        "window_revision": window_revision,
        "level": level,
        "prefill_mode": prefill_mode,
        "generation_revision": generation_revision,
        "ruler_prompt_mode": ruler_prompt_mode,
        "dataset_revisions": dict(dataset_revisions),
    }


def _manifest(
    checkpoint_path: str | Path,
    wandb_run_id: str | None,
    window_size: int | float,
    level: str,
    prefill_mode: str,
    ruler_prompt_mode: str,
    dataset_revisions: dict | None,
) -> dict:
    try:
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}") from error
    if not path.is_file():
        raise ValueError(f"checkpoint is not a file: {path}")
    return _validate_manifest(
        {
            "checkpoint_path": str(path),
            "wandb_run_id": wandb_run_id,
            "window_size": window_size,
            "window_revision": WINDOW_REVISION,
            "level": level,
            "prefill_mode": prefill_mode,
            "generation_revision": GENERATION_REVISION,
            "ruler_prompt_mode": ruler_prompt_mode,
            "dataset_revisions": dataset_revisions or {},
        }
    )


@dataclass(frozen=True)
class ExampleResult:
    """A validated, possibly partial evaluation example."""

    path: Path
    task: str
    example_index: int
    payload: dict

    @property
    def formats(self) -> tuple[str, ...]:
        return tuple(self.payload)

    @property
    def requested_ratios(self) -> tuple[float, ...]:
        return tuple(float(entry[0][0]) for entry in self.payload[self.formats[0]])

    @property
    def full_answers(self) -> dict[str, str | None]:
        return {fmt: self.payload[fmt][0][1]["full__"] for fmt in self.formats}

    @property
    def has_full_answers(self) -> bool:
        return all(answer is not None for answer in self.full_answers.values())

    @property
    def answers(self) -> dict[str, str | list[str]]:
        return {fmt: self.payload[fmt][0][1]["answer"] for fmt in self.formats}


def _check_output(payload, path: Path) -> None:
    full_presence = set()
    for fmt, entries in payload.items():
        ratios = [float(entry[0][0]) for entry in entries]
        if len(ratios) != len(set(ratios)):
            raise ValueError(f"duplicate requested ratio for {fmt}: {path}")
        full_answers = {entry[1]["full__"] for entry in entries}
        if len(full_answers) > 1:
            raise ValueError(f"full-cache answer changed for {fmt}: {path}")
        full_presence.add(next(iter(full_answers)) is not None)
    if len(full_presence) > 1:
        raise ValueError(
            f"full-cache answer coverage must be complete per example: {path}"
        )


class EvaluationRun:
    """One resumable evaluation run."""

    def __init__(self, results_root: Path, run_name: str, manifest: dict):
        self.results_root = results_root
        self.run_name = run_name
        self.run_dir = results_root / run_name
        self.manifest_path = self.run_dir / "manifest.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.datasets_path = self.run_dir / "datasets.json"
        self.outputs_dir = self.run_dir / "outputs"
        self._manifest = manifest

    @classmethod
    def open(
        cls,
        results_root: str | Path,
        run_name: str,
        *,
        checkpoint_path: str | Path,
        wandb_run_id: str | None,
        window_size: int | float,
        level: str,
        prefill_mode: str = "chunked",
        ruler_prompt_mode: str = "graphkv",
        dataset_revisions: dict | None = None,
        existing_results: str = "fail",
    ) -> "EvaluationRun":
        _safe_component(run_name, "run name")
        if existing_results not in EXISTING_RESULTS_MODES:
            raise ValueError(
                f"existing_results must be one of {sorted(EXISTING_RESULTS_MODES)}"
            )
        manifest = _manifest(
            checkpoint_path,
            wandb_run_id,
            window_size,
            level,
            prefill_mode,
            ruler_prompt_mode,
            dataset_revisions,
        )
        run = cls(Path(results_root), run_name, manifest)
        run.results_root.mkdir(parents=True, exist_ok=True)
        run._initialize(existing_results)
        return run

    @classmethod
    def load(cls, run_dir: str | Path) -> "EvaluationRun":
        run_dir = Path(run_dir)
        run_name = _safe_component(run_dir.name, "run name")
        if not run_dir.is_dir():
            raise ValueError(f"evaluation run does not exist: {run_dir}")
        manifest = _validate_manifest(
            _load_json(run_dir / "manifest.json"), allow_legacy=True
        )
        return cls(run_dir.parent, run_name, manifest)

    @property
    def manifest(self) -> dict:
        return dict(self._manifest)

    @property
    def dataset_sizes(self) -> dict[str, int]:
        """Full benchmark cardinalities saved before evaluation produced outputs."""
        if not self.datasets_path.exists():
            return {}
        sizes = _load_json(self.datasets_path)
        if not isinstance(sizes, dict) or any(
            not isinstance(task, str) or type(size) is not int or size < 0
            for task, size in sizes.items()
        ):
            raise ValueError(f"invalid dataset sizes in {self.datasets_path}")
        for task in sizes:
            _safe_component(task, "task")
        return sizes

    def record_dataset_size(self, task: str, size: int) -> None:
        """Record the complete size, never the currently requested pilot range."""
        _safe_component(task, "task")
        if type(size) is not int or size < 0:
            raise ValueError("dataset size must be a non-negative integer")
        sizes = self.dataset_sizes
        if task in sizes:
            if sizes[task] != size:
                raise ValueError(
                    f"dataset size changed for {task}: {sizes[task]} != {size}"
                )
            return
        sizes[task] = size
        atomic_write_json(self.datasets_path, sizes)

    def _initialize(self, mode: str) -> None:
        if mode == "overwrite" and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        elif mode == "fail" and self.run_dir.exists():
            raise FileExistsError(f"evaluation run already exists: {self.run_dir}")

        if not self.run_dir.exists() or not self.manifest_path.exists():
            self._create()
        self._validate_existing_manifest()

    def _create(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            atomic_write_json(self.manifest_path, self._manifest)

    def _validate_existing_manifest(self) -> None:
        existing = _validate_manifest(_load_json(self.manifest_path))
        if existing != self._manifest:
            differences = [
                key
                for key in sorted(_MANIFEST_KEYS)
                if existing[key] != self._manifest[key]
            ]
            raise ValueError(
                f"evaluation run manifest mismatch for {', '.join(differences)}"
            )

    def __enter__(self) -> "EvaluationRun":
        return self

    def __exit__(self, *_exc_info) -> None:
        return None

    def output_path(self, task: str, example_index: int) -> Path:
        _safe_component(task, "task")
        if (
            isinstance(example_index, bool)
            or not isinstance(example_index, int)
            or example_index < 0
        ):
            raise ValueError("example_index must be a non-negative integer")
        return self.outputs_dir / task / f"{example_index}.json"

    def load_example(self, task: str, example_index: int) -> ExampleResult | None:
        path = self.output_path(task, example_index)
        if not path.exists():
            return None
        payload = _load_json(path)
        _check_output(payload, path)
        return ExampleResult(
            path=path,
            task=task,
            example_index=example_index,
            payload=payload,
        )

    def merge_example(
        self,
        task: str,
        example_index: int,
        *,
        outputs: Mapping[str, list] | None = None,
        full_answers: Mapping[str, str] | None = None,
    ) -> ExampleResult:
        """Merge complete ratio results or add full-cache answers, then save once."""
        task = _safe_component(task, "task")
        path = self.output_path(task, example_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load_example(task, example_index)
        new_outputs = dict(outputs or {})
        if existing is None and not new_outputs:
            raise ValueError("cannot add full answers before any ratio result exists")

        if existing is None:
            payload = new_outputs
        else:
            payload = existing.payload
            if new_outputs:
                self._merge_ratios(payload, new_outputs, path)

        if full_answers is not None:
            self._merge_full_answers(payload, full_answers, path)
        atomic_write_json(path, payload)
        return ExampleResult(
            path=path,
            task=task,
            example_index=example_index,
            payload=payload,
        )

    @staticmethod
    def _merge_ratios(existing: dict, incoming: dict, path: Path) -> None:
        if set(existing) != set(incoming):
            raise ValueError(f"result formats do not match in {path}")
        for fmt in existing:
            saved = {float(entry[0][0]) for entry in existing[fmt]}
            repeated = saved.intersection(float(entry[0][0]) for entry in incoming[fmt])
            if repeated:
                raise ValueError(
                    f"duplicate result for ratio {min(repeated)} in {path}"
                )
        for fmt in existing:
            full_answers = {
                entry[1]["full__"]
                for entry in [*existing[fmt], *incoming[fmt]]
                if entry[1]["full__"] is not None
            }
            if len(full_answers) > 1:
                raise ValueError(f"conflicting full-cache answer for {fmt} in {path}")
            full_answer = next(iter(full_answers), None)
            existing[fmt].extend(incoming[fmt])
            if full_answer is not None:
                for _info, text in existing[fmt]:
                    text["full__"] = full_answer

    @staticmethod
    def _merge_full_answers(
        payload: dict, full_answers: Mapping[str, str], path: Path
    ) -> None:
        formats = list(payload)
        if set(full_answers) != set(formats):
            raise ValueError(f"full answers must exactly match formats in {path}")
        if any(not isinstance(answer, str) for answer in full_answers.values()):
            raise ValueError(f"full answers must be strings in {path}")
        for fmt in formats:
            answer = full_answers[fmt]
            for _info, text in payload[fmt]:
                if text["full__"] is not None and text["full__"] != answer:
                    raise ValueError(
                        f"conflicting full-cache answer for {fmt} in {path}"
                    )
                text["full__"] = answer

    def iter_examples(self) -> Iterator[ExampleResult]:
        if not self.outputs_dir.exists():
            return
        for task_dir in sorted(
            path for path in self.outputs_dir.iterdir() if path.is_dir()
        ):
            _safe_component(task_dir.name, "task")
            paths = sorted(
                task_dir.glob("*.json"),
                key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
            )
            for path in paths:
                if not path.stem.isdigit():
                    raise ValueError(
                        f"output filename must be a non-negative integer: {path}"
                    )
                if path.name != f"{int(path.stem)}.json":
                    raise ValueError(f"output filename is not canonical: {path}")
                payload = _load_json(path)
                _check_output(payload, path)
                yield ExampleResult(
                    path=path,
                    task=task_dir.name,
                    example_index=int(path.stem),
                    payload=payload,
                )

    def write_metrics(self, payload) -> None:
        atomic_write_json(self.metrics_path, payload)
