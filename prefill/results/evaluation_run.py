"""Durable, resumable storage for graph evaluation runs."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


LEVELS = {"pair", "pair-head", "pair-layer", "adakv-layer"}
EXISTING_RESULTS_MODES = {"fail", "resume", "overwrite"}
_MANIFEST_KEYS = {"checkpoint_path", "wandb_run_id", "window_size", "level"}


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


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def _validate_manifest(payload) -> dict:
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise ValueError(f"manifest must contain exactly {sorted(_MANIFEST_KEYS)}")
    checkpoint_path = payload["checkpoint_path"]
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise ValueError("manifest checkpoint_path must be absolute")
    run_id = payload["wandb_run_id"]
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("manifest wandb_run_id must be a non-empty string or null")
    window_size = payload["window_size"]
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError("manifest window_size must be a positive integer")
    level = payload["level"]
    if level not in LEVELS:
        raise ValueError(f"manifest level must be one of {sorted(LEVELS)}")
    return {
        "checkpoint_path": checkpoint_path,
        "wandb_run_id": run_id,
        "window_size": window_size,
        "level": level,
    }


def _manifest(
    checkpoint_path: str | Path,
    wandb_run_id: str | None,
    window_size: int,
    level: str,
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
            "level": level,
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
    def answers(self) -> dict[str, str]:
        return {fmt: self.payload[fmt][0][1]["answer"] for fmt in self.formats}


def _validate_output(payload, path: Path, *, task: str, example_index: int) -> ExampleResult:
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"result must be a non-empty object: {path}")
    formats = list(payload)
    if any(not isinstance(fmt, str) or not fmt for fmt in formats):
        raise ValueError(f"result keys must be non-empty strings: {path}")

    shared_ratios = None
    shared_info = {}
    full_presence = set()
    for fmt in formats:
        entries = payload[fmt]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{fmt} must contain at least one ratio result: {path}")
        ratios = []
        answer = None
        full_answer = None
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(f"invalid ratio entry for {fmt}: {path}")
            info, text = entry
            if not isinstance(info, list) or len(info) != 3:
                raise ValueError(f"ratio info must contain three numbers for {fmt}: {path}")
            requested = _number(info[0], "requested ratio")
            actual = _number(info[1], "actual ratio")
            threshold = _number(info[2], "threshold")
            if not 0 < requested < 1:
                raise ValueError(f"requested ratio must be between zero and one: {path}")
            if not 0 <= actual <= 1:
                raise ValueError(f"actual ratio must be between zero and one: {path}")
            if requested in ratios:
                raise ValueError(f"duplicate requested ratio {requested} for {fmt}: {path}")
            ratios.append(requested)
            if not isinstance(text, dict) or set(text) != {"pruned", "full__", "answer"}:
                raise ValueError(f"invalid generated-answer object for {fmt}: {path}")
            if not isinstance(text["pruned"], str) or not isinstance(text["answer"], str):
                raise ValueError(f"pruned and answer must be strings for {fmt}: {path}")
            if text["full__"] is not None and not isinstance(text["full__"], str):
                raise ValueError(f"full__ must be a string or null for {fmt}: {path}")
            if answer is not None and text["answer"] != answer:
                raise ValueError(f"answer changed across ratios for {fmt}: {path}")
            if full_answer is not None and text["full__"] not in {None, full_answer}:
                raise ValueError(f"full__ changed across ratios for {fmt}: {path}")
            answer = text["answer"]
            if text["full__"] is not None:
                full_answer = text["full__"]
            full_presence.add(text["full__"] is not None)
            current_info = (actual, threshold)
            if requested in shared_info and shared_info[requested] != current_info:
                raise ValueError(f"ratio metadata differs across formats at {requested}: {path}")
            shared_info[requested] = current_info
        if shared_ratios is None:
            shared_ratios = ratios
        elif ratios != shared_ratios:
            raise ValueError(f"formats must contain the same ratios in the same order: {path}")
    if len(full_presence) > 1:
        raise ValueError(f"full-cache answer coverage must be complete per example: {path}")
    return ExampleResult(
        path=path,
        task=task,
        example_index=example_index,
        payload=payload,
    )


class EvaluationRun:
    """One resumable evaluation run."""

    def __init__(self, results_root: Path, run_name: str, manifest: dict):
        self.results_root = results_root
        self.run_name = run_name
        self.run_dir = results_root / run_name
        self.manifest_path = self.run_dir / "manifest.json"
        self.metrics_path = self.run_dir / "metrics.json"
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
        window_size: int,
        level: str,
        existing_results: str = "fail",
    ) -> "EvaluationRun":
        _safe_component(run_name, "run name")
        if existing_results not in EXISTING_RESULTS_MODES:
            raise ValueError(
                f"existing_results must be one of {sorted(EXISTING_RESULTS_MODES)}"
            )
        manifest = _manifest(checkpoint_path, wandb_run_id, window_size, level)
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
        manifest = _validate_manifest(_load_json(run_dir / "manifest.json"))
        return cls(run_dir.parent, run_name, manifest)

    @property
    def manifest(self) -> dict:
        return dict(self._manifest)

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
                key for key in sorted(_MANIFEST_KEYS) if existing[key] != self._manifest[key]
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
        return _validate_output(
            _load_json(path),
            path,
            task=task,
            example_index=example_index,
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
            current = _validate_output(
                payload,
                path,
                task=task,
                example_index=example_index,
            )
            payload = current.payload
        else:
            payload = existing.payload
            if new_outputs:
                current = _validate_output(
                    new_outputs,
                    path,
                    task=task,
                    example_index=example_index,
                )
                self._merge_ratios(payload, current.payload, path)

        if full_answers is not None:
            self._merge_full_answers(payload, full_answers, path)
        validated = _validate_output(
            payload,
            path,
            task=task,
            example_index=example_index,
        )
        atomic_write_json(path, validated.payload)
        return validated

    @staticmethod
    def _merge_ratios(existing: dict, incoming: dict, path: Path) -> None:
        if set(existing) != set(incoming):
            raise ValueError(f"result formats do not match in {path}")
        for fmt in existing:
            by_ratio = {float(entry[0][0]): entry for entry in existing[fmt]}
            full_answers = {
                entry[1]["full__"]
                for entry in [*existing[fmt], *incoming[fmt]]
                if entry[1]["full__"] is not None
            }
            if len(full_answers) > 1:
                raise ValueError(f"conflicting full-cache answer for {fmt} in {path}")
            full_answer = next(iter(full_answers), None)
            for entry in incoming[fmt]:
                ratio = float(entry[0][0])
                if ratio not in by_ratio:
                    existing[fmt].append(entry)
                    by_ratio[ratio] = entry
                    continue
                old_info, old_text = by_ratio[ratio]
                new_info, new_text = entry
                if (
                    old_info != new_info
                    or old_text["pruned"] != new_text["pruned"]
                    or old_text["answer"] != new_text["answer"]
                ):
                    raise ValueError(f"conflicting result for ratio {ratio} in {path}")
                old_full, new_full = old_text["full__"], new_text["full__"]
                if old_full is not None and new_full is not None and old_full != new_full:
                    raise ValueError(f"conflicting full-cache answer for {fmt} in {path}")
                raise ValueError(f"duplicate result for ratio {ratio} in {path}")
            if full_answer is not None:
                for _info, text in existing[fmt]:
                    text["full__"] = full_answer

    @staticmethod
    def _merge_full_answers(payload: dict, full_answers: Mapping[str, str], path: Path) -> None:
        formats = list(payload)
        if set(full_answers) != set(formats):
            raise ValueError(f"full answers must exactly match formats in {path}")
        if any(not isinstance(answer, str) for answer in full_answers.values()):
            raise ValueError(f"full answers must be strings in {path}")
        for fmt in formats:
            answer = full_answers[fmt]
            for _info, text in payload[fmt]:
                if text["full__"] is not None and text["full__"] != answer:
                    raise ValueError(f"conflicting full-cache answer for {fmt} in {path}")
                text["full__"] = answer

    def iter_examples(self) -> Iterator[ExampleResult]:
        if not self.outputs_dir.exists():
            return
        for task_dir in sorted(path for path in self.outputs_dir.iterdir() if path.is_dir()):
            _safe_component(task_dir.name, "task")
            paths = sorted(
                task_dir.glob("*.json"),
                key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
            )
            for path in paths:
                if not path.stem.isdigit():
                    raise ValueError(f"output filename must be a non-negative integer: {path}")
                if path.name != f"{int(path.stem)}.json":
                    raise ValueError(f"output filename is not canonical: {path}")
                yield _validate_output(
                    _load_json(path),
                    path,
                    task=task_dir.name,
                    example_index=int(path.stem),
                )

    def write_metrics(self, payload) -> None:
        atomic_write_json(self.metrics_path, payload)
