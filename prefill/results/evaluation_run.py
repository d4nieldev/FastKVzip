"""Durable, resumable storage for graph evaluation runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


LEVELS = {"pair", "pair-head", "pair-layer", "adakv-layer"}
EXISTING_RESULTS_MODES = {"fail", "resume", "overwrite"}
_MANIFEST_KEYS = {"checkpoint_path", "checkpoint_sha256", "window_size", "level"}
_META_KEYS = {"task", "example_index", "dataset_size", "input_sha256", "formats"}


def checkpoint_sha256(path: str | Path) -> str:
    """Hash every byte of a checkpoint file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_input(sample: Mapping) -> str:
    """Hash the context, questions, and answers using canonical JSON."""
    try:
        identity = {
            "context": sample["context"],
            "question": sample["question"],
            "answers": sample["answers"],
        }
    except KeyError as error:
        raise ValueError(f"evaluation sample is missing {error.args[0]!r}") from error
    try:
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation sample is not JSON serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload) -> None:
    """Replace a JSON file atomically after fully writing it beside the target."""
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


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _validate_manifest(payload) -> dict:
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise ValueError(f"manifest must contain exactly {sorted(_MANIFEST_KEYS)}")
    checkpoint_path = payload["checkpoint_path"]
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise ValueError("manifest checkpoint_path must be absolute")
    window_size = payload["window_size"]
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError("manifest window_size must be a positive integer")
    level = payload["level"]
    if level not in LEVELS:
        raise ValueError(f"manifest level must be one of {sorted(LEVELS)}")
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _sha256(
            payload["checkpoint_sha256"], "manifest checkpoint_sha256"
        ),
        "window_size": window_size,
        "level": level,
    }


def _manifest(checkpoint_path: str | Path, window_size: int, level: str) -> dict:
    try:
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}") from error
    if not path.is_file():
        raise ValueError(f"checkpoint is not a file: {path}")
    return _validate_manifest(
        {
            "checkpoint_path": str(path),
            "checkpoint_sha256": checkpoint_sha256(path),
            "window_size": window_size,
            "level": level,
        }
    )


@dataclass(frozen=True)
class ExampleResult:
    """A validated, possibly partial evaluation example."""

    path: Path
    payload: dict

    @property
    def metadata(self) -> dict:
        return self.payload["_meta"]

    @property
    def task(self) -> str:
        return self.metadata["task"]

    @property
    def example_index(self) -> int:
        return self.metadata["example_index"]

    @property
    def dataset_size(self) -> int:
        return self.metadata["dataset_size"]

    @property
    def input_sha256(self) -> str:
        return self.metadata["input_sha256"]

    @property
    def formats(self) -> tuple[str, ...]:
        return tuple(self.metadata["formats"])

    @property
    def requested_ratios(self) -> tuple[float, ...]:
        first_format = self.formats[0]
        return tuple(float(entry[0][0]) for entry in self.payload[first_format])

    @property
    def full_answers(self) -> dict[str, str | None]:
        return {
            fmt: self.payload[fmt][0][1]["full__"] for fmt in self.formats
        }

    @property
    def has_full_answers(self) -> bool:
        return all(answer is not None for answer in self.full_answers.values())

    @property
    def answers(self) -> dict[str, str]:
        return {fmt: self.payload[fmt][0][1]["answer"] for fmt in self.formats}


def _validate_output(
    payload,
    path: Path,
    *,
    expected_task: str | None = None,
    expected_index: int | None = None,
    expected_dataset_size: int | None = None,
    expected_input_sha256: str | None = None,
) -> ExampleResult:
    if not isinstance(payload, dict) or "_meta" not in payload:
        raise ValueError(f"result must be an object with _meta: {path}")
    meta = payload["_meta"]
    if not isinstance(meta, dict) or set(meta) != _META_KEYS:
        raise ValueError(f"result _meta must contain exactly {sorted(_META_KEYS)}: {path}")

    task = _safe_component(meta["task"], "task")
    index = meta["example_index"]
    dataset_size = meta["dataset_size"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"example_index must be a non-negative integer: {path}")
    if (
        isinstance(dataset_size, bool)
        or not isinstance(dataset_size, int)
        or dataset_size < 1
        or index >= dataset_size
    ):
        raise ValueError(f"dataset_size must be larger than example_index: {path}")
    input_digest = _sha256(meta["input_sha256"], "input_sha256")
    formats = meta["formats"]
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(fmt, str) or not fmt for fmt in formats)
        or len(set(formats)) != len(formats)
    ):
        raise ValueError(f"formats must be a non-empty list of unique strings: {path}")
    if set(payload) != {"_meta", *formats}:
        raise ValueError(f"formats must exactly match result keys: {path}")

    expected = {
        "task": (task, expected_task),
        "example_index": (index, expected_index),
        "dataset_size": (dataset_size, expected_dataset_size),
        "input_sha256": (input_digest, expected_input_sha256),
    }
    for field, (actual, wanted) in expected.items():
        if wanted is not None and actual != wanted:
            raise ValueError(
                f"result {field} mismatch in {path}: expected {wanted!r}, got {actual!r}"
            )

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
    return ExampleResult(path=path, payload=payload)


def _lock(handle, operation: int, *, blocking: bool = True) -> None:
    if not blocking:
        operation |= fcntl.LOCK_NB
    descriptor = handle if isinstance(handle, int) else handle.fileno()
    fcntl.flock(descriptor, operation)


@contextmanager
def _directory_lock(path: Path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        _lock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        _lock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class EvaluationRun:
    """One evaluation run with resumable, process-safe example storage."""

    def __init__(self, results_root: Path, run_name: str, manifest: dict):
        self.results_root = results_root
        self.run_name = run_name
        self.run_dir = results_root / run_name
        self.manifest_path = self.run_dir / "manifest.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.outputs_dir = self.run_dir / "outputs"
        self._manifest = manifest
        self._run_directory_lock = None
        self._manifest_lock = None
        self._exclusive_depth = 0

    @classmethod
    def open(
        cls,
        results_root: str | Path,
        run_name: str,
        *,
        checkpoint_path: str | Path,
        window_size: int,
        level: str,
        existing_results: str = "fail",
    ) -> "EvaluationRun":
        _safe_component(run_name, "run name")
        if existing_results not in EXISTING_RESULTS_MODES:
            raise ValueError(
                f"existing_results must be one of {sorted(EXISTING_RESULTS_MODES)}"
            )
        manifest = _manifest(checkpoint_path, window_size, level)
        run = cls(Path(results_root), run_name, manifest)
        run.results_root.mkdir(parents=True, exist_ok=True)
        try:
            run._initialize(existing_results)
        except BaseException:
            run.close()
            raise
        return run

    @classmethod
    def load(cls, run_dir: str | Path) -> "EvaluationRun":
        """Open an existing run for metrics parsing and revalidate its checkpoint."""
        run_dir = Path(run_dir)
        run_name = _safe_component(run_dir.name, "run name")
        results_root = run_dir.parent
        if not results_root.is_dir():
            raise ValueError(f"evaluation results root does not exist: {results_root}")
        try:
            with _directory_lock(results_root):
                if not run_dir.is_dir():
                    raise ValueError(f"evaluation run does not exist: {run_dir}")
                manifest = _validate_manifest(_load_json(run_dir / "manifest.json"))
                run = cls(results_root, run_name, manifest)
                run._open_manifest_lock()
            current = _manifest(
                manifest["checkpoint_path"],
                manifest["window_size"],
                manifest["level"],
            )
            if current != manifest:
                raise ValueError("evaluation run checkpoint contents changed")
        except BaseException:
            if "run" in locals():
                run.close()
            raise
        return run

    @property
    def manifest(self) -> dict:
        return dict(self._manifest)

    def _initialize(self, mode: str) -> None:
        with _directory_lock(self.results_root):
            if mode == "overwrite":
                if self.run_dir.exists():
                    existing_lock = os.open(self.run_dir, os.O_RDONLY)
                    try:
                        _lock(existing_lock, fcntl.LOCK_EX, blocking=False)
                    except BlockingIOError as error:
                        os.close(existing_lock)
                        raise RuntimeError(
                            f"cannot overwrite active evaluation run {self.run_name!r}"
                        ) from error
                    os.close(existing_lock)
                    shutil.rmtree(self.run_dir)
                self._create()
                self._open_manifest_lock()
                return

            if self.run_dir.exists():
                if mode == "fail":
                    raise FileExistsError(
                        f"evaluation run already exists: {self.run_dir}"
                    )
                self._open_manifest_lock()
                self._validate_existing_manifest()
                return

            self._create()
            self._open_manifest_lock()

    def _open_manifest_lock(self) -> None:
        self._run_directory_lock = os.open(self.run_dir, os.O_RDONLY)
        _lock(self._run_directory_lock, fcntl.LOCK_SH)
        self._manifest_lock = self.manifest_path.open("rb")
        _lock(self._manifest_lock, fcntl.LOCK_SH)

    def _create(self) -> None:
        self.outputs_dir.mkdir(parents=True)
        atomic_write_json(self.manifest_path, self._manifest)

    def _validate_existing_manifest(self) -> None:
        if not self.run_dir.is_dir() or not self.manifest_path.is_file():
            raise ValueError(
                f"existing evaluation run has no valid manifest: {self.run_dir}"
            )
        existing = _validate_manifest(_load_json(self.manifest_path))
        if existing != self._manifest:
            differences = [
                key for key in sorted(_MANIFEST_KEYS) if existing[key] != self._manifest[key]
            ]
            raise ValueError(
                f"evaluation run manifest mismatch for {', '.join(differences)}"
            )

    def close(self) -> None:
        if self._manifest_lock is not None:
            fcntl.flock(self._manifest_lock.fileno(), fcntl.LOCK_UN)
            self._manifest_lock.close()
            self._manifest_lock = None
        if self._run_directory_lock is not None:
            fcntl.flock(self._run_directory_lock, fcntl.LOCK_UN)
            os.close(self._run_directory_lock)
            self._run_directory_lock = None

    def __enter__(self) -> "EvaluationRun":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._manifest_lock is None:
            raise RuntimeError("evaluation run is closed")

    def output_path(self, task: str, example_index: int) -> Path:
        _safe_component(task, "task")
        if (
            isinstance(example_index, bool)
            or not isinstance(example_index, int)
            or example_index < 0
        ):
            raise ValueError("example_index must be a non-negative integer")
        return self.outputs_dir / task / f"{example_index}.json"

    def load_example(
        self,
        task: str,
        example_index: int,
        *,
        dataset_size: int | None = None,
        input_sha256: str | None = None,
    ) -> ExampleResult | None:
        self._require_open()
        path = self.output_path(task, example_index)
        if not path.exists():
            return None
        return _validate_output(
            _load_json(path),
            path,
            expected_task=task,
            expected_index=example_index,
            expected_dataset_size=dataset_size,
            expected_input_sha256=input_sha256,
        )

    def merge_example(
        self,
        task: str,
        example_index: int,
        *,
        dataset_size: int,
        input_sha256: str,
        outputs: Mapping[str, list] | None = None,
        full_answers: Mapping[str, str] | None = None,
    ) -> ExampleResult:
        """Merge complete ratio results or add full-cache answers, then save once."""
        self._require_open()
        task = _safe_component(task, "task")
        _sha256(input_sha256, "input_sha256")
        path = self.output_path(task, example_index)
        task_dir = path.parent
        task_dir.mkdir(parents=True, exist_ok=True)
        with _directory_lock(task_dir):
            existing = self.load_example(
                task,
                example_index,
                dataset_size=dataset_size,
                input_sha256=input_sha256,
            )
            new_outputs = dict(outputs or {})
            if existing is None and not new_outputs:
                raise ValueError("cannot add full answers before any ratio result exists")

            if existing is None:
                formats = list(new_outputs)
                payload = {
                    "_meta": {
                        "task": task,
                        "example_index": example_index,
                        "dataset_size": dataset_size,
                        "input_sha256": input_sha256,
                        "formats": formats,
                    },
                    **new_outputs,
                }
                current = _validate_output(
                    payload,
                    path,
                    expected_task=task,
                    expected_index=example_index,
                    expected_dataset_size=dataset_size,
                    expected_input_sha256=input_sha256,
                )
                payload = current.payload
            else:
                payload = existing.payload
                if new_outputs:
                    fragment = {
                        "_meta": dict(payload["_meta"]),
                        **new_outputs,
                    }
                    current = _validate_output(
                        fragment,
                        path,
                        expected_task=task,
                        expected_index=example_index,
                        expected_dataset_size=dataset_size,
                        expected_input_sha256=input_sha256,
                    )
                    self._merge_ratios(payload, current.payload, path)

            if full_answers is not None:
                self._merge_full_answers(payload, full_answers, path)
            validated = _validate_output(
                payload,
                path,
                expected_task=task,
                expected_index=example_index,
                expected_dataset_size=dataset_size,
                expected_input_sha256=input_sha256,
            )
            atomic_write_json(path, validated.payload)
            return validated

    @staticmethod
    def _merge_ratios(existing: dict, incoming: dict, path: Path) -> None:
        formats = existing["_meta"]["formats"]
        for fmt in formats:
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
        formats = payload["_meta"]["formats"]
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
        self._require_open()
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
                    expected_task=task_dir.name,
                    expected_index=int(path.stem),
                )

    @contextmanager
    def exclusive_finalization(self):
        """Wait for all other evaluators and prevent writers during finalization."""
        self._require_open()
        assert self._manifest_lock is not None
        if self._exclusive_depth == 0:
            _lock(self._manifest_lock, fcntl.LOCK_EX)
        self._exclusive_depth += 1
        try:
            yield self
        finally:
            self._exclusive_depth -= 1
            if self._exclusive_depth == 0:
                _lock(self._manifest_lock, fcntl.LOCK_SH)

    def write_metrics(self, payload) -> None:
        self._require_open()
        with self.exclusive_finalization():
            atomic_write_json(self.metrics_path, payload)
