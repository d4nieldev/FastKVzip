"""On-disk append-only store for job logs.

The agent ships byte ranges; this module is the authority on how much of each
log the server actually holds, and reports that back so the agent can correct
its offsets (including re-shipping from zero after an ephemeral disk is wiped).
"""

from __future__ import annotations

import os
import re

from . import db

LOG_DIR = os.path.join(db.DATA_DIR, "logs")

# Job ids are "12345" or "12345_7"; anything else is refused rather than
# sanitized, so a hostile id can never walk out of LOG_DIR.
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_+.-]{1,64}$")


class UnsafeJobId(ValueError):
    pass


def log_path(job_id: str) -> str:
    if not _SAFE_JOB_ID.match(job_id) or job_id in {".", ".."}:
        raise UnsafeJobId(f"refusing unsafe job id: {job_id!r}")
    return os.path.join(LOG_DIR, f"{job_id}.log")


def current_size(job_id: str) -> int:
    try:
        return os.path.getsize(log_path(job_id))
    except (OSError, UnsafeJobId):
        return 0


def append(job_id: str, offset: int, data: bytes, truncate: bool = False) -> tuple[int, bool]:
    """Append ``data`` at ``offset``.

    Returns (next_offset, accepted). A mismatch is not an error: the caller
    replies with the true size and the agent re-aligns. Overlapping writes are
    trimmed rather than rejected, so a retried poll is idempotent.

    ``truncate`` replaces the stored log instead of appending. The agent sets it
    when the cluster-side file shrank or was replaced -- a requeued job writing
    over the same path. Without it the shorter new content would look like an
    already-stored duplicate and the stale log would never be replaced.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    path = log_path(job_id)

    if truncate:
        with open(path, "wb") as handle:
            handle.write(data)
        return len(data), True

    size = current_size(job_id)

    if offset > size:
        # A gap: we are missing bytes the agent already skipped past. Refuse and
        # let it restart, otherwise the stored log would be silently corrupt.
        return size, False

    if offset < size:
        overlap = size - offset
        if overlap >= len(data):
            # Entire chunk already stored (a duplicate retry).
            return size, True
        data = data[overlap:]

    with open(path, "ab") as handle:
        handle.write(data)
    return size + len(data), True


def read_range(job_id: str, offset: int = 0, limit: int | None = None) -> tuple[bytes, int]:
    """Read a slice of a stored log; returns (data, total_size)."""
    total = current_size(job_id)
    if total == 0 or offset >= total:
        return b"", total
    with open(log_path(job_id), "rb") as handle:
        handle.seek(max(0, offset))
        data = handle.read(limit) if limit is not None else handle.read()
    return data, total


def read_tail(job_id: str, tail_bytes: int) -> tuple[bytes, int, int]:
    """Read the last ``tail_bytes``; returns (data, start_offset, total_size)."""
    total = current_size(job_id)
    start = max(0, total - tail_bytes)
    data, _ = read_range(job_id, start, tail_bytes)
    return data, start, total


def delete(job_id: str) -> None:
    try:
        os.unlink(log_path(job_id))
    except (OSError, UnsafeJobId):
        pass
