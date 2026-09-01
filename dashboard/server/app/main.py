"""FastAPI app: agent ingest, read API, and the static dashboard bundle."""

from __future__ import annotations

import asyncio
import gzip
import hmac
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from . import db, ingest, logstore, queries

AGENT_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
MAX_INGEST_BYTES = int(os.environ.get("MAX_INGEST_BYTES", str(32 * 1024 * 1024)))
PRUNE_INTERVAL_SECONDS = 3600
DEFAULT_LOG_LIMIT = 256 * 1024
MAX_LOG_LIMIT = 8 * 1024 * 1024

STATIC_DIR = os.path.abspath(
    os.environ.get(
        "STATIC_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    )
)


async def _prune_loop() -> None:
    while True:
        try:
            removed = await asyncio.to_thread(queries.prune)
            if removed:
                print(f"retention: pruned {removed} jobs", flush=True)
        except Exception as exc:  # noqa: BLE001 - retention must not kill the app
            print(f"retention failed: {exc}", flush=True)
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    os.makedirs(logstore.LOG_DIR, exist_ok=True)
    task = asyncio.create_task(_prune_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="FastKVzip SLURM Dashboard", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


@app.post("/api/ingest")
async def post_ingest(request: Request) -> JSONResponse:
    # Reads are open by design; writes are not. Without this an outsider could
    # poison the job list, so the token guards ingest even though the UI is public.
    if not AGENT_TOKEN:
        raise HTTPException(503, "server has no DASHBOARD_TOKEN configured")
    supplied = request.headers.get("X-Agent-Token", "")
    if not hmac.compare_digest(supplied, AGENT_TOKEN):
        raise HTTPException(401, "bad agent token")

    body = await request.body()
    if len(body) > MAX_INGEST_BYTES:
        raise HTTPException(413, "payload too large")

    if request.headers.get("Content-Encoding", "").lower() == "gzip":
        try:
            body = await asyncio.to_thread(gzip.decompress, body)
        except (OSError, EOFError) as exc:
            raise HTTPException(400, f"bad gzip body: {exc}") from exc

    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise HTTPException(400, f"bad json body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "payload must be an object")

    result = await asyncio.to_thread(ingest.apply_payload, payload)
    return JSONResponse(result)


# --------------------------------------------------------------------------- #
# Read API
# --------------------------------------------------------------------------- #


@app.get("/api/status")
async def get_status() -> dict:
    return await asyncio.to_thread(queries.status)


@app.get("/api/jobs")
async def get_jobs(
    window_from: int | None = Query(None, alias="from"),
    window_to: int | None = Query(None, alias="to"),
    states: str | None = None,
    q: str | None = None,
) -> dict:
    state_list = [s for s in (states or "").split(",") if s.strip()]
    jobs = await asyncio.to_thread(
        queries.list_jobs,
        window_from=window_from,
        window_to=window_to,
        states=state_list or None,
        search=q,
    )
    return {"jobs": jobs, "server_time": int(time.time())}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = await asyncio.to_thread(queries.get_job, job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/api/jobs/{job_id}/script")
async def get_job_script(job_id: str) -> dict:
    """The sbatch script and submission environment, when SLURM still has them.

    Fetched on demand rather than ridden along with the job list: a script is
    read once, by one person, and the list is polled every ten seconds.
    """
    if await asyncio.to_thread(queries.get_job, job_id) is None:
        raise HTTPException(404, "no such job")
    record = await asyncio.to_thread(queries.get_script, job_id)
    return record or {"job_id": job_id, "batch_script": None, "job_env": None}


@app.get("/api/jobs/{job_id}/log")
async def get_job_log(
    job_id: str,
    offset: int = 0,
    limit: int = DEFAULT_LOG_LIMIT,
    tail: int | None = None,
) -> dict:
    """A byte range of a stored log.

    ``tail`` returns the last N bytes (how the viewer opens); ``offset`` reads
    forward from a known position (how it follows a running job).
    """
    if await asyncio.to_thread(queries.get_job, job_id) is None:
        raise HTTPException(404, "no such job")
    limit = max(1, min(limit, MAX_LOG_LIMIT))

    try:
        if tail:
            data, start, total = await asyncio.to_thread(
                logstore.read_tail, job_id, min(tail, MAX_LOG_LIMIT)
            )
        else:
            data, total = await asyncio.to_thread(
                logstore.read_range, job_id, max(0, offset), limit
            )
            start = max(0, offset)
    except logstore.UnsafeJobId as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "job_id": job_id,
        "offset": start,
        "next_offset": start + len(data),
        "total_size": total,
        # Logs are written by torch/tqdm and may hold stray bytes; never fail a
        # read over one bad character.
        "text": data.decode("utf-8", errors="replace"),
    }


@app.get("/api/jobs/{job_id}/log/download")
async def download_job_log(job_id: str) -> Response:
    try:
        path = logstore.log_path(job_id)
    except logstore.UnsafeJobId as exc:
        raise HTTPException(400, str(exc)) from exc
    if not os.path.exists(path):
        raise HTTPException(404, "no log stored for this job")
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}.log")


@app.post("/api/jobs/{job_id}/seen")
async def mark_job_seen(job_id: str) -> dict:
    """Stop a finished job announcing itself; the user has now read it."""
    if not await asyncio.to_thread(queries.mark_seen, job_id):
        raise HTTPException(404, "no such job")
    return {"job_id": job_id, "seen": True}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "time": int(time.time())}


# --------------------------------------------------------------------------- #
# Static bundle
# --------------------------------------------------------------------------- #


@app.get("/{full_path:path}")
async def serve_spa(full_path: str) -> Response:
    """Serve the built dashboard, falling back to index.html for client routes."""
    candidate = os.path.normpath(os.path.join(STATIC_DIR, full_path))
    if (
        full_path
        and (candidate == STATIC_DIR or candidate.startswith(STATIC_DIR + os.sep))
        and os.path.isfile(candidate)
    ):
        return FileResponse(candidate)

    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    raise HTTPException(404, "dashboard bundle not built; run `npm run build` in web/")
