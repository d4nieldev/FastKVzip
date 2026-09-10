import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import constants

from data.ruler import load_ruler


def test_requested_task_is_cached_and_a_new_process_can_reuse_it_without_network(
    monkeypatch, tmp_path
):
    cache = tmp_path / "hf"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache / "hub"))
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", cache / "datasets")
    buffer = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "input": f"context {index}\nQuestion: Where? Answer:",
                    "outputs": ["here", "there"],
                }
                for index in range(500)
            ]
        ),
        buffer,
    )
    content = buffer.getvalue().to_pybytes()
    revision = "90daf679d2893abc90bbc9451f2a1f33de86c66e"
    expected_url = (
        "https://huggingface.co/datasets/lighteval/RULER-4096-Qwen2.5-Instruct"
        f"/resolve/{revision}/data/qa_1-00000-of-00001.parquet"
    )
    requests_seen = []

    def response(_session, method, url, **_kwargs):
        # Only the external HTTP boundary is replaced. Native HF downloading,
        # disk caching, Parquet reading and our dataset adapter all run normally.
        assert url == expected_url, "must fetch only the requested pinned task"
        requests_seen.append(method)
        result = requests.Response()
        result.status_code = 200
        result.url = url
        result.request = requests.Request(method, url).prepare()
        result.headers.update(
            {
                "Content-Length": str(len(content)),
                "ETag": hashlib.sha256(content).hexdigest(),
                "X-Repo-Commit": revision,
            }
        )
        result.raw = io.BytesIO(content if method == "GET" else b"")
        return result

    monkeypatch.setattr(requests.sessions.Session, "request", response)
    rows = load_ruler("ruler_qa_1_4k")
    assert "GET" in requests_seen
    assert len(rows) == rows.full_size == 500
    assert rows[2] == {
        "context": "context 2",
        "question": ["\nQuestion: Where? Answer:"],
        "answers": [["here", "there"]],
    }

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import requests

def no_network(*args, **kwargs):
    raise AssertionError("warm cache must not use HTTP")

requests.sessions.Session.request = no_network
from data.ruler import load_ruler
rows = load_ruler("ruler_qa_1_4k")
assert len(rows) == rows.full_size == 500
assert load_ruler("ruler_qa_1_4k", start=2, n_data=1) == [rows[2]]
assert load_ruler("ruler_qa_1_4k", n_data=0) == []
print(json.dumps(rows))
""",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "HF_HOME": str(cache),
            "HF_HUB_CACHE": str(cache / "hub"),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == rows
