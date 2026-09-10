import hashlib
import json
from types import SimpleNamespace

from data.load import AgenticDataset
from generation import GENERATION_REVISION


class Teacher:
    sys_prompt_ids = [11]
    postfix_ids = [12]
    gen_kwargs = {"max_new_tokens": 2, "do_sample": False}

    def __init__(self):
        self.model = SimpleNamespace(
            name_or_path="org/model",
            config=SimpleNamespace(_commit_hash="a" * 40),
        )
        self.tokenizer = SimpleNamespace(
            name_or_path="org/tokenizer", init_kwargs={"_commit_hash": "b" * 40}
        )
        self.calls = 0

    def apply_template(self, query):
        return [13, query, 14]

    def generate(self, query, *, kv):
        self.calls += 1
        return f"complete answer {self.calls}"


def test_agentic_cache_reuses_current_answers_but_not_unversioned_outputs(tmp_path):
    teacher = Teacher()

    def dataset():
        return AgenticDataset(
            [{"prompt": [{"role": "user", "content": "context\n\nquestion"}]}],
            teacher=teacher,
            answer_cache_dir=tmp_path,
            count=1,
        )

    assert dataset().resolve_answers(0, object()) == ["complete answer 1"]
    assert dataset().resolve_answers(0, object()) == ["complete answer 1"]
    assert teacher.calls == 1
    current_path = next(tmp_path.glob("*.json"))
    entry = json.loads(current_path.read_text())
    assert entry["identity"]["generation_revision"] == GENERATION_REVISION

    del entry["identity"]["generation_revision"]
    entry["answer"] = "old truncated answer"
    identity = json.dumps(entry["identity"], sort_keys=True, separators=(",", ":"))
    legacy_key = hashlib.sha256(identity.encode()).hexdigest()
    legacy_path = tmp_path / f"{legacy_key}.json"
    legacy_path.write_text(json.dumps(entry))
    current_path.unlink()

    assert dataset().resolve_answers(0, object()) == ["complete answer 2"]
    assert teacher.calls == 2
    assert legacy_path.exists()
    assert len(list(tmp_path.glob("*.json"))) == 2
