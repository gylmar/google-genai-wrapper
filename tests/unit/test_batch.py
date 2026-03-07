import argparse
from pathlib import Path
from typing import Any, Dict, List

import pytest

from genai_cli import batch


def test_load_batch_requests_supports_object_and_string(tmp_path: Path) -> None:
    ndjson = tmp_path / "requests.ndjson"
    ndjson.write_text('{"id":"1","prompt":"alpha"}\n"beta"\n')

    rows = batch.load_batch_requests(str(ndjson))

    assert len(rows) == 2
    assert rows[0]["prompt"] == "alpha"
    assert rows[0]["_line_number"] == 1
    assert rows[1]["prompt"] == "beta"
    assert rows[1]["_line_number"] == 2


def test_normalize_list_field_accepts_string_or_list() -> None:
    assert batch.normalize_list_field("a.txt", 1, "file_paths") == ["a.txt"]
    assert batch.normalize_list_field(["a.txt", "b.txt"], 1, "file_paths") == ["a.txt", "b.txt"]

    with pytest.raises(ValueError, match="must be string or list"):
        batch.normalize_list_field([1, 2], 1, "file_paths")


def test_run_batch_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_execute_request(**kwargs: Any) -> Dict[str, Any]:
        return {
            "response_text": kwargs["prompt"].upper(),
            "conversation_id": None,
            "model": kwargs["model"],
            "usage_metadata": None,
            "metrics": {"latency_ms": 1.0, "cache_hit": False},
            "validated_json": None,
            "cache_hit": False,
        }

    monkeypatch.setattr(batch, "execute_request", fake_execute_request)

    args = argparse.Namespace(
        model="gemma-3-1b-it",
        api_key=None,
        vertexai=False,
        project=None,
        location="us-central1",
        response_schema=None,
        cache=False,
        cache_ttl=0,
        metrics=False,
        jobs=2,
    )

    requests: List[Dict[str, Any]] = [
        {"_line_number": 1, "id": "a", "prompt": "one"},
        {"_line_number": 2, "id": "b", "prompt": "two"},
    ]

    results = batch.run_batch(requests, args, response_schema=None, base_generation_config=None)

    assert [row["id"] for row in results] == ["a", "b"]
    assert [row["response"] for row in results] == ["ONE", "TWO"]


def test_run_batch_error_includes_error_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_execute_request(**kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr(batch, "execute_request", fake_execute_request)

    args = argparse.Namespace(
        model="gemma-3-1b-it",
        api_key=None,
        vertexai=False,
        project=None,
        location="us-central1",
        response_schema=None,
        cache=False,
        cache_ttl=0,
        metrics=False,
        jobs=1,
    )

    requests = [{"_line_number": 1, "id": "a", "prompt": "one"}]
    results = batch.run_batch(requests, args, response_schema=None, base_generation_config=None)

    assert results[0]["ok"] is False
    assert results[0]["error_type"] == "rate_limit"
