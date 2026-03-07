import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from genai_cli import core


def test_build_generation_config_includes_sampling_and_schema() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    cfg = core.build_generation_config(0.2, 0.9, 40, 128, schema)

    assert cfg == {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 128,
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }


def test_build_cache_key_changes_with_prompt(tmp_path: Path) -> None:
    key1 = core.build_cache_key(
        model="gemma-3-1b-it",
        prompt="a",
        generation_config={"temperature": 0.1},
        vertexai=False,
        project=None,
        location="us-central1",
        image_paths=None,
        file_paths=None,
    )
    key2 = core.build_cache_key(
        model="gemma-3-1b-it",
        prompt="b",
        generation_config={"temperature": 0.1},
        vertexai=False,
        project=None,
        location="us-central1",
        image_paths=None,
        file_paths=None,
    )

    assert key1 != key2


def test_cache_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)

    payload = {"response_text": "hello", "model": "gemma-3-1b-it"}
    cache_key = "abc123"
    core.write_cached_response(cache_key, payload)

    cached = core.read_cached_response(cache_key, cache_ttl=60)

    assert cached is not None
    assert cached["response_text"] == "hello"
    assert cached["model"] == "gemma-3-1b-it"
    assert "cache_age_seconds" in cached


def test_build_metrics_includes_usage_fields() -> None:
    usage = {
        "prompt_token_count": 10,
        "candidates_token_count": 5,
        "total_token_count": 15,
        "cached_content_token_count": 0,
    }

    metrics = core.build_metrics(123.456, usage, cache_hit=False)

    assert metrics["latency_ms"] == 123.46
    assert metrics["prompt_tokens"] == 10
    assert metrics["output_tokens"] == 5
    assert metrics["total_tokens"] == 15


def test_execute_request_retries_without_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_call_genai(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        config = kwargs.get("generation_config") or {}
        if "response_json_schema" in config:
            raise RuntimeError("400 INVALID_ARGUMENT: JSON mode is not enabled")
        return {
            "response_text": '{"answer":"yes"}',
            "conversation_id": None,
            "model": kwargs["model"],
            "usage_metadata": None,
            "latency_ms": 5.0,
        }

    monkeypatch.setattr(core, "call_genai", fake_call_genai)

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    result = core.execute_request(
        prompt="return json",
        model="gemma-3-1b-it",
        api_key=None,
        stream=False,
        vertexai=False,
        project=None,
        location="us-central1",
        image_paths=None,
        file_paths=None,
        conversation_id=None,
        generation_config={"response_mime_type": "application/json", "response_json_schema": schema},
        response_schema=schema,
        response_schema_path="schema.json",
        cache_enabled=False,
        cache_ttl=0,
        stream_handler=None,
    )

    assert len(calls) == 2
    assert "response_json_schema" in (calls[0].get("generation_config") or {})
    assert "response_json_schema" not in (calls[1].get("generation_config") or {})
    assert result["validated_json"] == {"answer": "yes"}
    assert result["cache_hit"] is False
