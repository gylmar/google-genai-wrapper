import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from genai_cli import core

TEST_MODEL = "gemma-3-1b-it"


def _credentials_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _integration_enabled() -> bool:
    return os.getenv("RUN_LIVE_INTEGRATION") == "1"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _integration_enabled(), reason="Set RUN_LIVE_INTEGRATION=1 to run live integration tests"),
    pytest.mark.skipif(not _credentials_available(), reason="No API credentials found (GEMINI_API_KEY/GOOGLE_API_KEY)"),
]


def test_live_list_models_includes_default_gemma() -> None:
    models = core.list_available_models(filter_pattern="gemma")

    assert TEST_MODEL in models


def test_live_execute_request_cache_roundtrip() -> None:
    prompt = f"Reply with exactly one word: ok-{uuid.uuid4()}"

    first = core.execute_request(
        prompt=prompt,
        model=TEST_MODEL,
        api_key=None,
        stream=False,
        vertexai=False,
        project=None,
        location="us-central1",
        image_paths=None,
        file_paths=None,
        conversation_id=None,
        generation_config=core.build_generation_config(0.0, None, None, 32, None),
        response_schema=None,
        response_schema_path=None,
        cache_enabled=True,
        cache_ttl=3600,
        stream_handler=None,
    )
    second = core.execute_request(
        prompt=prompt,
        model=TEST_MODEL,
        api_key=None,
        stream=False,
        vertexai=False,
        project=None,
        location="us-central1",
        image_paths=None,
        file_paths=None,
        conversation_id=None,
        generation_config=core.build_generation_config(0.0, None, None, 32, None),
        response_schema=None,
        response_schema_path=None,
        cache_enabled=True,
        cache_ttl=3600,
        stream_handler=None,
    )

    assert first["model"] == TEST_MODEL
    assert isinstance(first["response_text"], str) and first["response_text"].strip()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_live_cli_json_quiet_output() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "call_genai.py"

    cmd = [
        sys.executable,
        str(script_path),
        "Reply with one short word.",
        "--model",
        TEST_MODEL,
        "--format",
        "json",
        "--quiet",
        "--cache",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is True
    assert payload["model"] == TEST_MODEL
    assert isinstance(payload["response"], str)


def test_live_cli_stdin_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "call_genai.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--stdin",
        "--model",
        TEST_MODEL,
        "--quiet",
    ]
    result = subprocess.run(
        cmd,
        input="Reply with exactly one short word.",
        capture_output=True,
        text=True,
        check=True,
    )

    assert isinstance(result.stdout, str)
    assert result.stdout.strip()
