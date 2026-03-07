import json
import sys

import pytest

from genai_cli import cli


def test_main_rejects_stream_with_json_format(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["call_genai.py", "hello", "--stream", "--format", "json"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--stream is only supported with --format text" in stderr


def test_main_list_models_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "list_available_models", lambda **_: ["gemma-3-1b-it", "gemma-3-4b-it"])
    monkeypatch.setattr(sys, "argv", ["call_genai.py", "--list-models", "--format", "json", "--quiet"])

    cli.main()

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["models"] == ["gemma-3-1b-it", "gemma-3-4b-it"]


def test_main_single_request_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        cli,
        "execute_request",
        lambda **_: {
            "response_text": "ok",
            "conversation_id": None,
            "model": "gemma-3-1b-it",
            "usage_metadata": None,
            "metrics": {"latency_ms": 1.0, "cache_hit": False},
            "validated_json": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(sys, "argv", ["call_genai.py", "ping", "--model", "gemma-3-1b-it", "--format", "json", "--quiet"])

    cli.main()

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["model"] == "gemma-3-1b-it"
    assert payload["response"] == "ok"
