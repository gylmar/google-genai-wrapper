import io
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


def test_main_reads_prompt_from_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured = {}

    def fake_execute_request(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return {
            "response_text": "done",
            "conversation_id": None,
            "model": "gemma-3-1b-it",
            "usage_metadata": None,
            "metrics": {"latency_ms": 1.0, "cache_hit": False},
            "validated_json": None,
            "cache_hit": False,
        }

    monkeypatch.setattr(cli, "execute_request", fake_execute_request)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin prompt"))
    monkeypatch.setattr(sys, "argv", ["call_genai.py", "--stdin", "--model", "gemma-3-1b-it", "--quiet"])

    cli.main()

    assert captured["prompt"] == "stdin prompt"
    assert capsys.readouterr().out.strip() == "done"


def test_main_profile_applies_defaults(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured = {}

    def fake_execute_request(**kwargs):
        captured["generation_config"] = kwargs["generation_config"]
        return {
            "response_text": "ok",
            "conversation_id": None,
            "model": "gemma-3-1b-it",
            "usage_metadata": None,
            "metrics": {"latency_ms": 1.0, "cache_hit": False},
            "validated_json": None,
            "cache_hit": False,
        }

    monkeypatch.setattr(cli, "execute_request", fake_execute_request)
    monkeypatch.setattr(
        sys,
        "argv",
        ["call_genai.py", "ping", "--model", "gemma-3-1b-it", "--profile", "classify", "--format", "json", "--quiet"],
    )

    cli.main()

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["ok"] is True
    assert captured["generation_config"]["temperature"] == 0.0
    assert captured["generation_config"]["top_p"] == 0.1
    assert captured["generation_config"]["max_output_tokens"] == 128


def test_main_json_path_requires_schema(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["call_genai.py", "hello", "--json-path", "$.x", "--model", "gemma-3-1b-it"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "--json-path requires --response-schema" in capsys.readouterr().err


def test_main_quiet_text_uses_selected_json_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        '{"type":"object","properties":{"result":{"type":"string"}},"required":["result"],"additionalProperties":false}'
    )

    monkeypatch.setattr(
        cli,
        "execute_request",
        lambda **_: {
            "response_text": '{"result":"ok"}',
            "conversation_id": None,
            "model": "gemma-3-1b-it",
            "usage_metadata": None,
            "metrics": {"latency_ms": 1.0, "cache_hit": False},
            "validated_json": {"result": "ok"},
            "selected_json": "ok",
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "call_genai.py",
            "hello",
            "--model",
            "gemma-3-1b-it",
            "--response-schema",
            str(schema_path),
            "--json-path",
            "$.result",
            "--quiet",
        ],
    )

    cli.main()

    assert capsys.readouterr().out.strip() == "ok"


def test_main_maps_auth_error_to_exit_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "execute_request", lambda **_: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")))
    monkeypatch.setattr(sys, "argv", ["call_genai.py", "hello", "--model", "gemma-3-1b-it"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 10
    assert "Error: 401 Unauthorized" in capsys.readouterr().err


def test_main_prompt_file_and_instruction_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("file prompt")
    instruction_path = tmp_path / "instruction.txt"
    instruction_path.write_text("be precise")
    captured = {}

    def fake_execute_request(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["generation_config"] = kwargs["generation_config"]
        return {
            "response_text": "ok",
            "conversation_id": None,
            "model": "gemma-3-1b-it",
            "usage_metadata": None,
            "metrics": {"latency_ms": 1.0, "cache_hit": False},
            "validated_json": None,
            "cache_hit": False,
        }

    monkeypatch.setattr(cli, "execute_request", fake_execute_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "call_genai.py",
            "--prompt-file",
            str(prompt_path),
            "--instruction-file",
            str(instruction_path),
            "--system",
            "return json",
            "--model",
            "gemma-3-1b-it",
            "--quiet",
        ],
    )

    cli.main()

    assert captured["prompt"] == "file prompt"
    assert captured["generation_config"]["system_instruction"] == "be precise\nreturn json"
    assert capsys.readouterr().out.strip() == "ok"
