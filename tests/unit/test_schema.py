import pytest

from genai_cli.schema import extract_json_path, parse_and_validate_schema_response, validate_json_schema


def test_validate_json_schema_accepts_required_fields() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    payload = {"answer": "yes"}

    validate_json_schema(payload, schema)


def test_validate_json_schema_rejects_missing_required() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    with pytest.raises(ValueError, match="missing required property"):
        validate_json_schema({}, schema)


def test_parse_and_validate_schema_response_rejects_non_json() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    with pytest.raises(ValueError, match="output is not valid JSON"):
        parse_and_validate_schema_response("not-json", schema, "schema.json")


def test_parse_and_validate_schema_response_returns_parsed_json() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    parsed = parse_and_validate_schema_response('{"answer":"ok"}', schema, "schema.json")

    assert parsed == {"answer": "ok"}


def test_extract_json_path_supports_dot_and_index() -> None:
    payload = {"result": {"items": [{"value": "a"}, {"value": "b"}]}}

    extracted = extract_json_path(payload, "$.result.items[1].value")

    assert extracted == "b"


def test_extract_json_path_rejects_missing_key() -> None:
    payload = {"result": {"value": "ok"}}

    with pytest.raises(ValueError, match="not found"):
        extract_json_path(payload, "$.result.missing")
