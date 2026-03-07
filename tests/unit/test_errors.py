from genai_cli.errors import (
    ERROR_TYPE_API,
    ERROR_TYPE_AUTH,
    ERROR_TYPE_RATE_LIMIT,
    ERROR_TYPE_SCHEMA_VALIDATION,
    ERROR_TYPE_TIMEOUT,
    classify_exception,
    exit_code_for_error_type,
)


def test_classify_exception_detects_auth() -> None:
    exc = RuntimeError("401 Unauthorized: invalid API key")
    assert classify_exception(exc) == ERROR_TYPE_AUTH


def test_classify_exception_detects_rate_limit() -> None:
    exc = RuntimeError("429 RESOURCE_EXHAUSTED")
    assert classify_exception(exc) == ERROR_TYPE_RATE_LIMIT


def test_classify_exception_detects_timeout() -> None:
    exc = RuntimeError("request timed out")
    assert classify_exception(exc) == ERROR_TYPE_TIMEOUT


def test_classify_exception_detects_schema() -> None:
    exc = ValueError("Response schema validation failed (schema.json): output is not valid JSON")
    assert classify_exception(exc) == ERROR_TYPE_SCHEMA_VALIDATION


def test_classify_exception_defaults_to_api_error() -> None:
    exc = RuntimeError("unknown backend issue")
    assert classify_exception(exc) == ERROR_TYPE_API


def test_exit_codes_are_distinct() -> None:
    codes = {
        exit_code_for_error_type(ERROR_TYPE_AUTH),
        exit_code_for_error_type(ERROR_TYPE_RATE_LIMIT),
        exit_code_for_error_type(ERROR_TYPE_TIMEOUT),
        exit_code_for_error_type(ERROR_TYPE_SCHEMA_VALIDATION),
        exit_code_for_error_type(ERROR_TYPE_API),
    }
    assert len(codes) == 5
