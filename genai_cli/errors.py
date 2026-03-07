from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

ERROR_TYPE_AUTH = "auth"
ERROR_TYPE_RATE_LIMIT = "rate_limit"
ERROR_TYPE_TIMEOUT = "timeout"
ERROR_TYPE_SCHEMA_VALIDATION = "schema_validation"
ERROR_TYPE_API = "api_error"


EXIT_CODES: Dict[str, int] = {
    ERROR_TYPE_AUTH: 10,
    ERROR_TYPE_RATE_LIMIT: 11,
    ERROR_TYPE_TIMEOUT: 12,
    ERROR_TYPE_SCHEMA_VALIDATION: 13,
    ERROR_TYPE_API: 14,
}


@dataclass
class TypedError(Exception):
    message: str
    error_type: str

    def __str__(self) -> str:
        return self.message


class TimeoutRequestError(TypedError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_type=ERROR_TYPE_TIMEOUT)


class SchemaValidationError(TypedError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_type=ERROR_TYPE_SCHEMA_VALIDATION)


def classify_exception(exc: Exception) -> str:
    if isinstance(exc, TypedError):
        return exc.error_type

    message = str(exc).lower()

    auth_markers = [
        "api key",
        "invalid api key",
        "unauthorized",
        "permission_denied",
        "permission denied",
        "authentication",
        "access denied",
        "401",
        "403",
    ]
    if any(marker in message for marker in auth_markers):
        return ERROR_TYPE_AUTH

    rate_limit_markers = [
        "rate limit",
        "resource_exhausted",
        "too many requests",
        "quota exceeded",
        "429",
    ]
    if any(marker in message for marker in rate_limit_markers):
        return ERROR_TYPE_RATE_LIMIT

    timeout_markers = [
        "timed out",
        "timeout",
        "deadline exceeded",
    ]
    if any(marker in message for marker in timeout_markers):
        return ERROR_TYPE_TIMEOUT

    schema_markers = [
        "schema validation failed",
        "missing required property",
        "additional property",
        "expected type",
    ]
    if any(marker in message for marker in schema_markers):
        return ERROR_TYPE_SCHEMA_VALIDATION

    return ERROR_TYPE_API


def is_retryable_exception(exc: Exception) -> bool:
    error_type = classify_exception(exc)
    if error_type in {ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_TIMEOUT}:
        return True

    # Network and backend transient failures that are not explicit rate limits.
    transient_markers = [
        "temporarily unavailable",
        "service unavailable",
        "internal",
        "connection reset",
        "connection aborted",
        "connection error",
        "network error",
        "try again",
        "unavailable",
        "503",
    ]
    lowered = str(exc).lower()
    return error_type == ERROR_TYPE_API and any(marker in lowered for marker in transient_markers)


def exit_code_for_exception(exc: Exception) -> int:
    return EXIT_CODES.get(classify_exception(exc), EXIT_CODES[ERROR_TYPE_API])


def exit_code_for_error_type(error_type: str) -> int:
    return EXIT_CODES.get(error_type, EXIT_CODES[ERROR_TYPE_API])
