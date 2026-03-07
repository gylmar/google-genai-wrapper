import json
from pathlib import Path
from typing import Any, Optional


def load_json_file(path: str, description: str) -> Any:
    """Load a JSON file and raise a clear error if parsing fails."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} file not found: {path}")

    try:
        with open(file_path, "r") as file_handle:
            return json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description} file '{path}': {exc}") from exc


def _json_type_matches(value: Any, expected_type: str) -> bool:
    """Minimal type matcher for JSON Schema validation."""
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def validate_json_schema(value: Any, schema: Any, path: str = "$") -> None:
    """Validate a value against a practical JSON Schema subset."""
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")
    if schema_type is not None:
        allowed_types = schema_type if isinstance(schema_type, list) else [schema_type]
        if not any(_json_type_matches(value, t) for t in allowed_types):
            raise ValueError(f"{path}: expected type {allowed_types}, got {type(value).__name__}")

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value must equal const {schema['const']!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value {value!r} is not in enum {schema['enum']!r}")

    if "anyOf" in schema:
        errors = []
        for sub_schema in schema["anyOf"]:
            try:
                validate_json_schema(value, sub_schema, path)
                break
            except ValueError as exc:
                errors.append(str(exc))
        else:
            raise ValueError(f"{path}: failed anyOf validation ({'; '.join(errors)})")

    if "oneOf" in schema:
        success_count = 0
        errors = []
        for sub_schema in schema["oneOf"]:
            try:
                validate_json_schema(value, sub_schema, path)
                success_count += 1
            except ValueError as exc:
                errors.append(str(exc))
        if success_count != 1:
            raise ValueError(f"{path}: oneOf requires exactly one match (errors: {'; '.join(errors)})")

    if "allOf" in schema:
        for sub_schema in schema["allOf"]:
            validate_json_schema(value, sub_schema, path)

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}: missing required property '{key}'")

        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in value:
                validate_json_schema(value[key], prop_schema, f"{path}.{key}")

        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            allowed = set(properties.keys())
            for key in value:
                if key not in allowed:
                    raise ValueError(f"{path}: additional property '{key}' is not allowed")
        elif isinstance(additional_properties, dict):
            allowed = set(properties.keys())
            for key, sub_value in value.items():
                if key not in allowed:
                    validate_json_schema(sub_value, additional_properties, f"{path}.{key}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            raise ValueError(f"{path}: expected at least {min_items} items, got {len(value)}")

        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise ValueError(f"{path}: expected at most {max_items} items, got {len(value)}")

        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                validate_json_schema(item, item_schema, f"{path}[{index}]")


def parse_and_validate_schema_response(
    response_text: str,
    response_schema: Optional[dict],
    response_schema_path: Optional[str],
) -> Optional[Any]:
    """Parse and validate a JSON response against the schema file."""
    if response_schema is None:
        return None

    schema_label = response_schema_path or "<inline-schema>"
    try:
        parsed_json = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Response schema validation failed ({schema_label}): output is not valid JSON: {exc}"
        ) from exc

    validate_json_schema(parsed_json, response_schema)
    return parsed_json
