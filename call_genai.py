#!/usr/bin/env python3
"""
Script to call Google Generative AI models (Gemini, Gemma, etc.) using the Google Generative AI SDK.

Supports all Google Generative AI models including Gemini and Gemma model families.

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY  # Required for Gemini Developer API (unless --api-key is passed)
    GEMINI_MODEL                      # Optional default model name (e.g. gemini-2.5-flash, gemma-3-27b-it)
    GOOGLE_CLOUD_PROJECT              # Required for Vertex AI mode (--vertexai)
    GOOGLE_CLOUD_LOCATION             # Optional for Vertex AI (defaults to us-central1)
"""

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Error: google-genai package not found. Please install it with: pip install google-genai")
    sys.exit(1)

# Try to import PIL for image support
try:
    import PIL.Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Directory for storing conversation history
CONVERSATIONS_DIR = Path.home() / ".genai_conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Directory for cached responses
CACHE_DIR = Path.home() / ".genai_cache"
CACHE_DIR.mkdir(exist_ok=True)


def to_jsonable(value: Any) -> Any:
    """Convert SDK/pydantic/enum types into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            return str(value)
    return str(value)


def json_dumps(data: Any, pretty: bool = False) -> str:
    """Dump JSON consistently while preserving unicode."""
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def get_client(
    api_key: Optional[str] = None,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> genai.Client:
    """
    Initialize and return a Google Generative AI client.

    Args:
        api_key: API key for Gemini Developer API (or use GEMINI_API_KEY / GOOGLE_API_KEY env var)
        vertexai: Whether to use Vertex AI API instead of Gemini Developer API
        project: Google Cloud project ID (for Vertex AI)
        location: Google Cloud location (for Vertex AI)

    Returns:
        Initialized client
    """
    if vertexai:
        return genai.Client(
            vertexai=True,
            project=project or os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Provide --api-key or set GEMINI_API_KEY / GOOGLE_API_KEY."
        )
    return genai.Client(api_key=api_key)


def list_available_models(
    api_key: Optional[str] = None,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    filter_pattern: Optional[str] = None,
) -> List[str]:
    """
    List all available models from the API.

    Args:
        api_key: API key for Gemini Developer API (or use GEMINI_API_KEY env var)
        vertexai: Whether to use Vertex AI API instead of Gemini Developer API
        project: Google Cloud project ID (for Vertex AI)
        location: Google Cloud location (for Vertex AI)
        filter_pattern: Optional pattern to filter model names (e.g., 'gemma', 'gemini')

    Returns:
        List of available model names
    """
    client = get_client(api_key, vertexai, project, location)

    try:
        models = []
        pager = client.models.list(config={"page_size": 100, "query_base": True})

        def _iter_models(entry: Any) -> Sequence[Any]:
            if hasattr(entry, "models") and entry.models:
                return entry.models
            if isinstance(entry, list):
                return entry
            return [entry]

        for entry in pager:
            for model in _iter_models(entry):
                if not model or not hasattr(model, "name") or not model.name:
                    continue

                model_name = model.name
                if "/" in model_name:
                    model_name = model_name.split("/")[-1]

                if not filter_pattern or filter_pattern.lower() in model_name.lower():
                    models.append(model_name)

        return sorted(set(models))
    except Exception as exc:
        print(f"Error listing models: {exc}", file=sys.stderr)
        raise


def prepare_contents(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
) -> List[Union[str, genai_types.Part]]:
    """
    Prepare contents for the API call, including text, images, and files.

    Args:
        prompt: The text prompt
        image_paths: List of image file paths to include
        file_paths: List of document file paths to include

    Returns:
        List of content parts (text and/or media)
    """
    contents: List[Union[str, genai_types.Part]] = []

    if image_paths:
        for image_path in image_paths:
            img_path = Path(image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            if PIL_AVAILABLE:
                try:
                    img = PIL.Image.open(img_path)
                    contents.append(genai_types.Part(img))
                    continue
                except Exception:
                    pass

            with open(img_path, "rb") as image_file:
                image_data = image_file.read()

            mime_type, _ = mimetypes.guess_type(str(img_path))
            if not mime_type or not mime_type.startswith("image/"):
                ext = img_path.suffix.lower()
                mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                }
                mime_type = mime_map.get(ext, "image/jpeg")

            contents.append(genai_types.Part.from_bytes(data=image_data, mime_type=mime_type))

    if file_paths:
        for file_path in file_paths:
            file_p = Path(file_path)
            if not file_p.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            mime_type, _ = mimetypes.guess_type(str(file_p))
            if not mime_type:
                ext = file_p.suffix.lower()
                mime_map = {
                    ".pdf": "application/pdf",
                    ".txt": "text/plain",
                    ".md": "text/markdown",
                    ".csv": "text/csv",
                    ".json": "application/json",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".doc": "application/msword",
                }
                mime_type = mime_map.get(ext, "application/octet-stream")

            with open(file_p, "rb") as document_file:
                file_data = document_file.read()

            contents.append(genai_types.Part.from_bytes(data=file_data, mime_type=mime_type))

    if prompt:
        contents.append(prompt)

    return contents


def save_conversation(conversation_id: str, history: List[genai_types.Content], model: str) -> None:
    """Save conversation history to a file."""
    conversation_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    data = {
        "conversation_id": conversation_id,
        "model": model,
        "history": [content.model_dump() for content in history],
    }
    with open(conversation_file, "w") as history_file:
        json.dump(data, history_file, indent=2, default=str)


def load_conversation(conversation_id: str) -> Optional[dict]:
    """Load conversation history from a file."""
    conversation_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if not conversation_file.exists():
        return None

    try:
        with open(conversation_file, "r") as history_file:
            data = json.load(history_file)
        if "history" in data:
            data["history"] = [genai_types.Content.model_validate(h) for h in data["history"]]
        return data
    except Exception as exc:
        print(f"Warning: Could not load conversation {conversation_id}: {exc}", file=sys.stderr)
        return None


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


def build_generation_config(
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_output_tokens: Optional[int],
    response_schema: Optional[dict],
) -> Optional[Dict[str, Any]]:
    """Build a GenerateContent config dict from CLI values."""
    config: Dict[str, Any] = {}

    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["top_p"] = top_p
    if top_k is not None:
        config["top_k"] = top_k
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens

    if response_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = response_schema

    return config or None


def model_response_text(response: Optional[genai_types.GenerateContentResponse], fallback: str = "") -> str:
    """Extract text from a model response object."""
    if response is None:
        return fallback

    text = getattr(response, "text", None)
    if text:
        return text

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return json_dumps(to_jsonable(parsed), pretty=False)

    return fallback


def extract_usage_metadata(response: Optional[genai_types.GenerateContentResponse]) -> Optional[Dict[str, Any]]:
    """Extract usage metadata from a response object if present."""
    if response is None:
        return None

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        return to_jsonable(usage.model_dump())
    if isinstance(usage, dict):
        return to_jsonable(usage)
    return {"raw": str(usage)}


def build_metrics(
    latency_ms: float,
    usage_metadata: Optional[Dict[str, Any]],
    cache_hit: bool,
    cache_age_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Normalize runtime metrics for output."""
    metrics: Dict[str, Any] = {
        "latency_ms": round(latency_ms, 2),
        "cache_hit": cache_hit,
    }

    if cache_age_seconds is not None:
        metrics["cache_age_seconds"] = round(cache_age_seconds, 2)

    if usage_metadata:
        metrics["prompt_tokens"] = usage_metadata.get("prompt_token_count")
        metrics["output_tokens"] = usage_metadata.get("candidates_token_count")
        metrics["total_tokens"] = usage_metadata.get("total_token_count")
        metrics["cached_content_tokens"] = usage_metadata.get("cached_content_token_count")
        metrics["usage_metadata"] = usage_metadata

    return metrics


def file_fingerprint(path: str) -> Dict[str, Any]:
    """
    Build a stable fingerprint from file path + stat metadata.
    """
    file_path = Path(path)
    resolved = str(file_path.resolve())
    if not file_path.exists():
        return {"path": resolved, "exists": False}

    stats = file_path.stat()
    return {
        "path": resolved,
        "exists": True,
        "size": stats.st_size,
        "mtime_ns": stats.st_mtime_ns,
    }


def build_cache_key(
    model: str,
    prompt: str,
    generation_config: Optional[Dict[str, Any]],
    vertexai: bool,
    project: Optional[str],
    location: Optional[str],
    image_paths: Optional[List[str]],
    file_paths: Optional[List[str]],
) -> str:
    """
    Hash request inputs into a deterministic cache key.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "generation_config": to_jsonable(generation_config or {}),
        "vertexai": vertexai,
        "project": project,
        "location": location,
        "images": [file_fingerprint(path) for path in (image_paths or [])],
        "files": [file_fingerprint(path) for path in (file_paths or [])],
    }
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    return digest


def read_cached_response(cache_key: str, cache_ttl: int) -> Optional[Dict[str, Any]]:
    """Return cached payload if present and still fresh."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as cache_handle:
            cached = json.load(cache_handle)
    except Exception:
        return None

    created_at = cached.get("created_at")
    if not isinstance(created_at, (int, float)):
        return None

    age_seconds = time.time() - created_at
    if age_seconds > cache_ttl:
        return None

    payload = cached.get("payload")
    if not isinstance(payload, dict):
        return None

    payload["cache_age_seconds"] = age_seconds
    return payload


def write_cached_response(cache_key: str, payload: Dict[str, Any]) -> None:
    """Persist response payload to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    record = {"created_at": time.time(), "payload": to_jsonable(payload)}
    with open(cache_file, "w") as cache_handle:
        json.dump(record, cache_handle, indent=2, ensure_ascii=False)


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
    """
    Validate a value against a practical JSON Schema subset.
    """
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
    """
    Parse and validate a JSON response against the schema file.
    """
    if response_schema is None:
        return None

    schema_label = response_schema_path or "<inline-schema>"
    try:
        parsed_json = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response schema validation failed ({schema_label}): output is not valid JSON: {exc}") from exc

    validate_json_schema(parsed_json, response_schema)
    return parsed_json


def call_genai(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    stream: bool = False,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    stream_handler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Call a Google Generative AI model and return normalized response data.
    """
    client = get_client(api_key, vertexai, project, location)
    contents = prepare_contents(prompt, image_paths, file_paths)
    use_chat = conversation_id is not None

    started_at = time.perf_counter()
    usage_metadata: Optional[Dict[str, Any]] = None

    if use_chat:
        conversation_data = load_conversation(conversation_id) if conversation_id else None

        if conversation_data:
            history = conversation_data.get("history", [])
            if "model" in conversation_data:
                model = conversation_data["model"]
        else:
            history = []
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

        chat = client.chats.create(model=model, history=history)

        try:
            if stream:
                full_response = ""
                last_chunk: Optional[genai_types.GenerateContentResponse] = None
                for chunk in chat.send_message_stream(contents, config=generation_config):
                    last_chunk = chunk
                    if chunk.text:
                        if stream_handler:
                            stream_handler(chunk.text)
                        full_response += chunk.text
                response_text = full_response
                usage_metadata = extract_usage_metadata(last_chunk)
            else:
                response = chat.send_message(contents, config=generation_config)
                response_text = model_response_text(response)
                usage_metadata = extract_usage_metadata(response)

            save_conversation(conversation_id, chat.get_history(curated=True), model)
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            return {
                "response_text": response_text,
                "conversation_id": conversation_id,
                "model": model,
                "usage_metadata": usage_metadata,
                "latency_ms": latency_ms,
            }
        except Exception:
            raise

    try:
        if stream:
            full_response = ""
            last_chunk: Optional[genai_types.GenerateContentResponse] = None
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generation_config,
            ):
                last_chunk = chunk
                if chunk.text:
                    if stream_handler:
                        stream_handler(chunk.text)
                    full_response += chunk.text
            response_text = full_response
            usage_metadata = extract_usage_metadata(last_chunk)
        else:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generation_config,
            )
            response_text = model_response_text(response)
            usage_metadata = extract_usage_metadata(response)

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "response_text": response_text,
            "conversation_id": None,
            "model": model,
            "usage_metadata": usage_metadata,
            "latency_ms": latency_ms,
        }
    except Exception:
        raise


def execute_request(
    prompt: str,
    model: str,
    api_key: Optional[str],
    stream: bool,
    vertexai: bool,
    project: Optional[str],
    location: Optional[str],
    image_paths: Optional[List[str]],
    file_paths: Optional[List[str]],
    conversation_id: Optional[str],
    generation_config: Optional[Dict[str, Any]],
    response_schema: Optional[dict],
    response_schema_path: Optional[str],
    cache_enabled: bool,
    cache_ttl: int,
    stream_handler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute one model request with optional cache lookup.
    """
    cache_key: Optional[str] = None
    if cache_enabled and not stream and conversation_id is None:
        cache_key = build_cache_key(
            model=model,
            prompt=prompt,
            generation_config=generation_config,
            vertexai=vertexai,
            project=project,
            location=location,
            image_paths=image_paths,
            file_paths=file_paths,
        )
        cached = read_cached_response(cache_key, cache_ttl)
        if cached:
            response_text = cached.get("response_text", "")
            validated = parse_and_validate_schema_response(
                response_text=response_text,
                response_schema=response_schema,
                response_schema_path=response_schema_path,
            )
            usage_metadata = cached.get("usage_metadata")
            metrics = build_metrics(
                latency_ms=0.0,
                usage_metadata=usage_metadata,
                cache_hit=True,
                cache_age_seconds=cached.get("cache_age_seconds"),
            )
            return {
                "response_text": response_text,
                "conversation_id": cached.get("conversation_id"),
                "model": cached.get("model", model),
                "usage_metadata": usage_metadata,
                "metrics": metrics,
                "validated_json": validated,
                "cache_hit": True,
            }

    effective_config = dict(generation_config or {})
    try:
        result = call_genai(
            prompt=prompt,
            model=model,
            api_key=api_key,
            stream=stream,
            vertexai=vertexai,
            project=project,
            location=location,
            image_paths=image_paths,
            file_paths=file_paths,
            conversation_id=conversation_id,
            generation_config=effective_config or None,
            stream_handler=stream_handler,
        )
    except Exception as exc:
        # Some models (including Gemma variants) reject API-level JSON mode. Retry once
        # without JSON-mode config and still enforce schema validation locally.
        error_text = str(exc).lower()
        has_json_mode = "response_json_schema" in effective_config or "response_mime_type" in effective_config
        if response_schema is None or not has_json_mode:
            raise
        if "json mode is not enabled" not in error_text and "invalid_argument" not in error_text:
            raise

        fallback_config = dict(effective_config)
        fallback_config.pop("response_json_schema", None)
        fallback_config.pop("response_mime_type", None)
        result = call_genai(
            prompt=prompt,
            model=model,
            api_key=api_key,
            stream=stream,
            vertexai=vertexai,
            project=project,
            location=location,
            image_paths=image_paths,
            file_paths=file_paths,
            conversation_id=conversation_id,
            generation_config=fallback_config or None,
            stream_handler=stream_handler,
        )

    validated_json = parse_and_validate_schema_response(
        response_text=result["response_text"],
        response_schema=response_schema,
        response_schema_path=response_schema_path,
    )

    if cache_key:
        write_cached_response(
            cache_key=cache_key,
            payload={
                "response_text": result["response_text"],
                "conversation_id": result["conversation_id"],
                "model": result["model"],
                "usage_metadata": result["usage_metadata"],
            },
        )

    metrics = build_metrics(
        latency_ms=result["latency_ms"],
        usage_metadata=result["usage_metadata"],
        cache_hit=False,
    )

    return {
        "response_text": result["response_text"],
        "conversation_id": result["conversation_id"],
        "model": result["model"],
        "usage_metadata": result["usage_metadata"],
        "metrics": metrics,
        "validated_json": validated_json,
        "cache_hit": False,
    }


def normalize_batch_item(item: Any, line_number: int) -> Dict[str, Any]:
    """Normalize one NDJSON line into a request dict."""
    if isinstance(item, str):
        return {"prompt": item}
    if isinstance(item, dict):
        return item
    raise ValueError(f"Batch line {line_number}: expected JSON object or string, got {type(item).__name__}")


def load_batch_requests(batch_path: str) -> List[Dict[str, Any]]:
    """Load NDJSON requests from disk."""
    path = Path(batch_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    requests: List[Dict[str, Any]] = []
    with open(path, "r") as batch_file:
        for line_number, raw_line in enumerate(batch_file, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Batch line {line_number}: invalid JSON: {exc}") from exc
            request = normalize_batch_item(parsed, line_number)
            request["_line_number"] = line_number
            requests.append(request)
    return requests


def normalize_list_field(value: Any, line_number: int, field_name: str) -> Optional[List[str]]:
    """Normalize batch line fields that may be string or list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"Batch line {line_number}: '{field_name}' must be string or list of strings")


def batch_item_generation_config(
    base_config: Optional[Dict[str, Any]],
    item: Dict[str, Any],
    response_schema: Optional[dict],
) -> Optional[Dict[str, Any]]:
    """Merge per-item overrides into generation config."""
    config = dict(base_config or {})
    if "temperature" in item and item["temperature"] is not None:
        config["temperature"] = item["temperature"]
    if "top_p" in item and item["top_p"] is not None:
        config["top_p"] = item["top_p"]
    if "top_k" in item and item["top_k"] is not None:
        config["top_k"] = item["top_k"]
    if "max_output_tokens" in item and item["max_output_tokens"] is not None:
        config["max_output_tokens"] = item["max_output_tokens"]

    if response_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = response_schema

    return config or None


def single_output_record(result: Dict[str, Any], include_metrics: bool) -> Dict[str, Any]:
    """Build one structured response object."""
    record: Dict[str, Any] = {
        "ok": True,
        "model": result["model"],
        "response": result["response_text"],
        "cache_hit": result["cache_hit"],
    }
    if result.get("conversation_id"):
        record["conversation_id"] = result["conversation_id"]
    if include_metrics:
        record["metrics"] = result["metrics"]
    if result.get("validated_json") is not None:
        record["validated_json"] = result["validated_json"]
    return record


def run_batch(
    requests: List[Dict[str, Any]],
    args: argparse.Namespace,
    response_schema: Optional[dict],
    base_generation_config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Execute NDJSON batch concurrently."""

    def _run_item(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        line_number = item.get("_line_number", index + 1)
        try:
            prompt = item.get("prompt")
            if not prompt or not isinstance(prompt, str):
                raise ValueError(f"Batch line {line_number}: missing 'prompt' string")

            model = item.get("model") or args.model
            if not model:
                raise ValueError(
                    f"Batch line {line_number}: missing model (provide line 'model' or global --model)"
                )

            image_paths = normalize_list_field(
                item.get("image_paths", item.get("images", item.get("image"))),
                line_number,
                "image_paths",
            )
            file_paths = normalize_list_field(
                item.get("file_paths", item.get("files", item.get("file"))),
                line_number,
                "file_paths",
            )
            conversation_id = item.get("conversation_id")
            if conversation_id is not None and not isinstance(conversation_id, str):
                raise ValueError(f"Batch line {line_number}: 'conversation_id' must be a string")

            generation_config = batch_item_generation_config(
                base_config=base_generation_config,
                item=item,
                response_schema=response_schema,
            )

            result = execute_request(
                prompt=prompt,
                model=model,
                api_key=args.api_key,
                stream=False,
                vertexai=args.vertexai,
                project=args.project,
                location=args.location,
                image_paths=image_paths,
                file_paths=file_paths,
                conversation_id=conversation_id,
                generation_config=generation_config,
                response_schema=response_schema,
                response_schema_path=args.response_schema,
                cache_enabled=args.cache,
                cache_ttl=args.cache_ttl,
            )
            output = single_output_record(result, include_metrics=args.metrics)
            output["line"] = line_number
            if "id" in item:
                output["id"] = item["id"]
            return output
        except Exception as exc:
            output = {
                "ok": False,
                "line": line_number,
                "error": str(exc),
            }
            if "id" in item:
                output["id"] = item["id"]
            return output

    results: List[Dict[str, Any]] = [None] * len(requests)  # type: ignore[assignment]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(_run_item, idx, request): idx for idx, request in enumerate(requests)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return results


def print_models(models: List[str], args: argparse.Namespace) -> None:
    """Print model list in requested format."""
    if args.format == "json":
        payload = {"models": models}
        print(json_dumps(payload, pretty=not args.quiet))
        return
    if args.format == "ndjson":
        for model in models:
            print(json_dumps({"model": model}, pretty=False))
        return

    if args.quiet:
        for model in models:
            print(model)
        return

    if args.filter:
        print(f"Available models (filtered by '{args.filter}'):")
    else:
        print("All available models:")

    if models:
        for model in models:
            print(f"  - {model}")
    else:
        print("No models found. Make sure your API key is set correctly.")


def print_metrics_text(metrics: Dict[str, Any], quiet: bool) -> None:
    """Print metrics in text mode."""
    if quiet:
        print(json_dumps({"metrics": metrics}, pretty=False), file=sys.stderr)
        return

    print("\nMetrics:")
    print(f"  latency_ms: {metrics.get('latency_ms')}")
    print(f"  cache_hit: {metrics.get('cache_hit')}")
    if "cache_age_seconds" in metrics:
        print(f"  cache_age_seconds: {metrics.get('cache_age_seconds')}")
    if metrics.get("prompt_tokens") is not None:
        print(f"  prompt_tokens: {metrics.get('prompt_tokens')}")
    if metrics.get("output_tokens") is not None:
        print(f"  output_tokens: {metrics.get('output_tokens')}")
    if metrics.get("total_tokens") is not None:
        print(f"  total_tokens: {metrics.get('total_tokens')}")
    if metrics.get("cached_content_tokens") is not None:
        print(f"  cached_content_tokens: {metrics.get('cached_content_tokens')}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Call Google Generative AI models (Gemini, Gemma, etc.) using Google Generative AI SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  GEMINI_API_KEY or GOOGLE_API_KEY  Required for Gemini Developer API (unless --api-key is passed)
  GEMINI_MODEL                      Optional default model name (e.g. gemini-2.5-flash, gemma-3-27b-it)
  GOOGLE_CLOUD_PROJECT              Required for Vertex AI mode (--vertexai)
  GOOGLE_CLOUD_LOCATION             Optional for Vertex AI (defaults to us-central1)

Examples:
  python call_genai.py "Explain quantum computing"
  python call_genai.py "Write a poem" --model gemma-3-27b-it --temperature 0.2 --max-output-tokens 256
  python call_genai.py "Extract fields" --response-schema schema.json --format json --quiet
  python call_genai.py --batch requests.ndjson --model gemma-3-1b-it --jobs 8 --format ndjson
  python call_genai.py "Hi" --cache --cache-ttl 3600 --metrics --quiet
  python call_genai.py --list-models --filter gemma
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The text prompt to send to the model",
    )

    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("GEMINI_MODEL"),
        help="Model to use (e.g., 'gemini-2.5-flash', 'gemma-3-27b-it'). Defaults to GEMINI_MODEL if set.",
    )

    parser.add_argument(
        "--api-key",
        "-k",
        help="API key for Gemini Developer API (or set GEMINI_API_KEY / GOOGLE_API_KEY)",
    )

    parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Stream the response as it's generated",
    )

    parser.add_argument(
        "--vertexai",
        action="store_true",
        help="Use Vertex AI API instead of Gemini Developer API",
    )

    parser.add_argument(
        "--project",
        help="Google Cloud project ID (required for Vertex AI)",
    )

    parser.add_argument(
        "--location",
        default="us-central1",
        help="Google Cloud location (for Vertex AI, default: us-central1)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from the API and exit",
    )

    parser.add_argument(
        "--filter",
        help="Filter model names by pattern when using --list-models (e.g., 'gemma', 'gemini')",
    )

    parser.add_argument(
        "--conversation-id",
        "--convo-id",
        dest="conversation_id",
        help="Conversation ID to maintain thread context",
    )

    parser.add_argument(
        "--image",
        "-i",
        action="append",
        dest="image_paths",
        help="Image path (can be provided multiple times)",
    )

    parser.add_argument(
        "--file",
        "-f",
        action="append",
        dest="file_paths",
        help="Document path (can be provided multiple times)",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "ndjson"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential text wrappers; useful for scripting",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        help="Nucleus sampling parameter",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        dest="top_k",
        help="Top-k sampling parameter",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        dest="max_output_tokens",
        help="Maximum output tokens",
    )

    parser.add_argument(
        "--response-schema",
        help="Path to JSON schema file. Response is validated before printing.",
    )

    parser.add_argument(
        "--batch",
        help="Path to NDJSON input file for batch execution",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Concurrent workers for --batch (default: 4)",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable local response caching for non-stream, non-conversation requests",
    )

    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=3600,
        help="Cache TTL in seconds (default: 3600)",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Include latency and token usage metadata",
    )

    args = parser.parse_args()

    if args.stream and args.format != "text":
        parser.error("--stream is only supported with --format text")
    if args.stream and args.batch:
        parser.error("--stream cannot be used with --batch")
    if args.batch and args.conversation_id:
        parser.error("--conversation-id cannot be used with --batch (set per line instead)")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.cache_ttl < 0:
        parser.error("--cache-ttl must be >= 0")

    response_schema: Optional[dict] = None
    if args.response_schema:
        loaded_schema = load_json_file(args.response_schema, "response schema")
        if not isinstance(loaded_schema, dict):
            parser.error("--response-schema must contain a JSON object")
        response_schema = loaded_schema

    base_generation_config = build_generation_config(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
        response_schema=response_schema,
    )

    if args.list_models:
        try:
            models = list_available_models(
                api_key=args.api_key,
                vertexai=args.vertexai,
                project=args.project,
                location=args.location,
                filter_pattern=args.filter,
            )
            print_models(models, args)
        except Exception as exc:
            print(f"Error listing models: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.batch:
        try:
            requests = load_batch_requests(args.batch)
            batch_results = run_batch(
                requests=requests,
                args=args,
                response_schema=response_schema,
                base_generation_config=base_generation_config,
            )

            if args.format == "json":
                print(json_dumps(batch_results, pretty=not args.quiet))
            elif args.format == "ndjson":
                for item in batch_results:
                    print(json_dumps(item, pretty=False))
            else:
                for index, item in enumerate(batch_results, start=1):
                    if args.quiet:
                        if item.get("ok"):
                            print(item.get("response", ""))
                        else:
                            print(item.get("error", "unknown error"), file=sys.stderr)
                        continue

                    status = "ok" if item.get("ok") else "error"
                    print(f"\n[{index}] line={item.get('line')} status={status}")
                    if item.get("id") is not None:
                        print(f"id: {item['id']}")
                    if item.get("ok"):
                        print(item.get("response", ""))
                        if args.metrics and item.get("metrics"):
                            print_metrics_text(item["metrics"], quiet=False)
                    else:
                        print(f"error: {item.get('error')}")

            if any(not item.get("ok") for item in batch_results):
                sys.exit(1)
            return
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if not args.prompt:
        parser.error("prompt is required unless using --list-models or --batch")
    if not args.model:
        parser.error("--model is required or set GEMINI_MODEL environment variable")

    try:
        if args.stream and not args.quiet:
            print("Response (streaming):\n")

        stream_handler = None
        if args.stream:
            stream_handler = lambda text: print(text, end="", flush=True)

        result = execute_request(
            prompt=args.prompt,
            model=args.model,
            api_key=args.api_key,
            stream=args.stream,
            vertexai=args.vertexai,
            project=args.project,
            location=args.location,
            image_paths=args.image_paths,
            file_paths=args.file_paths,
            conversation_id=args.conversation_id,
            generation_config=base_generation_config,
            response_schema=response_schema,
            response_schema_path=args.response_schema,
            cache_enabled=args.cache,
            cache_ttl=args.cache_ttl,
            stream_handler=stream_handler,
        )

        if args.stream:
            print()

        if args.format == "json":
            record = single_output_record(result, include_metrics=args.metrics)
            print(json_dumps(record, pretty=not args.quiet))
            return
        if args.format == "ndjson":
            record = single_output_record(result, include_metrics=args.metrics)
            print(json_dumps(record, pretty=False))
            return

        if not args.stream:
            if args.quiet:
                print(result["response_text"])
            else:
                print("\nResponse:\n")
                print(result["response_text"])

        if args.metrics:
            print_metrics_text(result["metrics"], quiet=args.quiet)

        if result.get("conversation_id") and not args.quiet:
            print(f"\n[Conversation ID: {result['conversation_id']}]")
            print(f"To continue this conversation, use: --conversation-id {result['conversation_id']}")

    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
