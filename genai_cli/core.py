import hashlib
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .common import json_dumps, to_jsonable
from .schema import parse_and_validate_schema_response

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:
    raise RuntimeError(
        "google-genai package not found. Please install it with: pip install google-genai"
    ) from exc

try:
    import PIL.Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONVERSATIONS_DIR = PROJECT_ROOT / ".genai_conversations"
CACHE_DIR = PROJECT_ROOT / ".genai_cache"

CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_client(
    api_key: Optional[str] = None,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> genai.Client:
    """Initialize and return a Google Generative AI client."""
    if vertexai:
        return genai.Client(
            vertexai=True,
            project=project or os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key required. Provide --api-key or set GEMINI_API_KEY / GOOGLE_API_KEY.")
    return genai.Client(api_key=api_key)


def list_available_models(
    api_key: Optional[str] = None,
    vertexai: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    filter_pattern: Optional[str] = None,
) -> List[str]:
    """List all available models from the API."""
    client = get_client(api_key, vertexai, project, location)

    models: List[str] = []
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


def prepare_contents(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
) -> List[Union[str, genai_types.Part]]:
    """Prepare contents for the API call, including text, images, and files."""
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
    except Exception:
        return None


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
    """Build a stable fingerprint from file path + stat metadata."""
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
    """Hash request inputs into a deterministic cache key."""
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
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


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
    """Call a Google Generative AI model and return normalized response data."""
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
    """Execute one model request with optional cache lookup."""
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
