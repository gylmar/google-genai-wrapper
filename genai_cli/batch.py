import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import execute_request
from .output import single_output_record


def normalize_batch_item(item: Any, line_number: int) -> Dict[str, Any]:
    """Normalize one NDJSON line into a request dict."""
    if isinstance(item, str):
        return {"prompt": item}
    if isinstance(item, dict):
        return item
    raise ValueError(
        f"Batch line {line_number}: expected JSON object or string, got {type(item).__name__}"
    )


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
        futures = {executor.submit(_run_item, idx, request): idx for idx, request in enumerate(requests)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return results
