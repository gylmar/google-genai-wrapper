import argparse
import sys
from typing import Any, Dict, List

from .common import json_dumps


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
