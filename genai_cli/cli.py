import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .batch import load_batch_requests, run_batch
from .common import json_dumps
from .core import build_generation_config, execute_request, list_available_models
from .errors import ERROR_TYPE_API, exit_code_for_error_type, exit_code_for_exception
from .output import print_metrics_text, print_models, single_output_record
from .schema import load_json_file

PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "extract": {"temperature": 0.0, "top_p": 0.1, "max_output_tokens": 512},
    "classify": {"temperature": 0.0, "top_p": 0.1, "max_output_tokens": 128},
    "summarize": {"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 512},
    "codefix": {"temperature": 0.1, "top_p": 0.9, "top_k": 40, "max_output_tokens": 1024},
}


def build_parser() -> argparse.ArgumentParser:
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
  printf "extract invoice fields" | python call_genai.py --stdin --model gemma-3-1b-it --profile extract --quiet
  python call_genai.py --prompt-file prompt.txt --instruction-file system.txt --model gemma-3-1b-it --retries 2 --timeout 20
  python call_genai.py "Extract fields" --response-schema schema.json --format json --quiet
  python call_genai.py "Extract fields" --response-schema schema.json --json-path '$.customer.id' --quiet
  python call_genai.py --batch requests.ndjson --model gemma-3-1b-it --jobs 8 --format ndjson
  python call_genai.py "Hi" --cache --cache-ttl 3600 --metrics --quiet
  python call_genai.py --list-models --filter gemma
        """,
    )

    parser.add_argument("prompt", nargs="?", help="The text prompt to send to the model")
    parser.add_argument("--stdin", action="store_true", help="Read prompt text from STDIN")
    parser.add_argument("--prompt-file", help="Read prompt text from file")
    parser.add_argument("--system", help="System instruction text to steer model behavior")
    parser.add_argument("--instruction-file", help="Read system instruction text from file")

    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model to use (e.g., 'gemini-2.5-flash', 'gemma-3-27b-it'). Defaults to GEMINI_MODEL if set.",
    )

    parser.add_argument(
        "--api-key",
        "-k",
        help="API key for Gemini Developer API (or set GEMINI_API_KEY / GOOGLE_API_KEY)",
    )

    parser.add_argument("--stream", "-s", action="store_true", help="Stream the response as it's generated")

    parser.add_argument("--vertexai", action="store_true", help="Use Vertex AI API instead of Gemini Developer API")
    parser.add_argument("--project", help="Google Cloud project ID (required for Vertex AI)")
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Google Cloud location (for Vertex AI, default: us-central1)",
    )

    parser.add_argument("--list-models", action="store_true", help="List available models from the API and exit")
    parser.add_argument(
        "--filter",
        help="Filter model names by pattern when using --list-models (e.g., 'gemma', 'gemini')",
    )

    parser.add_argument("--conversation-id", "--convo-id", dest="conversation_id", help="Conversation ID to maintain thread context")

    parser.add_argument("--image", "-i", action="append", dest="image_paths", help="Image path (can be provided multiple times)")
    parser.add_argument("--file", "-f", action="append", dest="file_paths", help="Document path (can be provided multiple times)")

    parser.add_argument("--format", choices=["text", "json", "ndjson"], default="text", help="Output format (default: text)")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-essential text wrappers; useful for scripting")

    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, dest="top_p", help="Nucleus sampling parameter")
    parser.add_argument("--top-k", type=int, dest="top_k", help="Top-k sampling parameter")
    parser.add_argument("--max-output-tokens", type=int, dest="max_output_tokens", help="Maximum output tokens")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS.keys()), help="Apply tuned generation defaults")

    parser.add_argument("--response-schema", help="Path to JSON schema file. Response is validated before printing.")
    parser.add_argument("--json-path", help="Select one value from validated JSON output (e.g., $.result.value)")

    parser.add_argument("--batch", help="Path to NDJSON input file for batch execution")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent workers for --batch (default: 4)")

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable local response caching for non-stream, non-conversation requests",
    )
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds (default: 3600)")
    parser.add_argument("--retries", type=int, default=0, help="Retry count for transient request failures")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Base backoff in seconds between retries (default: 1.0, exponential)",
    )
    parser.add_argument("--timeout", type=float, help="Per-request timeout in seconds")

    parser.add_argument("--metrics", action="store_true", help="Include latency and token usage metadata")

    return parser


def _resolve_model_arg(args: argparse.Namespace) -> Optional[str]:
    if args.model:
        return args.model

    # Defer env lookup here so parser defaults stay clear for testing/import use.
    import os

    return os.getenv("GEMINI_MODEL")


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    if not args.profile:
        return

    defaults = PROFILE_DEFAULTS[args.profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def _load_text_file(path: str, description: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} file not found: {path}")
    return file_path.read_text()


def _resolve_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Optional[str]:
    sources = 0
    prompt_value: Optional[str] = None

    if args.prompt is not None:
        sources += 1
        prompt_value = args.prompt
    if args.stdin:
        sources += 1
    if args.prompt_file:
        sources += 1

    if sources > 1:
        parser.error("Use only one prompt source: positional prompt, --stdin, or --prompt-file")

    if args.stdin:
        prompt_value = sys.stdin.read()
    elif args.prompt_file:
        prompt_value = _load_text_file(args.prompt_file, "prompt")

    return prompt_value


def _resolve_system_instruction(args: argparse.Namespace) -> Optional[str]:
    instruction_parts = []
    if args.instruction_file:
        instruction_parts.append(_load_text_file(args.instruction_file, "instruction"))
    if args.system:
        instruction_parts.append(args.system)
    if not instruction_parts:
        return None
    return "\n".join(part for part in instruction_parts if part)


def _format_selected_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json_dumps(value, pretty=False)


def _batch_exit_code(batch_results: Any) -> int:
    for item in batch_results:
        if not item.get("ok"):
            return exit_code_for_error_type(item.get("error_type", ERROR_TYPE_API))
    return exit_code_for_error_type(ERROR_TYPE_API)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.model = _resolve_model_arg(args)

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
    if args.retries < 0:
        parser.error("--retries must be >= 0")
    if args.retry_backoff < 0:
        parser.error("--retry-backoff must be >= 0")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.json_path and not args.response_schema:
        parser.error("--json-path requires --response-schema")

    _apply_profile_defaults(args)

    response_schema = None
    if args.response_schema:
        loaded_schema = load_json_file(args.response_schema, "response schema")
        if not isinstance(loaded_schema, dict):
            parser.error("--response-schema must contain a JSON object")
        response_schema = loaded_schema

    try:
        system_instruction = _resolve_system_instruction(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for_exception(exc))

    base_generation_config = build_generation_config(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
        response_schema=response_schema,
    )
    if system_instruction:
        base_generation_config = dict(base_generation_config or {})
        base_generation_config["system_instruction"] = system_instruction

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
            sys.exit(exit_code_for_exception(exc))
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
                        if item.get("error_type"):
                            print(f"type: {item.get('error_type')}")

            if any(not item.get("ok") for item in batch_results):
                sys.exit(_batch_exit_code(batch_results))
            return
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(exit_code_for_exception(exc))

    try:
        prompt = _resolve_prompt(args, parser)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for_exception(exc))

    if not prompt:
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
            prompt=prompt,
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
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            timeout_seconds=args.timeout,
            json_path=args.json_path,
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
                if args.json_path:
                    print(_format_selected_value(result.get("selected_json")))
                else:
                    print(result["response_text"])
            else:
                print("\nResponse:\n")
                print(result["response_text"])
                if args.json_path:
                    print(f"\nSelected (--json-path {args.json_path}):")
                    print(_format_selected_value(result.get("selected_json")))

        if args.metrics:
            print_metrics_text(result["metrics"], quiet=args.quiet)

        if result.get("conversation_id") and not args.quiet:
            print(f"\n[Conversation ID: {result['conversation_id']}]")
            print(f"To continue this conversation, use: --conversation-id {result['conversation_id']}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for_exception(exc))
