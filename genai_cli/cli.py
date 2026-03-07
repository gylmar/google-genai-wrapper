import argparse
import sys
from typing import Optional

from .batch import load_batch_requests, run_batch
from .common import json_dumps
from .core import build_generation_config, execute_request, list_available_models
from .output import print_metrics_text, print_models, single_output_record
from .schema import load_json_file


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
  python call_genai.py "Extract fields" --response-schema schema.json --format json --quiet
  python call_genai.py --batch requests.ndjson --model gemma-3-1b-it --jobs 8 --format ndjson
  python call_genai.py "Hi" --cache --cache-ttl 3600 --metrics --quiet
  python call_genai.py --list-models --filter gemma
        """,
    )

    parser.add_argument("prompt", nargs="?", help="The text prompt to send to the model")

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

    parser.add_argument("--response-schema", help="Path to JSON schema file. Response is validated before printing.")

    parser.add_argument("--batch", help="Path to NDJSON input file for batch execution")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent workers for --batch (default: 4)")

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable local response caching for non-stream, non-conversation requests",
    )
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds (default: 3600)")

    parser.add_argument("--metrics", action="store_true", help="Include latency and token usage metadata")

    return parser


def _resolve_model_arg(args: argparse.Namespace) -> Optional[str]:
    if args.model:
        return args.model

    # Defer env lookup here so parser defaults stay clear for testing/import use.
    import os

    return os.getenv("GEMINI_MODEL")


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

    response_schema = None
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
