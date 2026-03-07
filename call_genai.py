#!/usr/bin/env python3
"""CLI entrypoint for calling Google Generative AI models."""

import sys

from genai_cli.cli import main


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
