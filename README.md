### Overview
This repo wraps the Google Generative AI SDK (Gemini/Gemma) behind `call_genai.py`. The CLI is ready after creating a Python virtual environment, installing dependencies, and providing your API key or Google Cloud credentials.

### Project structure
- `call_genai.py`: thin compatibility entrypoint that launches the CLI.
- `genai_cli/cli.py`: argparse wiring and top-level command flow.
- `genai_cli/core.py`: API client, content prep, caching, conversations, request execution.
- `genai_cli/batch.py`: NDJSON batch loading and parallel execution.
- `genai_cli/schema.py`: JSON schema load/validation helpers.
- `genai_cli/output.py`: `text/json/ndjson` output formatting helpers.
- `tests/unit/`: fast unit tests for parsing, schema, batch, core, and CLI behavior.
- `tests/integration/`: live API integration tests (hardcoded to `gemma-3-1b-it`).

### Setup commands
Run the following commands from the repo root. On macOS/Linux, use the shell form shown; Windows command prompt users should adapt the `source` step to `.\Scripts\activate` and `export`/`unset` steps to `set`.

```bash
source bin/activate                              # activate the repo's virtual environment
python -m pip install --upgrade pip              # optional but recommended
pip install -r requirements.txt                  # install project + test dependencies
```

### Running the CLI
1. Provide credentials:
   ```bash
   export GEMINI_API_KEY=your_key_here            # macOS/Linux
   set GEMINI_API_KEY=your_key_here               # Windows cmd
   ```
   You can also pass `--api-key` to `call_genai.py` or set `GOOGLE_API_KEY`.

2. Confirm the environment is functional:
   ```bash
   python call_genai.py --list-models
   ```

3. Invoke the script with a prompt:
   ```bash
   python call_genai.py "Summarize the README" --model gemini-2.5-flash
   python call_genai.py "Compare these images" --image example.jpg --model gemma-3-14b
   printf "Extract one action item" | python call_genai.py --stdin --model gemma-3-1b-it --profile extract --quiet
   python call_genai.py --prompt-file prompt.txt --instruction-file system.txt --model gemma-3-1b-it --retries 2 --timeout 20
   python call_genai.py "Return JSON" --model gemma-3-1b-it --response-schema schema.json --json-path '$.result.value' --quiet
   ```

The script supports streaming (`--stream`), document input (`--file`), conversations (`--conversation-id`), prompt/system input from files or STDIN, retry/timeout controls (`--retries`, `--retry-backoff`, `--timeout`), tuned profiles (`--profile`), and Vertex AI mode (`--vertexai --project ... --location ...`) as documented in `call_genai.py --help`.

When using schema mode, `--json-path` can be used in single or batch runs to select one structured value for downstream scripting.

Distinct non-zero exit codes are emitted for orchestration:
- `10`: authentication failures
- `11`: rate-limit failures
- `12`: timeout failures
- `13`: schema validation / JSON-path failures
- `14`: other API failures

### Running tests
Install dev dependencies (including `pytest`) and run:

```bash
pip install -r requirements.txt
pytest
```

The default test run executes unit tests only. Live API integration tests are available and use the hardcoded low-cost model `gemma-3-1b-it`; enable them explicitly:

```bash
RUN_LIVE_INTEGRATION=1 pytest -m integration
```

## Check utilization
https://aistudio.google.com/usage
