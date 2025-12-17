### Overview
This repo wraps the Google Generative AI SDK (Gemini/Gemma) behind `call_genai.py`. The CLI is ready after creating a Python virtual environment, installing dependencies, and providing your API key or Google Cloud credentials.

### Setup commands
Run the following commands from the repo root. On macOS/Linux, use the shell form shown; Windows command prompt users should adapt the `source` step to `.\.venv\Scripts\activate` and `export`/`unset` steps to `set`.

```bash
python3 -m venv .venv                            # create the virtual environment
source .venv/bin/activate                        # macOS/Linux; use .\.venv\Scripts\activate on Windows
python -m pip install --upgrade pip              # optional but recommended to refresh pip
pip install -r requirements.txt                  # install project dependencies
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
   ```

The script supports streaming (`--stream`), document input (`--file`), conversations (`--conversation-id`), and Vertex AI mode (`--vertexai --project ... --location ...`) as documented in `call_genai.py --help`.

## Check utilization
https://aistudio.google.com/usage
