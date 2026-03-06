# AGENTS Instructions

## Local environment file
1. This repo now ships with a default `local.env.example` template (see the file header for guidance). Copy it to `local.env` and fill in your API key or model default there if you prefer, but avoid committing actual secrets.
2. If you want to manage the secrets from a secure location instead, replace the template with a symlink:
   ```sh
   rm local.env
   ln -s /path/to/local.env local.env
   ```
   Use `git update-index --skip-worktree local.env` or `.git/info/exclude` to stop git from flagging the replacement, then `source local.env` before running the script.
3. If you still need to create credentials, the minimum required exports are `GEMINI_API_KEY` (plus `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` for Vertex AI). Keep the real file outside git, and point the symlink to it after you create it.
4. Keep any helper files you actually edit locally (for example, `*.env.local`) listed in `.gitignore`, and double-check with `ls -l local.env` that it points where expected.

## Python virtual environment
1. The repo already includes a virtual environment under `./bin`. Activate it before running anything:
   ```sh
   source bin/activate
   ```
2. Once active, install the dependencies if you haven’t already:
   ```sh
   pip install -r requirements.txt
   ```
3. After activation and credential sourcing, you can safely run `python call_genai.py …` (the script will pick up the env vars and the venv’s Python).
4. If you need to recreate the venv later, delete the current `lib/`/`bin/` tree and run:
   ```sh
   python3 -m venv .
   source bin/activate
   pip install -r requirements.txt
   ```
```
