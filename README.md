# chef - Agentic Coder

Educational project for learning how agentic coding works.

## Architecture

Two-node hybrid:

1. **Plan** — ReAct loop with `read_file` / `list_dir` / `glob`; produces a plan
2. **Build** — ReAct loop with those plus `write_file` / `edit_file`; implements the plan

Entry: `cli` / `main.py` → `Orchestrator` → `Agent`.

## Run

```bash
uv sync --group dev
uv run chef "Create sandbox/hello.py that prints Hello, chef"
# or
uv run python main.py "your goal" --plan-steps 8 --quiet
```

Flags: `--plan-steps N`, `--build-steps N`, `-q` / `--quiet`, `--no-persist`, `--runs-dir PATH`.

Traces are written by default to `runs/<timestamp>.jsonl` (gitignored).

## Test

```bash
python -m pytest test/ -v
```

## Eval

Run all cases:

```bash
python evals/runner.py
```

Run specific cases:

```bash
python evals/runner.py --case 01 02
```

Flags: `--case IDs`, `-q` / `--quiet`, `--verbose`, `--model NAME`, `--base-url URL`.
