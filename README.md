# chef - Agentic Coder

Educational project for learning how agentic coding works.

## Architecture

Two-node hybrid:

1. **Plan** — ReAct loop with `read_file` only; produces an implementation plan
2. **Build** — ReAct loop with `read_file` + `write_file`; implements the plan

Entry point: `main.py` → `Orchestrator` → `Agent`.

## Run

```bash
uv sync --group dev
python main.py
```

## Test

```bash
python -m pytest test/ -v
```
