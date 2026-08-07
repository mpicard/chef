import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_VENV_PY = _ROOT / ".venv" / "bin" / "python"

# Prefer the project venv: a bare `python3` on PATH is often Homebrew and
# won't see the editable install that maps `src/` onto imports.
if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    os.execv(str(_VENV_PY), [str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import llm
from memory_store import MemoryStore
from orchestrator import Orchestrator
from tools_registry import TOOLS


def main():
    store = MemoryStore()
    store.add(
        "Previously built a 2-city comparison; forgot to include timezone. "
        "Reminder: always verify every requested attribute is present.",
        kind="episodic",
    )
    store.add("The Eiffel Tower is in Paris and is 330 meters tall.", kind="semantic")
    store.add(
        "Tokyo's Shibuya crossing is one of the busiest intersections in the world.",
        kind="semantic",
    )

    orch = Orchestrator(provider=llm, tools=TOOLS, memory=store)

    GOAL = (
        "Build a comparison report of Paris, Tokyo, and New York covering "
        "population, timezone, and a narrative summary for each."
    )
    REQUIRED = ["paris", "tokyo", "new york"]

    result = asyncio.run(orch.run(GOAL, REQUIRED))

    print(f"Status:        {result.status}")
    print(
        f"Verdict:       passed={result.verdict.passed if result.verdict else 'n/a'}  "
        f"tier={result.verdict.tier if result.verdict else 'n/a'}"
    )
    print(f"Tokens used:   {result.budget.tokens_used}")
    print(f"Tool calls:    {result.budget.tool_calls_used}")
    print(f"Cost (est):    ${result.budget.cost_eur:.4f}")
    print(f"Budget pressure final: {result.budget.pressure():.2f}")
    print()
    print("--- REPORT ---")
    print(result.report)


if __name__ == "__main__":
    main()
