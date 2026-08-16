import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator import RunResult


def persist_run(
    result: RunResult,
    runs_dir: Path,
    *,
    stamp: str | None = None,
) -> Path:
    """Write run meta + events as JSONL under ``runs_dir``. Returns the file path."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if stamp is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = runs_dir / f"{stamp}.jsonl"

    meta: dict[str, Any] = {
        "type": "meta",
        "goal": result.goal,
        "plan": result.plan,
        "report": result.report,
        "status": result.status,
        "n_events": len(result.events),
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for ev in result.events:
            f.write(json.dumps({"type": "event", **ev}, ensure_ascii=False) + "\n")
    return path
