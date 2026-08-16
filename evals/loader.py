import os
from pathlib import Path

CASES_DIR = Path(__file__).parent.resolve() / "cases"


def resolve_case_id(case_id: str) -> str:
    """Resolve a short case ID (e.g. '04') to the full directory name."""
    matching = sorted(
        d for d in os.listdir(CASES_DIR)
        if (CASES_DIR / d).is_dir() and d.startswith(case_id)
    )
    if not matching:
        raise FileNotFoundError(f"No case found for ID '{case_id}'")
    if len(matching) > 1:
        raise FileNotFoundError(
            f"Ambiguous case ID '{case_id}' — matches: {matching}"
        )
    return matching[0]


def list_case_ids():
    return sorted(d for d in os.listdir(CASES_DIR) if (CASES_DIR / d).is_dir())


def load_goal(case_id: str) -> str:
    case_id = resolve_case_id(case_id)
    goal_path = CASES_DIR / case_id / "goal.md"
    return goal_path.read_text(encoding="utf-8")
