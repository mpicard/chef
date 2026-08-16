from pathlib import Path
from typing import Any

from orchestrator import RunResult


def grade(result: RunResult, output_dir: Path) -> tuple[bool, str]:
    """Grade the math utils case.

    Checks that math_utils.py was created with all four expected functions.
    """
    math_utils = output_dir / "math_utils.py"
    if not math_utils.exists():
        return False, "math_utils.py was not created"

    content = math_utils.read_text()
    for fn in ("add", "subtract", "multiply", "divide"):
        if f"def {fn}" not in content:
            return False, f"missing function: {fn}"

    return True, "math_utils.py contains all expected functions"
