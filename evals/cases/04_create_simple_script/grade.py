from pathlib import Path
from typing import Any

from orchestrator import RunResult


def grade(result: RunResult, output_dir: Path) -> tuple[bool, str]:
    """Grade the simple script case.

    Checks that processor.py was created with expected functionality.
    """
    processor_py = output_dir / "processor.py"
    if not processor_py.exists():
        return False, "processor.py was not created"

    content = processor_py.read_text()
    has_read = "read" in content.lower() or "open" in content.lower()

    if not has_read:
        return False, "script doesn't contain file read operations"

    return True, "processor.py exists with expected functionality"
