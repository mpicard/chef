from pathlib import Path
from typing import Any

from orchestrator import RunResult


def grade(result: RunResult, output_dir: Path) -> tuple[bool, str]:
    """Grade the hello world case.

    Checks that sandbox/hello.py was created with expected content.
    """
    hello_py = output_dir / "sandbox" / "hello.py"
    if not hello_py.exists():
        return False, "sandbox/hello.py was not created"

    content = hello_py.read_text()
    has_hello = "hello" in content.lower()
    has_chef = "chef" in content.lower()
    has_print = "print" in content.lower()

    if not (has_hello and has_chef and has_print):
        missing: list[str] = []
        if not has_hello:
            missing.append("hello")
        if not has_chef:
            missing.append("chef")
        if not has_print:
            missing.append("print")
        return False, f"content missing keywords: {', '.join(missing)}"

    return True, "sandbox/hello.py exists with expected content"
