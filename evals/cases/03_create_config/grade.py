import json
from pathlib import Path
from typing import Any

from orchestrator import RunResult


def grade(result: RunResult, output_dir: Path) -> tuple[bool, str]:
    """Grade the config case.

    Checks that config/settings.json was created with expected structure.
    """
    settings_json = output_dir / "config" / "settings.json"
    if not settings_json.exists():
        return False, "config/settings.json was not created"

    try:
        data = json.loads(settings_json.read_text())
    except json.JSONDecodeError as e:
        return False, f"settings.json is not valid JSON: {e}"

    if "name" not in data:
        return False, "missing 'name' field"
    if "version" not in data:
        return False, "missing 'version' field"

    return True, "settings.json has expected structure"
