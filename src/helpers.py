import json
import re


def extract_json_text(text: str) -> str:
    """Best-effort extraction of a JSON payload from an LLM response.

    Handles: raw JSON, ```json ... ``` / ``` ... ``` fences, and prose
    surrounding a JSON object/array. Returns '' if nothing JSON-like is found.
    """
    if not text:
        return ""
    s = text.strip()
    # Strip markdown code fences if present
    fence = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE
    )
    if fence:
        s = fence.group(1).strip()
    # If it already looks like JSON, return as-is
    if s.startswith(("{", "[")):
        return s
    # Otherwise pull out the first {...} or [...] block
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s


def safe_json_loads(raw: str) -> dict:
    """Tolerant JSON parse: strips code fences / prose, raises ValueError on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        cleaned = extract_json_text(raw or "")
        if not cleaned:
            raise ValueError(f"LLM returned no JSON payload (got {raw!r})")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned malformed JSON: {e}. Raw={raw!r}") from e
