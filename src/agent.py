from collections.abc import Callable
from typing import Any

from llm_provider import LLMProvider, safe_json_loads
from tools_registry import TOOLS
from typed_tool import TypedTool

DEFAULT_SYSTEM_PROMPT = """You are a coding agent. Reply with ONLY one JSON object per turn:

- {"tool": "read_file", "args": {"path": "..."}}
- {"tool": "list_dir", "args": {"path": "."}}
- {"tool": "glob", "args": {"pattern": "**/*.py"}}
- {"tool": "write_file", "args": {"path": "...", "content": "..."}}
- {"tool": "edit_file", "args": {"path": "...", "old_string": "...", "new_string": "..."}}
- {"done": true, "summary": "..."}
"""

OnStep = Callable[[dict[str, Any]], None]

_OBS_LIMIT = 500
_CONTENT_PREVIEW = 80


def _truncate_args(args: dict) -> dict:
    out = dict(args)
    for key in ("content", "old_string", "new_string"):
        val = out.get(key)
        if isinstance(val, str) and len(val) > _CONTENT_PREVIEW:
            out[key] = val[:_CONTENT_PREVIEW] + "…"
    return out


class Agent:
    """ReAct-style coding agent: chat → JSON action → tool → observe."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: dict[str, TypedTool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self._provider = provider
        self._tools = tools if tools is not None else TOOLS
        self._system_prompt = system_prompt

    def run(
        self,
        goal: str,
        max_steps: int = 20,
        on_step: OnStep | None = None,
    ) -> str:
        messages: list[dict] = [{"role": "user", "content": goal}]
        for step in range(max_steps):
            raw = self._provider.chat(
                system=self._system_prompt, messages=messages, role="agent"
            )
            if on_step:
                on_step({"kind": "llm", "raw": raw[:_OBS_LIMIT], "step": step})
            messages.append({"role": "assistant", "content": raw})
            try:
                action = safe_json_loads(raw)
            except ValueError as e:
                if on_step:
                    on_step({"kind": "invalid_json", "error": str(e), "step": step})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid JSON: {e}. Reply with one JSON object.",
                    }
                )
                continue
            if action.get("done"):
                summary = action.get("summary") or "done"
                if on_step:
                    on_step({"kind": "done", "summary": summary, "step": step})
                return summary
            name = action.get("tool")
            args = action.get("args") or {}
            if name not in self._tools:
                obs = f"Unknown tool: {name!r}"
            else:
                try:
                    obs = str(self._tools[name].run(args))
                except Exception as e:  # noqa: BLE001
                    obs = f"Tool error: {e}"
            if on_step:
                on_step(
                    {
                        "kind": "tool",
                        "tool": name,
                        "args": _truncate_args(args),
                        "observation": obs[:_OBS_LIMIT],
                        "step": step,
                    }
                )
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
        return "Stopped: max_steps reached"
