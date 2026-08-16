import json
import time
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

To finish, you MUST use {"done": true, "summary": "..."}.
Never use {"tool": "done", ...} — done is not a tool.
"""

OnStep = Callable[[dict[str, Any]], None]

_LOG_LIMIT = 500
_MSG_OBS_LIMIT = 8000
_CONTENT_PREVIEW = 80
_KEEP_LAST = 12


def _truncate_args(args: dict) -> dict:
    out = dict(args)
    for key in ("content", "old_string", "new_string"):
        val = out.get(key)
        if isinstance(val, str) and len(val) > _CONTENT_PREVIEW:
            out[key] = val[:_CONTENT_PREVIEW] + "…"
    return out


def truncate_for_messages(text: str, limit: int = _MSG_OBS_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def history_assistant_text(raw: str, action: dict[str, Any] | None) -> str:
    """Store a redacted assistant turn in history (full tool args already applied)."""
    if not isinstance(action, dict):
        return truncate_for_messages(raw, _MSG_OBS_LIMIT)
    name = action.get("tool")
    if name in ("write_file", "edit_file"):
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        redacted = {"tool": name, "args": _truncate_args(args)}
        return json.dumps(redacted, ensure_ascii=False)
    return truncate_for_messages(raw, _MSG_OBS_LIMIT)


def compact_messages(
    messages: list[dict],
    *,
    keep_last: int = _KEEP_LAST,
) -> list[dict]:
    """Keep the goal plus the last ``keep_last`` turns; drop the middle."""
    if len(messages) <= 1 + keep_last:
        return list(messages)
    goal = messages[0]
    rest = messages[1:]
    omitted = len(rest) - keep_last
    kept = rest[-keep_last:]
    stub = {
        "role": "user",
        "content": f"[omitted {omitted} earlier messages to save context]",
    }
    return [goal, stub, *kept]


def _ms_since(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def done_summary(action: Any) -> str | None:
    """Return summary if action is a finish signal, else None.

    Accepts both {"done": true, "summary": "..."} and the common misfire
    {"tool": "done", "summary": "..."} / {"tool": "done", "args": {"summary": "..."}}.
    """
    if not isinstance(action, dict):
        return None
    if action.get("done") is True or action.get("done") == "true":
        return action.get("summary") or "done"
    if action.get("tool") == "done":
        args = action.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        return action.get("summary") or args.get("summary") or "done"
    return None


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
        self._prompt_tokens = 0
        self._completion_token = 0

    def run(
        self,
        goal: str,
        max_steps: int = 20,
        on_step: OnStep | None = None,
        *,
        keep_last: int = _KEEP_LAST,
    ) -> str:
        messages: list[dict] = [{"role": "user", "content": goal}]
        for step in range(max_steps):
            t_llm = time.monotonic()
            raw, usage = self._provider.chat(
                system=self._system_prompt,
                messages=compact_messages(messages, keep_last=keep_last),
                role="agent",
            )
            if usage:
                self._prompt_tokens += usage.prompt_tokens or 0
                self._completion_token += usage.completion_tokens or 0
            llm_ms = _ms_since(t_llm)
            if on_step:
                on_step(
                    {
                        "kind": "llm",
                        "raw": raw[:_LOG_LIMIT],
                        "step": step,
                        "latency_ms": llm_ms,
                    }
                )
            try:
                action = safe_json_loads(raw)
            except ValueError as e:
                messages.append({"role": "assistant", "content": raw[:_MSG_OBS_LIMIT]})
                if on_step:
                    on_step(
                        {
                            "kind": "invalid_json",
                            "error": str(e),
                            "step": step,
                            "latency_ms": 0,
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid JSON: {e}. Reply with one JSON object.",
                    }
                )
                continue
            if not isinstance(action, dict):
                err = (
                    f"Expected a JSON object, got {type(action).__name__}: "
                    f"{raw[:200]!r}"
                )
                messages.append({"role": "assistant", "content": raw[:_MSG_OBS_LIMIT]})
                if on_step:
                    on_step(
                        {
                            "kind": "invalid_json",
                            "error": err,
                            "step": step,
                            "latency_ms": 0,
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": f"{err}. Reply with one JSON object.",
                    }
                )
                continue
            if (summary := done_summary(action)) is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": history_assistant_text(raw, action),
                    }
                )
                if on_step:
                    on_step(
                        {
                            "kind": "done",
                            "summary": summary,
                            "step": step,
                            "latency_ms": 0,
                        }
                    )
                return summary
            name = action.get("tool")
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            t_tool = time.monotonic()
            if name not in self._tools:
                obs = f"Unknown tool: {name!r}"
            else:
                try:
                    obs = str(self._tools[name].run(args))
                except Exception as e:  # noqa: BLE001
                    obs = f"Tool error: {e}"
            tool_ms = _ms_since(t_tool)
            if on_step:
                on_step(
                    {
                        "kind": "tool",
                        "tool": name,
                        "args": _truncate_args(args),
                        "observation": obs[:_LOG_LIMIT],
                        "step": step,
                        "latency_ms": tool_ms,
                    }
                )
            messages.append(
                {"role": "assistant", "content": history_assistant_text(raw, action)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation:\n{truncate_for_messages(obs)}",
                }
            )
        return "Stopped: max_steps reached"

    @property
    def token_usage(self) -> tuple[int, int]:
        return self._prompt_tokens, self._completion_token
