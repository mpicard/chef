from llm_provider import LLMProvider, safe_json_loads
from tools_registry import TOOLS
from typed_tool import TypedTool

DEFAULT_SYSTEM_PROMPT = """You are a coding agent. Reply with ONLY one JSON object per turn:

- {"tool": "read_file", "args": {"path": "..."}}
- {"tool": "write_file", "args": {"path": "...", "content": "..."}}
- {"done": true, "summary": "..."}
"""


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

    def run(self, goal: str, max_steps: int = 20) -> str:
        messages: list[dict] = [{"role": "user", "content": goal}]
        for _ in range(max_steps):
            raw = self._provider.chat(
                system=self._system_prompt, messages=messages, role="agent"
            )
            messages.append({"role": "assistant", "content": raw})
            try:
                action = safe_json_loads(raw)
            except ValueError as e:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid JSON: {e}. Reply with one JSON object.",
                    }
                )
                continue
            if action.get("done"):
                return action.get("summary") or "done"
            name = action.get("tool")
            args = action.get("args") or {}
            if name not in self._tools:
                obs = f"Unknown tool: {name!r}"
            else:
                try:
                    obs = str(self._tools[name].run(args))
                except Exception as e:  # noqa: BLE001
                    obs = f"Tool error: {e}"
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
        return "Stopped: max_steps reached"
