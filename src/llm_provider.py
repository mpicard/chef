import asyncio
import re


class LLMProvider:
    """Shared interface. Subclass to plug in a different backend"""

    def complete(self, system: str, user: str, role: str = "defaalt") -> str:
        raise NotImplementedError

    async def acomplete(self, system: str, user: str, role: str = "defaalt") -> str:
        # Wrap sync call in a thread; works for any SDK
        return await asyncio.to_thread(self.complete, system, user, role)


def _extract_json_text(text: str) -> str:
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


class OllamaProvider(LLMProvider):
    # TODO
    pass


# class AnthropicProvider(LLMProvider):
#     def __init__(self, model: str = "claude-haiku-4-5-20251001"):
#         if not os.environ.get("ANTHROPIC_API_KEY"):
#             raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")
#         self._client = anthropic.Anthropic()
#         self._model = model

#     def complete(self, system: str, user: str, role: str = "default") -> str:
#         # Nudge the model toward strict JSON for roles that are parsed downstream.
#         structured_roles = {
#             "planner",
#             "replanner",
#             "summarizer",
#             "aggregator",
#             "critic",
#         }
#         sys_prompt = system
#         if role in structured_roles:
#             sys_prompt = (
#                 system + "\n\nRespond with ONLY a single JSON value. No prose, "
#                 "no markdown fences, no commentary."
#             )
#         response = self._client.messages.create(
#             model=self._model,
#             max_tokens=1024,
#             system=sys_prompt,
#             messages=[{"role": "user", "content": user}],
#         )
#         raw = "".join(b.text for b in response.content if b.type == "text")
#         return _extract_json_text(raw) if role in structured_roles else raw
