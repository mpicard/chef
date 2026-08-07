import asyncio
import json
import re

from typing import Any

from openai import OpenAI

STRUCTURED_ROLES = frozenset({"agent"})


class LLMProvider:
    """Shared interface. Subclass to plug in a different backend."""

    def chat(self, system: str, messages: list[dict], role: str = "default") -> str:
        raise NotImplementedError

    def complete(self, system: str, user: str, role: str = "default") -> str:
        return self.chat(
            system, [{"role": "user", "content": user}], role=role
        )

    async def acomplete(self, system: str, user: str, role: str = "default") -> str:
        return await asyncio.to_thread(self.complete, system, user, role)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str = "",
        client: Any | None = None,
    ):
        self._client = client or OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(self, system: str, messages: list[dict], role: str = "default") -> str:
        system_prompt = system
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        if role in STRUCTURED_ROLES:
            system_prompt = (
                system + "\n\nRespond with ONLY a single JSON value. "
                "No prose, no markdown fences, no commentary."
            )
            kwargs["messages"][0]["content"] = system_prompt
            kwargs["response_format"] = {"type": "json_object"}
        res = self._client.chat.completions.create(**kwargs)
        raw = res.choices[0].message.content or ""
        return _extract_json_text(raw) if role in STRUCTURED_ROLES else raw


def safe_json_loads(raw: str) -> dict:
    """Tolerant JSON parse: strips code fences / prose, raises ValueError on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        cleaned = _extract_json_text(raw or "")
        if not cleaned:
            raise ValueError(f"LLM returned no JSON payload (got {raw!r})")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned malformed JSON: {e}. Raw={raw!r}") from e


def _extract_json_text(text: str) -> str:
    """Best-effort extraction of a JSON payload from an LLM response.

    Handles: raw JSON, ```json ... ``` / ``` ... ``` fences, and prose
    surrounding a JSON object/array. Returns '' if nothing JSON-like is found.
    """
    if not text:
        return ""
    s = text.strip()
    fence = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE
    )
    if fence:
        s = fence.group(1).strip()
    if s.startswith(("{", "[")):
        return s
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s
