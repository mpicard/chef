from types import SimpleNamespace

import pytest

from llm_provider import (
    OpenAICompatibleProvider,
    _extract_json_text,
    safe_json_loads,
)


class FakeCompletions:
    def __init__(self):
        self.kwargs: dict | None = None
        self.response_text = '{"ok": true}'

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_complete_wraps_user_as_message():
    client = FakeClient()
    provider = OpenAICompatibleProvider(model="test", client=client)
    out = provider.complete("sys", "hello", role="default")
    assert out == '{"ok": true}'
    msgs = client.chat.completions.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "hello"}
    assert "response_format" not in client.chat.completions.kwargs


def test_chat_passes_full_history():
    client = FakeClient()
    provider = OpenAICompatibleProvider(model="test", client=client)
    provider.chat(
        "sys",
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
        role="default",
    )
    msgs = client.chat.completions.kwargs["messages"]
    assert len(msgs) == 4
    assert msgs[1]["content"] == "u1"
    assert msgs[3]["content"] == "u2"


def test_agent_role_forces_json_and_extracts():
    client = FakeClient()
    client.chat.completions.response_text = 'Here you go:\n```json\n{"done": true}\n```'
    provider = OpenAICompatibleProvider(model="test", client=client)
    out = provider.chat("sys", [{"role": "user", "content": "go"}], role="agent")
    assert out == '{"done": true}'
    kwargs = client.chat.completions.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "ONLY a single JSON value" in kwargs["messages"][0]["content"]


def test_safe_json_loads_raw_object():
    assert safe_json_loads('{"a": 1}') == {"a": 1}


def test_safe_json_loads_fenced():
    assert safe_json_loads('```json\n{"a": 2}\n```') == {"a": 2}


def test_safe_json_loads_prose():
    assert safe_json_loads('Sure: {"a": 3} thanks') == {"a": 3}


def test_safe_json_loads_raises_on_garbage():
    with pytest.raises(ValueError):
        safe_json_loads("no json here")


def test_extract_json_text_empty():
    assert _extract_json_text("") == ""
