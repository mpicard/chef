import json

from agent import DEFAULT_SYSTEM_PROMPT, Agent
from tools_registry import make_tools


class ScriptedProvider:
    """Returns canned chat() responses in order."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, system: str, messages: list[dict], role: str = "default") -> str:
        self.calls.append(
            {
                "system": system,
                "messages": [dict(m) for m in messages],
                "role": role,
            }
        )
        if not self.replies:
            raise AssertionError("ScriptedProvider: no replies left")
        return self.replies.pop(0)

    def complete(self, system: str, user: str, role: str = "default") -> str:
        return self.chat(system, [{"role": "user", "content": user}], role)


def test_agent_done_immediately():
    provider = ScriptedProvider([json.dumps({"done": True, "summary": "all good"})])
    agent = Agent(provider, tools={})
    assert agent.run("do nothing") == "all good"
    assert provider.calls[0]["role"] == "agent"
    assert provider.calls[0]["system"] == DEFAULT_SYSTEM_PROMPT


def test_agent_tool_then_done(tmp_path):
    tools = make_tools(root=tmp_path)
    provider = ScriptedProvider(
        [
            json.dumps(
                {"tool": "write_file", "args": {"path": "x.txt", "content": "hi"}}
            ),
            json.dumps({"done": True, "summary": "wrote x.txt"}),
        ]
    )
    agent = Agent(provider, tools=tools)
    assert agent.run("write x") == "wrote x.txt"
    assert (tmp_path / "x.txt").read_text() == "hi"
    # second user message should be the observation
    assert "Observation:" in provider.calls[1]["messages"][-1]["content"]


def test_agent_unknown_tool_is_observation_then_done():
    provider = ScriptedProvider(
        [
            json.dumps({"tool": "bash", "args": {"command": "ls"}}),
            json.dumps({"done": True, "summary": "gave up"}),
        ]
    )
    agent = Agent(provider, tools={})
    assert agent.run("ls") == "gave up"
    obs = provider.calls[1]["messages"][-1]["content"]
    assert "Unknown tool" in obs


def test_agent_invalid_json_retries():
    provider = ScriptedProvider(
        [
            "not json at all {{{",
            json.dumps({"done": True, "summary": "recovered"}),
        ]
    )
    agent = Agent(provider, tools={})
    assert agent.run("goal") == "recovered"
    assert "Invalid JSON" in provider.calls[1]["messages"][-1]["content"]


def test_agent_respects_tool_allowlist(tmp_path):
    """Plan-style: only read_file is available."""
    tools = make_tools(root=tmp_path)
    (tmp_path / "a.txt").write_text("secret", encoding="utf-8")
    read_only = {"read_file": tools["read_file"]}
    provider = ScriptedProvider(
        [
            json.dumps(
                {"tool": "write_file", "args": {"path": "a.txt", "content": "hacked"}}
            ),
            json.dumps({"tool": "read_file", "args": {"path": "a.txt"}}),
            json.dumps({"done": True, "summary": "read only"}),
        ]
    )
    agent = Agent(provider, tools=read_only)
    assert agent.run("try write") == "read only"
    assert (tmp_path / "a.txt").read_text() == "secret"
    assert "Unknown tool" in provider.calls[1]["messages"][-1]["content"]
    assert "secret" in provider.calls[2]["messages"][-1]["content"]


def test_agent_custom_system_prompt():
    provider = ScriptedProvider([json.dumps({"done": True, "summary": "ok"})])
    agent = Agent(provider, tools={}, system_prompt="PLAN MODE ONLY")
    agent.run("g")
    assert provider.calls[0]["system"] == "PLAN MODE ONLY"


def test_agent_max_steps():
    forever = json.dumps({"tool": "read_file", "args": {"path": "nope.txt"}})
    provider = ScriptedProvider([forever, forever, forever])
    agent = Agent(provider, tools={})
    assert agent.run("loop", max_steps=2) == "Stopped: max_steps reached"
    assert len(provider.calls) == 2


def test_agent_on_step_tool_then_done(tmp_path):
    tools = make_tools(root=tmp_path)
    events: list[dict] = []
    provider = ScriptedProvider(
        [
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "x.txt", "content": "hi" + ("!" * 100)},
                }
            ),
            json.dumps({"done": True, "summary": "wrote x.txt"}),
        ]
    )
    agent = Agent(provider, tools=tools)
    assert agent.run("write x", on_step=events.append) == "wrote x.txt"

    kinds = [e["kind"] for e in events]
    assert kinds == ["llm", "tool", "llm", "done"]

    tool_ev = events[1]
    assert tool_ev["tool"] == "write_file"
    assert tool_ev["step"] == 0
    assert tool_ev["args"]["path"] == "x.txt"
    assert tool_ev["args"]["content"].endswith("…")
    assert len(tool_ev["args"]["content"]) <= 81
    assert "Wrote" in tool_ev["observation"]

    assert events[3]["summary"] == "wrote x.txt"
    assert events[3]["step"] == 1


def test_agent_on_step_invalid_json():
    events: list[dict] = []
    provider = ScriptedProvider(
        [
            "not json {{{",
            json.dumps({"done": True, "summary": "ok"}),
        ]
    )
    agent = Agent(provider, tools={})
    assert agent.run("g", on_step=events.append) == "ok"
    kinds = [e["kind"] for e in events]
    assert kinds == ["llm", "invalid_json", "llm", "done"]
    assert "error" in events[1]
