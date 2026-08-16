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
    assert isinstance(events[0]["latency_ms"], int)
    assert events[0]["latency_ms"] >= 0
    assert isinstance(events[1]["latency_ms"], int)
    assert events[1]["latency_ms"] >= 0
    assert events[3]["latency_ms"] == 0


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
    assert events[1]["latency_ms"] == 0


def test_agent_accepts_tool_done_misfire():
    events: list[dict] = []
    provider = ScriptedProvider(
        [
            json.dumps(
                {"tool": "done", "summary": "finished via tool done"}
            ),
        ]
    )
    agent = Agent(provider, tools={})
    assert agent.run("g", on_step=events.append) == "finished via tool done"
    assert [e["kind"] for e in events] == ["llm", "done"]


def test_agent_accepts_tool_done_summary_in_args():
    provider = ScriptedProvider(
        [json.dumps({"tool": "done", "args": {"summary": "via args"}})]
    )
    agent = Agent(provider, tools={})
    assert agent.run("g") == "via args"


def test_done_summary_helper():
    from agent import done_summary

    assert done_summary({"done": True, "summary": "a"}) == "a"
    assert done_summary({"tool": "done", "summary": "b"}) == "b"
    assert done_summary({"tool": "done", "args": {"summary": "c"}}) == "c"
    assert done_summary({"tool": "read_file", "args": {"path": "x"}}) is None
    assert done_summary([2608038142.43914]) is None
    assert done_summary("nope") is None


def test_agent_rejects_json_array_then_recovers():
    events: list[dict] = []
    provider = ScriptedProvider(
        [
            "[2608038142.43914]",
            json.dumps({"done": True, "summary": "recovered"}),
        ]
    )
    agent = Agent(provider, tools={})
    assert agent.run("g", on_step=events.append) == "recovered"
    assert events[1]["kind"] == "invalid_json"
    assert "Expected a JSON object" in events[1]["error"]
    assert "JSON object" in provider.calls[1]["messages"][-1]["content"]


def test_truncate_for_messages():
    from agent import _MSG_OBS_LIMIT, truncate_for_messages

    assert truncate_for_messages("short") == "short"
    big = "x" * (_MSG_OBS_LIMIT + 10)
    out = truncate_for_messages(big)
    assert out.endswith("\n...[truncated]")
    assert len(out) == _MSG_OBS_LIMIT + len("\n...[truncated]")


def test_compact_messages_keeps_goal_and_tail():
    from agent import compact_messages

    msgs = [{"role": "user", "content": "GOAL"}]
    for i in range(30):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "user", "content": f"u{i}"})
    out = compact_messages(msgs, keep_last=12)
    assert out[0]["content"] == "GOAL"
    assert "omitted" in out[1]["content"]
    assert out[1]["content"].startswith("[omitted")
    assert len(out) == 1 + 1 + 12  # goal + stub + last 12
    assert out[-1]["content"] == "u29"
    assert out[-2]["content"] == "a29"


def test_compact_messages_noop_when_short():
    from agent import compact_messages

    msgs = [
        {"role": "user", "content": "GOAL"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    assert compact_messages(msgs, keep_last=12) == msgs


def test_history_assistant_redacts_write_content():
    from agent import history_assistant_text

    raw = json.dumps(
        {
            "tool": "write_file",
            "args": {"path": "x.py", "content": "y" * 200},
        }
    )
    action = json.loads(raw)
    stored = history_assistant_text(raw, action)
    parsed = json.loads(stored)
    assert parsed["tool"] == "write_file"
    assert parsed["args"]["path"] == "x.py"
    assert parsed["args"]["content"].endswith("…")
    assert len(parsed["args"]["content"]) <= 81


def test_agent_message_history_truncates_huge_observation(tmp_path):
    from agent import _MSG_OBS_LIMIT

    tools = make_tools(root=tmp_path)
    blob = "Z" * (_MSG_OBS_LIMIT + 500)
    (tmp_path / "big.txt").write_text(blob, encoding="utf-8")
    provider = ScriptedProvider(
        [
            json.dumps({"tool": "read_file", "args": {"path": "big.txt"}}),
            json.dumps({"done": True, "summary": "read it"}),
        ]
    )
    agent = Agent(provider, tools=tools)
    assert agent.run("read big", keep_last=12) == "read it"

    second = provider.calls[1]["messages"]
    joined = "\n".join(m["content"] for m in second)
    assert blob not in joined
    assert "[truncated]" in joined
    assert joined.count("Z") <= _MSG_OBS_LIMIT


def test_agent_compacts_long_history(tmp_path):
    tools = make_tools(root=tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    replies = []
    for i in range(8):
        replies.append(
            json.dumps({"tool": "read_file", "args": {"path": "a.txt"}})
        )
    replies.append(json.dumps({"done": True, "summary": "done"}))
    provider = ScriptedProvider(replies)
    agent = Agent(provider, tools=tools)
    assert agent.run("loop reads", keep_last=4) == "done"

    last_call_msgs = provider.calls[-1]["messages"]
    assert last_call_msgs[0]["content"] == "loop reads"
    # With keep_last=4, after many turns we should see the omit stub
    assert any("omitted" in m["content"] for m in last_call_msgs)
    assert len(last_call_msgs) <= 1 + 1 + 4
