import json

from orchestrator import BUILD_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT, Orchestrator
from test_agent import ScriptedProvider
from tools_registry import make_tools


def test_orchestrator_plan_then_build(tmp_path):
    tools = make_tools(root=tmp_path)
    (tmp_path / "notes.txt").write_text("base", encoding="utf-8")

    provider = ScriptedProvider(
        [
            # plan phase: read then done
            json.dumps({"tool": "read_file", "args": {"path": "notes.txt"}}),
            json.dumps(
                {
                    "done": True,
                    "summary": "Create hello.py that prints hello",
                }
            ),
            # build phase: write then done
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {
                        "path": "hello.py",
                        "content": "print('hello')\n",
                    },
                }
            ),
            json.dumps({"done": True, "summary": "created hello.py"}),
        ]
    )

    orch = Orchestrator(provider, tools)
    result = orch.run("add a hello script")

    assert result.status == "ok"
    assert "Create hello.py" in result.plan
    assert result.report == "created hello.py"
    assert (tmp_path / "hello.py").read_text() == "print('hello')\n"

    # first call uses plan prompt; first build call uses build prompt
    assert provider.calls[0]["system"] == PLAN_SYSTEM_PROMPT
    assert provider.calls[2]["system"] == BUILD_SYSTEM_PROMPT
    assert "PLAN:" in provider.calls[2]["messages"][0]["content"]
    assert "Create hello.py" in provider.calls[2]["messages"][0]["content"]


def test_orchestrator_plan_cannot_write(tmp_path):
    tools = make_tools(root=tmp_path)
    provider = ScriptedProvider(
        [
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "evil.py", "content": "x"},
                }
            ),
            json.dumps({"done": True, "summary": "plan without writes"}),
            json.dumps({"done": True, "summary": "build noop"}),
        ]
    )

    orch = Orchestrator(provider, tools)
    result = orch.run("try to write in plan")

    assert not (tmp_path / "evil.py").exists()
    assert "Unknown tool" in provider.calls[1]["messages"][-1]["content"]
    assert result.plan == "plan without writes"
    assert result.report == "build noop"
