import pytest

from tools_registry import (
    MAX_READ,
    make_tools,
    read_file,
    resolve_repo_path,
    write_file,
)


def test_resolve_repo_path_joins_under_root(tmp_path):
    p = resolve_repo_path("sandbox/hello.py", root=tmp_path)
    assert p == (tmp_path / "sandbox" / "hello.py").resolve()


def test_resolve_repo_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        resolve_repo_path("../outside", root=tmp_path)


def test_write_then_read_roundtrip(tmp_path):
    msg = write_file("a/b.txt", "hello", root=tmp_path)
    assert "Wrote 5 chars" in msg
    assert read_file("a/b.txt", root=tmp_path) == "hello"


def test_read_truncates_large_file(tmp_path):
    big = "x" * (MAX_READ + 50)
    write_file("big.txt", big, root=tmp_path)
    out = read_file("big.txt", root=tmp_path)
    assert out.startswith("x" * MAX_READ)
    assert "[truncated" in out


def test_make_tools_instances_run(tmp_path):
    tools = make_tools(root=tmp_path)
    assert set(tools) == {"read_file", "write_file"}

    tools["write_file"].run({"path": "hi.txt", "content": "yo"})
    assert tools["read_file"].run({"path": "hi.txt"}) == "yo"


def test_write_file_validates_args(tmp_path):
    tools = make_tools(root=tmp_path)
    with pytest.raises(Exception):  # noqa: B017
        tools["write_file"].run({"path": "x.txt"})  # missing content
