import pytest

from tools_registry import (
    MAX_LIST,
    MAX_READ,
    PLAN_TOOLS,
    edit_file,
    glob_files,
    list_dir,
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
    assert set(tools) == {
        "read_file",
        "write_file",
        "list_dir",
        "glob",
        "edit_file",
    }

    tools["write_file"].run({"path": "hi.txt", "content": "yo"})
    assert tools["read_file"].run({"path": "hi.txt"}) == "yo"


def test_write_file_validates_args(tmp_path):
    tools = make_tools(root=tmp_path)
    with pytest.raises(Exception):  # noqa: B017
        tools["write_file"].run({"path": "x.txt"})  # missing content


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y", encoding="utf-8")
    out = list_dir(".", root=tmp_path)
    assert "a.txt" in out.splitlines()
    assert "sub/" in out.splitlines()


def test_list_dir_truncates(tmp_path):
    for i in range(MAX_LIST + 5):
        (tmp_path / f"f{i:03d}.txt").write_text("x", encoding="utf-8")
    out = list_dir(".", root=tmp_path)
    assert f"[truncated at {MAX_LIST} entries]" in out


def test_glob_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "src" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.py").write_text("c", encoding="utf-8")
    out = glob_files("**/*.py", root=tmp_path)
    lines = out.splitlines()
    assert "src/a.py" in lines
    assert "c.py" in lines
    assert "src/b.txt" not in lines


def test_glob_no_matches(tmp_path):
    assert glob_files("**/*.rs", root=tmp_path) == "(no matches)"


def test_edit_file_unique(tmp_path):
    write_file("x.py", "print('a')\nprint('b')\n", root=tmp_path)
    msg = edit_file("x.py", "print('a')", "print('A')", root=tmp_path)
    assert "Edited" in msg
    assert read_file("x.py", root=tmp_path) == "print('A')\nprint('b')\n"


def test_edit_file_missing_raises(tmp_path):
    write_file("x.py", "hello\n", root=tmp_path)
    with pytest.raises(ValueError, match="not found"):
        edit_file("x.py", "nope", "x", root=tmp_path)


def test_edit_file_ambiguous_raises(tmp_path):
    write_file("x.py", "aa\naa\n", root=tmp_path)
    with pytest.raises(ValueError, match="2 times"):
        edit_file("x.py", "aa", "b", root=tmp_path)


def test_plan_tools_exclude_writes():
    tools = make_tools()
    assert "edit_file" not in PLAN_TOOLS
    assert "write_file" not in PLAN_TOOLS
    assert PLAN_TOOLS <= set(tools)
