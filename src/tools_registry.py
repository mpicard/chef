from pathlib import Path

from pydantic import BaseModel, Field

from typed_tool import TypedTool

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_READ = 20_000
MAX_LIST = 200

PLAN_TOOLS = frozenset({"read_file", "list_dir", "glob"})


class ReadArgs(BaseModel):
    path: str = Field(..., description="Path relative to repo root")


class WriteArgs(ReadArgs):
    content: str = Field(..., description="Full file contents to write")


class ListDirArgs(BaseModel):
    path: str = Field(
        ".",
        description="Directory relative to repo root (default: '.')",
    )


class GlobArgs(BaseModel):
    pattern: str = Field(
        ...,
        description="Glob pattern relative to repo root, e.g. 'src/**/*.py'",
    )


class EditArgs(BaseModel):
    path: str = Field(..., description="Path relative to repo root")
    old_string: str = Field(..., description="Exact text to find (must be unique)")
    new_string: str = Field(..., description="Replacement text")


def resolve_repo_path(path: str, root: Path | None = None) -> Path:
    base = (root or REPO_ROOT).resolve()
    p = (base / path).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f"Path escapes repo: {path!r}")
    return p


def read_file(path: str, root: Path | None = None) -> str:
    p = resolve_repo_path(path, root=root)
    text = p.read_text(encoding="utf-8")
    if len(text) > MAX_READ:
        return text[:MAX_READ] + f"\n...[truncated, {len(text)} chars total]"
    return text


def write_file(path: str, content: str, root: Path | None = None) -> str:
    p = resolve_repo_path(path, root=root)
    p.parent.mkdir(parents=True, exist_ok=True)
    _ = p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


def list_dir(path: str = ".", root: Path | None = None) -> str:
    base = (root or REPO_ROOT).resolve()
    p = resolve_repo_path(path, root=root)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path!r}")
    entries: list[str] = []
    for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
        rel = child.relative_to(base).as_posix()
        entries.append(rel + ("/" if child.is_dir() else ""))
        if len(entries) >= MAX_LIST:
            entries.append(f"...[truncated at {MAX_LIST} entries]")
            break
    return "\n".join(entries) if entries else "(empty)"


def glob_files(pattern: str, root: Path | None = None) -> str:
    base = (root or REPO_ROOT).resolve()
    if pattern.startswith(("/", "..")):
        # still allow patterns like 'src/../src/*.py'
        pass
    matches: list[str] = []
    for hit in sorted(base.glob(pattern)):
        resolved = hit.resolve()
        if not resolved.is_relative_to(base):
            continue
        if not resolved.is_file():
            continue
        matches.append(resolved.relative_to(base).as_posix())
        if len(matches) >= MAX_LIST:
            matches.append(f"...[truncated at {MAX_LIST} matches]")
            break
    return "\n".join(matches) if matches else "(no matches)"


def edit_file(
    path: str, old_string: str, new_string: str, root: Path | None = None
) -> str:
    p = resolve_repo_path(path, root=root)
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in {path!r}")
    if count > 1:
        raise ValueError(f"old_string found {count} times in {path!r}; must be unique")
    _ = p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"Edited {path} ({len(old_string)} → {len(new_string)} chars)"


def make_tools(root: Path | None = None) -> dict[str, TypedTool]:
    """Build tool instances rooted at ``root`` (defaults to repo root)."""

    def _read(path: str) -> str:
        return read_file(path, root=root)

    def _write(path: str, content: str) -> str:
        return write_file(path, content, root=root)

    def _list(path: str = ".") -> str:
        return list_dir(path, root=root)

    def _glob(pattern: str) -> str:
        return glob_files(pattern, root=root)

    def _edit(path: str, old_string: str, new_string: str) -> str:
        return edit_file(path, old_string, new_string, root=root)

    return {
        "read_file": TypedTool(
            name="read_file",
            description="Read a file relative to the workspace root.",
            args_model=ReadArgs,
            fn=_read,
            cost_hint=0.1,
        ),
        "write_file": TypedTool(
            name="write_file",
            description="Create or overwrite a full file relative to the workspace root.",
            args_model=WriteArgs,
            fn=_write,
            cost_hint=0.2,
        ),
        "list_dir": TypedTool(
            name="list_dir",
            description="List files and directories under a path (relative to workspace root).",
            args_model=ListDirArgs,
            fn=_list,
            cost_hint=0.1,
        ),
        "glob": TypedTool(
            name="glob",
            description="Find files matching a glob pattern relative to the workspace root.",
            args_model=GlobArgs,
            fn=_glob,
            cost_hint=0.1,
        ),
        "edit_file": TypedTool(
            name="edit_file",
            description=(
                "Replace a unique old_string with new_string in an existing file. "
                "Prefer this over write_file for small edits."
            ),
            args_model=EditArgs,
            fn=_edit,
            cost_hint=0.2,
        ),
    }


TOOLS = make_tools()
