from pathlib import Path

from pydantic import BaseModel, Field

from typed_tool import TypedTool

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_READ = 100_000


class ReadArgs(BaseModel):
    path: str = Field(..., description="Path relative to repo root")


class WriteArgs(ReadArgs):
    content: str = Field(..., description="Full file contents to write")


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
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


def make_tools(root: Path | None = None) -> dict[str, TypedTool]:
    """Build tool instances rooted at ``root`` (defaults to repo root)."""

    def _read(path: str) -> str:
        return read_file(path, root=root)

    def _write(path: str, content: str) -> str:
        return write_file(path, content, root=root)

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
            description="Write a full file relative to the workspace root.",
            args_model=WriteArgs,
            fn=_write,
            cost_hint=0.2,
        ),
    }


TOOLS = make_tools()
