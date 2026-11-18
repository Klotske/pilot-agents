"""CrewAI tools wrapping read-only git access."""

from __future__ import annotations

from crewai.tools import BaseTool, tool

from ..git_repo import BranchDiffScope, GitRepo

MAX_DIFF_CHARS = 60_000
MAX_LIST_CHARS = 8_000


def make_git_tools(
    repo: GitRepo,
    scope: BranchDiffScope,
    *,
    only: frozenset[str] | set[str] | None = None,
) -> list[BaseTool]:
    """Build the read-only git tools bound to ``repo`` and ``scope``.

    ``only`` keeps a subset by tool name, for agents that should not explore the
    repository but only check a claim (see the review lead in ``crew.py``).
    """

    @tool("get_diff")
    def get_diff() -> str:
        """Fetch the full unified diff of the changes under review."""
        return repo.diff(scope)[:MAX_DIFF_CHARS]

    @tool("get_diff_stat")
    def get_diff_stat() -> str:
        """Fetch a per-file summary of the diff (files and added/removed lines)."""
        return repo.diff_stat(scope)

    @tool("get_changed_files")
    def get_changed_files() -> str:
        """List the files changed by the review, one path per line."""
        return "\n".join(repo.changed_files(scope))

    @tool("get_commits")
    def get_commits() -> str:
        """List the commits under review (one per line, newest first)."""
        return repo.log(scope)

    @tool("list_files")
    def list_files(ref: str, path: str = "") -> str:
        """List files tracked at ``ref``, optionally only under the ``path`` directory.

        Use this to discover what actually exists before reading: ``read_file``
        takes a single file and rejects directories. ``ref`` is usually the head
        or base ref under review.
        """
        files = repo.list_files(ref, path)
        if not files:
            return f"No files found at {ref!r} under {path!r}."
        listed = "\n".join(files)
        if len(listed) > MAX_LIST_CHARS:
            listed = listed[:MAX_LIST_CHARS] + "\n… (truncated, narrow it down with path)"
        return listed

    @tool("read_file")
    def read_file(path: str, ref: str) -> str:
        """Read the text content of the single file ``path`` at ``ref`` (a branch, commit, or HEAD)."""
        return repo.read_file(path, ref)

    return [
        tool_item
        for tool_item in (
            get_diff,
            get_diff_stat,
            get_changed_files,
            get_commits,
            list_files,
            read_file,
        )
        if only is None or tool_item.name in only
    ]
