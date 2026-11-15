"""CrewAI tools wrapping read-only git access."""

from __future__ import annotations

from crewai.tools import BaseTool, tool

from ..git_repo import BranchDiffScope, GitRepo

MAX_DIFF_CHARS = 60_000


def make_git_tools(repo: GitRepo, scope: BranchDiffScope) -> list[BaseTool]:
    """Build the read-only git tools bound to ``repo`` and ``scope``."""

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

    @tool("read_file")
    def read_file(path: str, ref: str) -> str:
        """Read the text content of ``path`` at ``ref`` (a branch, commit, or HEAD)."""
        return repo.read_file(path, ref)

    return [get_diff, get_diff_stat, get_changed_files, get_commits, read_file]
