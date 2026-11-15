"""Read-only access to a local git repository via the git CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_CHARS = 20_000


class GitError(RuntimeError):
    """Raised when a git command fails or the path is not a repository."""


@dataclass(frozen=True)
class BranchDiffScope:
    """The changeset under review: ``head`` against the merge-base with ``base``.

    This is the "review this branch" semantics (like a PR diff). Other scopes
    (uncommitted worktree, explicit range, single commit) can be added later as
    sibling dataclasses exposing the same surface the tools consume.
    """

    base: str
    head: str

    @property
    def diff_spec(self) -> str:
        """Three-dot spec: the net change introduced by ``head`` since divergence."""
        return f"{self.base}...{self.head}"

    @property
    def log_spec(self) -> str:
        """Two-dot spec: commits reachable from ``head`` but not from ``base``."""
        return f"{self.base}..{self.head}"


class GitRepo:
    """Thin read-only wrapper over ``git`` for a single repository."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._run("rev-parse", "--is-inside-work-tree")  # raises if not a repo

    # ------------------------------------------------------------------ internal

    def _run(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            [
                "git", "-C", str(self.path),
                "--no-pager",
                "-c", "color.ui=false",
                "-c", "core.quotepath=false",
                *args,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check and result.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed in {self.path}: {result.stderr.strip()}"
            )
        return result.stdout

    # ------------------------------------------------------------------- public

    def default_base(self) -> str:
        """Return the default base ref: ``main`` if present, else ``master``."""
        for candidate in ("main", "master"):
            if self._run("rev-parse", "--verify", "--quiet", candidate, check=False).strip():
                return candidate
        raise GitError(
            "cannot auto-detect base ref (tried 'main' and 'master'); pass --base"
        )

    def changed_files(self, scope: BranchDiffScope) -> list[str]:
        out = self._run("diff", "--name-only", scope.diff_spec)
        return [line for line in out.splitlines() if line.strip()]

    def diff(self, scope: BranchDiffScope) -> str:
        return self._run("diff", scope.diff_spec)

    def diff_stat(self, scope: BranchDiffScope) -> str:
        return self._run("diff", "--stat", scope.diff_spec)

    def log(self, scope: BranchDiffScope) -> str:
        return self._run("log", "--oneline", scope.log_spec)

    def read_file(self, path: str, ref: str) -> str:
        """Return the text content of ``path`` at ``ref`` (or a binary marker)."""
        rel = path.lstrip("/")
        out = self._run("show", f"{ref}:{rel}")
        if "\x00" in out[:1000]:
            return "<no text content (binary)>"
        return out[:MAX_FILE_CHARS]
