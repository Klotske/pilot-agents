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
        """Return the default base ref: local ``main``/``master``, else the remote one.

        A fresh clone usually has both, but a repository cloned with a narrow
        refspec may know only ``origin/main``.
        """
        for candidate in ("main", "master", "origin/main", "origin/master"):
            if self._exists(candidate):
                return candidate
        raise GitError(
            "cannot auto-detect base ref (tried main, master, origin/main, "
            "origin/master); pass --base"
        )

    def resolve_ref(self, ref: str) -> str:
        """Return an existing ref for ``ref``, falling back to ``origin/<ref>``.

        A branch that exists only on the remote is what a fresh clone has: the
        local ``feature/x`` does not resolve, and ``git diff main...feature/x``
        fails with "ambiguous argument". The agent then improvises (reads files
        by guessed paths) instead of reviewing a diff, so the fallback -- and,
        failing that, an error that names the available refs -- matters more
        than it looks.
        """
        for candidate in (ref, f"origin/{ref}", f"refs/remotes/origin/{ref}"):
            if self._exists(candidate):
                return candidate
        raise GitError(
            f"cannot resolve ref {ref!r} in {self.path}; "
            f"available refs: {', '.join(self._refs()) or '(none)'}"
        )

    def _exists(self, ref: str) -> bool:
        """Whether ``ref`` resolves to a commit in this repository."""
        return bool(
            self._run(
                "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False
            ).strip()
        )

    def _refs(self) -> list[str]:
        """Return local and remote branch names, for error messages."""
        out = self._run(
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
            "refs/remotes",
            check=False,
        )
        return [line for line in out.splitlines() if line.strip()]

    def changed_files(self, scope: BranchDiffScope) -> list[str]:
        out = self._run("diff", "--name-only", scope.diff_spec)
        return [line for line in out.splitlines() if line.strip()]

    def list_files(self, ref: str, path: str = "") -> list[str]:
        """List tracked files at ``ref``, optionally under ``path``.

        This is how an agent learns what exists instead of guessing file names:
        unlike ``read_file`` it accepts a directory, and unlike
        ``get_changed_files`` it covers the whole tree, not just the diff.
        """
        args = ["ls-tree", "-r", "--name-only", ref]
        prefix = path.strip().strip("/")
        if prefix:
            args.extend(["--", prefix])
        out = self._run(*args)
        return [line for line in out.splitlines() if line.strip()]

    def diff(self, scope: BranchDiffScope) -> str:
        return self._run("diff", scope.diff_spec)

    def diff_stat(self, scope: BranchDiffScope) -> str:
        return self._run("diff", "--stat", scope.diff_spec)

    def log(self, scope: BranchDiffScope) -> str:
        return self._run("log", "--oneline", scope.log_spec)

    def read_file(self, path: str, ref: str) -> str:
        """Return the text content of ``path`` at ``ref`` (or a binary marker)."""
        rel = path.strip().lstrip("/")
        is_tree = self._run("cat-file", "-t", f"{ref}:{rel}", check=False).strip() == "tree"
        if is_tree or rel in ("", ".", "..") or rel.endswith("/"):
            # ``git show`` fails on a tree with a message about the path, which
            # sends an agent guessing. Name the tool that answers the question.
            raise GitError(
                f"{path!r} is a directory, not a file; use list_files to see "
                "what it contains, or get_changed_files to see what changed"
            )
        out = self._run("show", f"{ref}:{rel}", check=False)
        if not out:
            self._fail_missing(path, ref)
        if "\x00" in out[:1000]:
            return "<no text content (binary)>"
        return out[:MAX_FILE_CHARS]

    def _fail_missing(self, path: str, ref: str) -> None:
        """Raise for a failed ``read_file``, naming the way out.

        Raw git text ("path 'x' does not exist in 'main'") makes an agent retry
        with guessed variants; pointing at what the tools can answer stops that.
        """
        if not self._exists(ref):
            raise GitError(
                f"ref {ref!r} was not found in this repository; "
                f"available refs: {', '.join(self._refs()) or '(none)'}"
            )
        raise GitError(
            f"{path!r} was not found at {ref!r}; use list_files({ref!r}, path) to "
            "see what exists, or get_changed_files to see what changed"
        )
