"""Helpers for building throwaway git repositories in tests."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout


def init_repo(path: Path, branch: str = "main") -> Path:
    """Create an empty git repository (no commits) with the given unborn branch."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    return path


def write(repo: Path, rel: str, content: str) -> None:
    """Write ``content`` (LF newlines) at ``repo/rel``."""
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


def commit_all(repo: Path, message: str) -> Path:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return repo


def create_repo(path: Path, files: dict[str, str], branch: str = "main") -> Path:
    """Create a repo with one commit containing ``files`` (path -> content)."""
    init_repo(path, branch=branch)
    for rel, content in files.items():
        write(path, rel, content)
    commit_all(path, "init")
    return path
