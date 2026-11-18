"""Tests for the read-only git wrapper and the review scope."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from review_agent_core.git_repo import BranchDiffScope, GitError, GitRepo

from .helpers import commit_all, git, init_repo, write

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"


def test_not_a_repo(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        GitRepo(tmp_path / "not-a-repo")


def test_default_base_main(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "a.txt", "x\n")
    commit_all(repo_path, "init")
    assert GitRepo(repo_path).default_base() == "main"


def test_diff_changed_files_log_and_read_file(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "app.py", "def f():\n    return 1\n")
    commit_all(repo_path, "base")

    repo = GitRepo(repo_path)
    git(repo_path, "checkout", "-q", "-b", "feature")
    write(repo_path, "app.py", "def f():\n    return 2\n")
    write(repo_path, "new.py", "x = 1\n")
    commit_all(repo_path, "change return value")

    scope = BranchDiffScope(base="main", head="feature")

    assert set(repo.changed_files(scope)) == {"app.py", "new.py"}

    diff = repo.diff(scope)
    assert "return 2" in diff
    assert "return 1" in diff
    assert "new.py" in diff

    assert "app.py" in repo.diff_stat(scope)
    assert "change return value" in repo.log(scope)

    assert repo.read_file("app.py", "feature") == "def f():\n    return 2\n"
    assert repo.read_file("app.py", "main") == "def f():\n    return 1\n"


def test_read_file_binary(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    (repo_path / "b.bin").write_bytes(b"\x00\x01\x02")
    commit_all(repo_path, "binary")

    assert GitRepo(repo_path).read_file("b.bin", "HEAD") == "<no text content (binary)>"


def test_branch_diff_over_fixture(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    shutil.copytree(FIXTURE, target)
    init_repo(target)
    commit_all(target, "base: initial state")

    repo = GitRepo(target)
    git(target, "checkout", "-q", "-b", "feature")
    write(
        target,
        "calculator.py",
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError('zero')\n"
        "    return a / b\n",
    )
    commit_all(target, "guard division by zero")

    scope = BranchDiffScope(base="main", head="feature")

    assert "calculator.py" in repo.changed_files(scope)
    assert "raise ValueError" in repo.diff(scope)
    assert "guard division by zero" in repo.log(scope)


def test_resolve_ref_falls_back_to_the_remote_branch(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "a.txt", "x\n")
    commit_all(repo_path, "init")
    git(repo_path, "branch", "feature")
    git(repo_path, "update-ref", "refs/remotes/origin/feature", "feature")
    git(repo_path, "branch", "-D", "feature")

    repo = GitRepo(repo_path)

    # A fresh clone knows the branch only as origin/<name>.
    assert repo.resolve_ref("feature") == "origin/feature"
    assert repo.resolve_ref("main") == "main"
    assert repo.resolve_ref("HEAD") == "HEAD"


def test_resolve_ref_error_lists_available_refs(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "a.txt", "x\n")
    commit_all(repo_path, "init")

    with pytest.raises(GitError, match="available refs"):
        GitRepo(repo_path).resolve_ref("nope")


def test_read_file_rejects_a_directory(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "src/app.py", "x = 1\n")
    commit_all(repo_path, "init")

    repo = GitRepo(repo_path)

    for path in (".", "", "src", "src/"):
        with pytest.raises(GitError, match="is a directory"):
            repo.read_file(path, "HEAD")


def test_list_files_lists_the_tree_at_a_ref(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "app.py", "x = 1\n")
    write(repo_path, "tests/test_app.py", "def test_x(): pass\n")
    write(repo_path, "docs/readme.md", "# docs\n")
    commit_all(repo_path, "init")

    repo = GitRepo(repo_path)

    assert repo.list_files("HEAD") == ["app.py", "docs/readme.md", "tests/test_app.py"]
    assert repo.list_files("HEAD", "tests") == ["tests/test_app.py"]
    assert repo.list_files("HEAD", "tests/") == ["tests/test_app.py"]
    assert repo.list_files("HEAD", "missing") == []


def test_read_file_error_points_at_list_files(tmp_path: Path) -> None:
    repo_path = init_repo(tmp_path / "repo")
    write(repo_path, "app.py", "x = 1\n")
    commit_all(repo_path, "init")

    repo = GitRepo(repo_path)

    with pytest.raises(GitError, match="list_files"):
        repo.read_file("missing.py", "main")
    with pytest.raises(GitError, match="available refs"):
        repo.read_file("app.py", "nope")
