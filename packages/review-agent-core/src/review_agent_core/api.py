"""High-level entry points for the git review agent."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from crewai import LLM, Crew

from .config import GitConfig, LLMConfig
from .crew import (
    build_code_review_crew,
    build_llm,
    build_review_guide_crew,
    build_review_lead_crew,
    build_test_quality_crew,
)
from .git_repo import BranchDiffScope, GitRepo
from .tracing import record_usage


def _resolve_repo(cfg: GitConfig) -> GitRepo:
    path = Path(cfg.repo_path) if cfg.repo_path else Path.cwd()
    return GitRepo(path)


def _resolve_scope(cfg: GitConfig, repo: GitRepo) -> BranchDiffScope:
    head = cfg.head_ref or "HEAD"
    base = cfg.base_ref or repo.default_base()
    return BranchDiffScope(base=base, head=head)


class CrewBuilder(Protocol):
    """A factory that assembles a single review Crew."""

    def __call__(
        self,
        llm: LLM,
        repo: GitRepo,
        scope: BranchDiffScope,
        verbose: bool = False,
    ) -> Crew: ...


def _review(
    git: GitConfig,
    llm: LLMConfig,
    build_crew: CrewBuilder,
    verbose: bool = False,
) -> str:
    repo = _resolve_repo(git)
    scope = _resolve_scope(git, repo)
    crew = build_crew(build_llm(llm), repo, scope, verbose=verbose)
    result = crew.kickoff(inputs={  # pyright: ignore[reportUnknownMemberType]
        "repo_path": str(repo.path),
        "base_ref": scope.base,
        "head_ref": scope.head,
    })
    record_usage(crew.usage_metrics)
    return str(result)


def review_code(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the Code Reviewer and return a Markdown report."""
    return _review(git, llm, build_code_review_crew, verbose=verbose)


def review_tests(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the Test Quality Reviewer and return a Markdown report."""
    return _review(git, llm, build_test_quality_crew, verbose=verbose)


def review_guide(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the Review Guide and return a Markdown report."""
    return _review(git, llm, build_review_guide_crew, verbose=verbose)


def review_all(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the three reviewers, then a lead that merges them into one report."""
    repo = _resolve_repo(git)
    scope = _resolve_scope(git, repo)

    code_report = review_code(git=git, llm=llm, verbose=verbose)
    tests_report = review_tests(git=git, llm=llm, verbose=verbose)
    guide_report = review_guide(git=git, llm=llm, verbose=verbose)

    lead_crew = build_review_lead_crew(build_llm(llm), verbose=verbose)
    result = lead_crew.kickoff(inputs={  # pyright: ignore[reportUnknownMemberType]
        "repo_path": str(repo.path),
        "base_ref": scope.base,
        "head_ref": scope.head,
        "code_report": code_report,
        "tests_report": tests_report,
        "guide_report": guide_report,
    })
    record_usage(lead_crew.usage_metrics)
    lead_report = str(result)

    return f"# Review: {scope.base}...{scope.head}\n\n{lead_report}"
