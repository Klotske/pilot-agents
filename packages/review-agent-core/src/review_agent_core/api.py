"""High-level entry points for the git review agent."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from crewai import LLM, Crew, CrewOutput
from crewai.types.streaming import CrewStreamingOutput

from .config import GitConfig, LLMConfig
from .crew import (
    build_code_review_crew,
    build_llm,
    build_review_guide_crew,
    build_review_pipeline_crew,
    build_test_quality_crew,
)
from .git_repo import BranchDiffScope, GitRepo
from .render import render_findings, render_merged
from .schemas import MergedReport, ReviewerFindings, parse_findings, parse_merged
from .tracing import record_usage, trace_note


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


def _kickoff(
    crew: Crew, repo: GitRepo, scope: BranchDiffScope
) -> CrewOutput:
    """Run ``crew`` and record its token usage."""
    result = crew.kickoff(inputs={  # pyright: ignore[reportUnknownMemberType]
        "repo_path": str(repo.path),
        "base_ref": scope.base,
        "head_ref": scope.head,
    })
    record_usage(crew.usage_metrics)
    # Streaming is never enabled here, but the return type is a union.
    if isinstance(result, CrewStreamingOutput):
        return result.result
    return result


def _review(
    git: GitConfig,
    llm: LLMConfig,
    build_crew: CrewBuilder,
    *,
    title: str | None = None,
    verbose: bool = False,
) -> str:
    repo = _resolve_repo(git)
    scope = _resolve_scope(git, repo)
    result = _kickoff(build_crew(build_llm(llm), repo, scope, verbose=verbose), repo, scope)

    structured = result.pydantic
    if not isinstance(structured, ReviewerFindings):
        # CrewAI could not validate the answer; try it ourselves before giving up.
        structured = parse_findings(result.raw or "")
    if structured is not None:
        return render_findings(structured, title=title or "Findings")
    trace_note("reviewer answer is not structured; using raw text")
    return result.raw or ""


def review_code(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the Code Reviewer and return a Markdown report."""
    return _review(
        git, llm, build_code_review_crew, title="Code Review", verbose=verbose
    )


def review_tests(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the Test Quality Reviewer and return a Markdown report."""
    return _review(
        git, llm, build_test_quality_crew, title="Test Quality Review", verbose=verbose
    )


def review_guide(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the Review Guide and return a Markdown report."""
    return _review(git, llm, build_review_guide_crew, verbose=verbose)


def review_all(*, git: GitConfig, llm: LLMConfig, verbose: bool = False) -> str:
    """Run the full pipeline and return one consolidated Markdown report.

    The guide runs first, then the code and test reviewers run in parallel on
    its output, and the lead merges all three reports.
    """
    repo = _resolve_repo(git)
    scope = _resolve_scope(git, repo)

    crew = build_review_pipeline_crew(build_llm(llm), repo, scope, verbose=verbose)
    result = _kickoff(crew, repo, scope)

    tasks = list(result.tasks_output)
    guide_report = tasks[0].raw if tasks else ""
    lead_text = tasks[-1].raw if tasks else (result.raw or "")
    merged = result.pydantic
    if not isinstance(merged, MergedReport):
        merged = parse_merged(lead_text or "")

    header = f"# Review: {scope.base}...{scope.head}"
    if merged is not None:
        return f"{header}\n\n{render_merged(merged, guide_report=guide_report)}"

    # The lead's answer could not be validated; keep its text.
    trace_note("lead answer is not structured; using raw text")
    return f"{header}\n\n{lead_text}"
