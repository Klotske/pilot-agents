"""High-level entry points for the git review agent."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from crewai import LLM, Crew, CrewOutput
from crewai.hooks import InterceptionPoint
from crewai.hooks.dispatch import register_scoped, scoped_hooks
from crewai.types.streaming import CrewStreamingOutput

from .compaction import Compactor
from .config import CompactionConfig, GitConfig, LLMConfig
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
from .tracing import ContextBudget, record_usage, set_context_budget, trace_note


def _resolve_repo(cfg: GitConfig) -> GitRepo:
    path = Path(cfg.repo_path) if cfg.repo_path else Path.cwd()
    return GitRepo(path)


def _resolve_scope(cfg: GitConfig, repo: GitRepo) -> BranchDiffScope:
    """Resolve ``base``/``head`` to refs git can actually diff.

    A branch that exists only as ``origin/<name>`` is resolved to that form
    rather than failing every diff tool with "ambiguous argument".
    """
    head = repo.resolve_ref(cfg.head_ref or "HEAD")
    base = repo.resolve_ref(cfg.base_ref) if cfg.base_ref else repo.default_base()
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


def _settings(compaction: CompactionConfig | None) -> CompactionConfig:
    """Return the compaction settings for a run (defaults when unspecified)."""
    return compaction or CompactionConfig()


def _compactor(cfg: LLMConfig, compaction: CompactionConfig | None) -> Compactor | None:
    """Build the compaction hook for a run, or ``None`` when it is disabled.

    The context window is required rather than guessed: CrewAI falls back to a
    catalog value (4096 tokens for an unknown gateway alias), which would make
    compaction trigger far too early. The CLI fills
    ``LLMConfig.context_window_size`` from ``REVIEW_AGENT_CONTEXT_WINDOW``.
    """
    settings = _settings(compaction)
    if not settings.enabled:
        return None
    if cfg.context_window_size is None:
        raise ValueError(
            "history compaction needs the model's context window: set "
            "LLMConfig.context_window_size (the CLI reads "
            "REVIEW_AGENT_CONTEXT_WINDOW), or pass "
            "CompactionConfig(enabled=False)"
        )
    return Compactor(
        settings, model=cfg.model, window_tokens=cfg.context_window_size
    )


def _budget(
    cfg: LLMConfig, compaction: CompactionConfig | None
) -> ContextBudget | None:
    """Describe the run's window for the per-call ``ctx`` trace stats.

    Unlike the compactor this is not optional: the numbers are useful even when
    compaction is off, as long as the window is known.
    """
    if cfg.context_window_size is None:
        return None
    settings = _settings(compaction)
    return ContextBudget.from_window(
        cfg.context_window_size, settings.trigger_ratio, settings.target_ratio
    )


def _kickoff(
    crew: Crew,
    repo: GitRepo,
    scope: BranchDiffScope,
    compactor: Compactor | None = None,
    budget: ContextBudget | None = None,
) -> CrewOutput:
    """Run ``crew`` and record its token usage.

    The compaction hook is registered in an execution scope instead of
    globally: it then applies to this run only (including the parallel reviewer
    tasks, which copy the context) and disappears when the run ends.
    """
    inputs = {
        "repo_path": str(repo.path),
        "base_ref": scope.base,
        "head_ref": scope.head,
    }
    set_context_budget(budget)
    if compactor is None:
        result = crew.kickoff(inputs=inputs)  # pyright: ignore[reportUnknownMemberType]
    else:
        with scoped_hooks():
            register_scoped(InterceptionPoint.PRE_MODEL_CALL, compactor.hook)
            # Announce the budget: otherwise a run whose history never reaches
            # the trigger looks exactly like a compactor that is not running.
            trace_note(compactor.describe())
            result = crew.kickoff(inputs=inputs)  # pyright: ignore[reportUnknownMemberType]
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
    compaction: CompactionConfig | None = None,
) -> str:
    repo = _resolve_repo(git)
    scope = _resolve_scope(git, repo)
    compactor = _compactor(llm, compaction)
    crew = build_crew(build_llm(llm), repo, scope, verbose=verbose)
    result = _kickoff(crew, repo, scope, compactor, _budget(llm, compaction))

    structured = result.pydantic
    if not isinstance(structured, ReviewerFindings):
        # CrewAI could not validate the answer; try it ourselves before giving up.
        structured = parse_findings(result.raw or "")
    if structured is not None:
        return render_findings(structured, title=title or "Findings")
    trace_note("reviewer answer is not structured; using raw text")
    return result.raw or ""


def review_code(
    *,
    git: GitConfig,
    llm: LLMConfig,
    verbose: bool = False,
    compaction: CompactionConfig | None = None,
) -> str:
    """Run the Code Reviewer and return a Markdown report."""
    return _review(
        git,
        llm,
        build_code_review_crew,
        title="Code Review",
        verbose=verbose,
        compaction=compaction,
    )


def review_tests(
    *,
    git: GitConfig,
    llm: LLMConfig,
    verbose: bool = False,
    compaction: CompactionConfig | None = None,
) -> str:
    """Run the Test Quality Reviewer and return a Markdown report."""
    return _review(
        git,
        llm,
        build_test_quality_crew,
        title="Test Quality Review",
        verbose=verbose,
        compaction=compaction,
    )


def review_guide(
    *,
    git: GitConfig,
    llm: LLMConfig,
    verbose: bool = False,
    compaction: CompactionConfig | None = None,
) -> str:
    """Run the Review Guide and return a Markdown report."""
    return _review(
        git, llm, build_review_guide_crew, verbose=verbose, compaction=compaction
    )


def review_all(
    *,
    git: GitConfig,
    llm: LLMConfig,
    verbose: bool = False,
    compaction: CompactionConfig | None = None,
) -> str:
    """Run the full pipeline and return one consolidated Markdown report.

    The guide runs first, then the code and test reviewers run in parallel on
    its output, and the lead merges all three reports.
    """
    repo = _resolve_repo(git)
    scope = _resolve_scope(git, repo)

    compactor = _compactor(llm, compaction)
    crew = build_review_pipeline_crew(build_llm(llm), repo, scope, verbose=verbose)
    result = _kickoff(crew, repo, scope, compactor, _budget(llm, compaction))

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
