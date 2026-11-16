"""Build review Crews: single-mode reviewers and the full pipeline.

A single-mode crew runs one reviewer agent over the git tools. The pipeline
crew runs all of them in one Crew, wired as a DAG:

    Review Guide  ->  Code Reviewer  (parallel)
                  ->  Test Quality Reviewer  (parallel)
                  ->  Review Lead (merges the three reports)

Parallelism uses CrewAI's ``Task.async_execution``: async tasks are started as
asyncio tasks and awaited by the next synchronous task, so the lead always sees
both reports. Downstream tasks receive the guide's output through
``Task.context``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel

from .config import LLMConfig
from .git_repo import BranchDiffScope, GitRepo
from .schemas import MergedReport, ReviewerFindings
from .tools.git_tools import make_git_tools

REPORT_OUTPUT = "A concise Markdown report following the sections in the instructions."
FINDINGS_OUTPUT = "Structured findings for every distinct problem."
MERGED_OUTPUT = "A consolidated, deduplicated report ordered by severity."

# Structured output goes through ``Task.output_pydantic``, which makes CrewAI
# append the pydantic schema to the prompt and validate the answer afterwards.
#
# Gotcha: CrewAI only skips the provider-level ``response_format`` request when
# the agent has at least one tool (``effective_response_model = None if
# self.original_tools else ...``). Without tools it sends a ``json_schema``
# response_format, which some OpenAI-compatible gateways reject with HTTP 400.
# That is why the lead gets the git tools too, and why ``api`` still parses the
# raw answer as a fallback.


def load_skill(name: str) -> str:
    """Read a ``skills/<name>.md`` prompt module."""
    return files("review_agent_core").joinpath("skills", f"{name}.md").read_text(encoding="utf-8")


def build_llm(cfg: LLMConfig) -> LLM:
    """Build a CrewAI ``LLM`` for an OpenAI-compatible endpoint."""
    return LLM(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
    )


@dataclass(frozen=True)
class _AgentSpec:
    """Identity of a review agent, shared by the pipeline and single-mode crews."""

    role: str
    goal: str
    backstory: str
    skill: str


CODE_REVIEWER = _AgentSpec(
    role="Code Reviewer",
    goal="Find potential bugs, risks, and complex areas in the change.",
    backstory=(
        "You are an experienced code reviewer who reads diffs carefully "
        "and flags real, concrete problems."
    ),
    skill="code_reviewer",
)

TEST_QUALITY_REVIEWER = _AgentSpec(
    role="Test Quality Reviewer",
    goal="Analyze test coverage and test quality for the change.",
    backstory=(
        "You are a testing expert who spots untested scenarios and weak tests "
        "that would let bugs through."
    ),
    skill="test_quality",
)

REVIEW_GUIDE = _AgentSpec(
    role="Review Guide",
    goal="Explain the change and its invariants, and prepare useful comments for reviewers.",
    backstory=(
        "You are a reviewer's assistant who distills business logic into "
        "clear, actionable guidance."
    ),
    skill="review_guide",
)

REVIEW_LEAD = _AgentSpec(
    role="Review Lead",
    goal="Merge the specialized review reports into one coherent, deduplicated report.",
    backstory=(
        "You are a senior reviewer who consolidates findings from "
        "specialized reviewers into a single actionable report."
    ),
    skill="review_lead",
)


def _make_agent(
    spec: _AgentSpec,
    llm: LLM,
    tools: list[BaseTool],
    *,
    verbose: bool = False,
) -> Agent:
    return Agent(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        tools=tools,
        llm=llm,
        verbose=verbose,
    )


def _make_task(
    spec: _AgentSpec,
    agent: Agent,
    *,
    expected_output: str = REPORT_OUTPUT,
    output_pydantic: type[BaseModel] | None = None,
    async_execution: bool = False,
    context: list[Task] | None = None,
) -> Task:
    return Task(
        description=load_skill(spec.skill),
        expected_output=expected_output,
        agent=agent,
        output_pydantic=output_pydantic,
        async_execution=async_execution,
        context=context,
    )


def _build_single_crew(
    llm: LLM,
    repo: GitRepo,
    scope: BranchDiffScope,
    spec: _AgentSpec,
    *,
    expected_output: str = REPORT_OUTPUT,
    output_pydantic: type[BaseModel] | None = None,
    verbose: bool = False,
) -> Crew:
    agent = _make_agent(spec, llm, make_git_tools(repo, scope), verbose=verbose)
    task = _make_task(
        spec,
        agent,
        expected_output=expected_output,
        output_pydantic=output_pydantic,
    )
    return Crew(agents=[agent], tasks=[task], verbose=verbose)


def build_code_review_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Code Reviewer: potential bugs, risks, and complex areas."""
    return _build_single_crew(
        llm,
        repo,
        scope,
        CODE_REVIEWER,
        expected_output=FINDINGS_OUTPUT,
        output_pydantic=ReviewerFindings,
        verbose=verbose,
    )


def build_test_quality_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Test Quality Reviewer: untested scenarios and weak tests."""
    return _build_single_crew(
        llm,
        repo,
        scope,
        TEST_QUALITY_REVIEWER,
        expected_output=FINDINGS_OUTPUT,
        output_pydantic=ReviewerFindings,
        verbose=verbose,
    )


def build_review_guide_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Review Guide: business logic, invariants, and comments for reviewers."""
    return _build_single_crew(llm, repo, scope, REVIEW_GUIDE, verbose=verbose)


def build_review_pipeline_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Run guide -> (code, tests in parallel) -> lead as a single Crew.

    Each reviewer gets its own tool instances: ``BaseTool`` keeps mutable
    per-instance state (usage counters and a lock), which must not be shared by
    agents running concurrently.
    """
    guide_agent = _make_agent(
        REVIEW_GUIDE, llm, make_git_tools(repo, scope), verbose=verbose
    )
    code_agent = _make_agent(
        CODE_REVIEWER, llm, make_git_tools(repo, scope), verbose=verbose
    )
    tests_agent = _make_agent(
        TEST_QUALITY_REVIEWER, llm, make_git_tools(repo, scope), verbose=verbose
    )
    # The lead gets the git tools so structured output stays on the tool-present
    # path (see the note next to REPORT_OUTPUT). It uses them only to verify a
    # finding it is about to merge.
    lead_agent = _make_agent(
        REVIEW_LEAD, llm, make_git_tools(repo, scope), verbose=verbose
    )

    guide_task = _make_task(REVIEW_GUIDE, guide_agent)
    code_task = _make_task(
        CODE_REVIEWER,
        code_agent,
        expected_output=FINDINGS_OUTPUT,
        output_pydantic=ReviewerFindings,
        async_execution=True,
        context=[guide_task],
    )
    tests_task = _make_task(
        TEST_QUALITY_REVIEWER,
        tests_agent,
        expected_output=FINDINGS_OUTPUT,
        output_pydantic=ReviewerFindings,
        async_execution=True,
        context=[guide_task],
    )
    lead_task = _make_task(
        REVIEW_LEAD,
        lead_agent,
        expected_output=MERGED_OUTPUT,
        output_pydantic=MergedReport,
        context=[guide_task, code_task, tests_task],
    )

    return Crew(
        agents=[guide_agent, code_agent, tests_agent, lead_agent],
        tasks=[guide_task, code_task, tests_task, lead_task],
        process=Process.sequential,
        verbose=verbose,
    )
