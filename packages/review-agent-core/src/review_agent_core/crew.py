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
FINDINGS_OUTPUT = (
    "One JSON object matching the schema above: findings plus notes, no prose."
)
MERGED_OUTPUT = (
    "One JSON object matching the schema above: summary, findings, remaining_notes, "
    "no prose."
)

# The lead merges reports that already contain the diff and the file contents, so
# it must not fetch anything: a tool round is a chance to end the task without a
# final answer, which then costs an LLM schema conversion plus a task retry. One
# cheap tool stays attached for protocol reasons (CrewAI only skips the
# provider-level response_format request when an agent has a tool), and the skill
# tells the model not to call it.
LEAD_TOOLS: frozenset[str] = frozenset({"list_files"})
LEAD_MAX_ITER = 2

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


# The provider is pinned instead of letting CrewAI parse the model string.
# A gateway id is ``vendor/model`` (``deepseek/deepseek-v4.1-flash``), and
# CrewAI's ``LLM.__new__`` reads that leading segment as a provider route: for
# ``deepseek/...`` it strips the prefix before sending and the gateway answers
# 404 ``model_not_found``; for ``anthropic/...`` or ``google/...`` it tries to
# import a native provider that is not a dependency here; for anything else it
# wants the LiteLLM fallback, which is not installed either. An explicit
# provider skips that resolution entirely (``model_string = model`` in
# ``LLM._resolve_route``), so the id goes on the wire exactly as the gateway
# lists it in ``/models``.
PROVIDER = "openai"


def build_llm(cfg: LLMConfig) -> LLM:
    """Build a CrewAI ``LLM`` for an OpenAI-compatible endpoint.

    ``context_window_size`` is forwarded when known: CrewAI otherwise guesses a
    window from its model catalog (4096 tokens for an unknown gateway alias),
    which misleads both its own handling and ours.
    """
    return LLM(
        model=cfg.model,
        provider=PROVIDER,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        context_window_size=cfg.context_window_size,
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
    max_iter: int | None = None,
) -> Agent:
    extra = {} if max_iter is None else {"max_iter": max_iter}
    return Agent(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        tools=tools,
        llm=llm,
        verbose=verbose,
        # Proactive compaction (``compaction.Compactor``) keeps the history under
        # the window. The built-in path only fires on a provider error and then
        # replaces *all* non-system messages with one summary -- which drops the
        # task prompt and the output schema. It stays enabled purely as a
        # last-resort net for a mis-estimated window; it never runs otherwise.
        respect_context_window=True,
        **extra,
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
    # The lead gets only the lookup pair so it cannot spend its rounds re-reading
    # what the attached reports already contain, and a hard cap on iterations so
    # a failed round cannot cascade into a conversion plus a task retry.
    lead_agent = _make_agent(
        REVIEW_LEAD,
        llm,
        make_git_tools(repo, scope, only=LEAD_TOOLS),
        verbose=verbose,
        max_iter=LEAD_MAX_ITER,
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
