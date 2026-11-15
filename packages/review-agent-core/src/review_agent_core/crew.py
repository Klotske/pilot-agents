"""Build single-agent review Crews (one per review mode)."""

from __future__ import annotations

from importlib.resources import files

from crewai import Agent, Crew, LLM, Task

from .config import LLMConfig
from .git_repo import BranchDiffScope, GitRepo
from .tools.git_tools import make_git_tools


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


def _build_review_crew(
    llm: LLM,
    repo: GitRepo,
    scope: BranchDiffScope,
    *,
    skill: str,
    role: str,
    goal: str,
    backstory: str,
    verbose: bool = False,
) -> Crew:
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=make_git_tools(repo, scope),
        llm=llm,
        verbose=verbose,
    )

    task = Task(
        description=load_skill(skill),
        expected_output="A concise Markdown report following the sections in the instructions.",
        agent=agent,
    )

    return Crew(agents=[agent], tasks=[task], verbose=verbose)


def build_code_review_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Code Reviewer: potential bugs, risks, and complex areas."""
    return _build_review_crew(
        llm, repo, scope,
        verbose=verbose,
        skill="code_reviewer",
        role="Code Reviewer",
        goal="Find potential bugs, risks, and complex areas in the change.",
        backstory=(
            "You are an experienced code reviewer who reads diffs carefully "
            "and flags real, concrete problems."
        ),
    )


def build_test_quality_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Test Quality Reviewer: untested scenarios and weak tests."""
    return _build_review_crew(
        llm, repo, scope,
        verbose=verbose,
        skill="test_quality",
        role="Test Quality Reviewer",
        goal="Analyze test coverage and test quality for the change.",
        backstory=(
            "You are a testing expert who spots untested scenarios and weak tests "
            "that would let bugs through."
        ),
    )


def build_review_guide_crew(
    llm: LLM, repo: GitRepo, scope: BranchDiffScope, verbose: bool = False
) -> Crew:
    """Review Guide: business logic, invariants, and comments for reviewers."""
    return _build_review_crew(
        llm, repo, scope,
        verbose=verbose,
        skill="review_guide",
        role="Review Guide",
        goal="Explain the change and its invariants, and prepare useful comments for reviewers.",
        backstory=(
            "You are a reviewer's assistant who distills business logic into "
            "clear, actionable guidance."
        ),
    )


def build_review_lead_crew(llm: LLM, verbose: bool = False) -> Crew:
    """Review Lead: merges the specialized reports into one consolidated report."""
    agent = Agent(
        role="Review Lead",
        goal="Merge the specialized review reports into one coherent, deduplicated report.",
        backstory=(
            "You are a senior reviewer who consolidates findings from "
            "specialized reviewers into a single actionable report."
        ),
        tools=[],
        llm=llm,
        verbose=verbose,
    )

    task = Task(
        description=load_skill("review_lead"),
        expected_output="A single consolidated Markdown report.",
        agent=agent,
    )

    return Crew(agents=[agent], tasks=[task], verbose=verbose)
