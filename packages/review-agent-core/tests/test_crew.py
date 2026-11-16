"""Tests for the Crew assembly (structure only, no LLM calls)."""

from __future__ import annotations

from pathlib import Path

from crewai import Crew, TaskOutput

from review_agent_core import (
    LLMConfig,
    build_code_review_crew,
    build_llm,
    build_review_guide_crew,
    build_review_pipeline_crew,
    build_test_quality_crew,
)
from review_agent_core.crew import load_skill
from review_agent_core.git_repo import BranchDiffScope, GitRepo

from .helpers import commit_all, init_repo, write


def _repo(tmp_path: Path) -> GitRepo:
    path = init_repo(tmp_path / "repo")
    write(path, "a.py", "x = 1\n")
    commit_all(path, "init")
    return GitRepo(path)


def _scope() -> BranchDiffScope:
    return BranchDiffScope(base="main", head="feature")


def test_load_skills() -> None:
    for name, marker in [
        ("code_reviewer", "severity"),
        ("test_quality", "test_gap"),
        ("review_guide", "Key invariants"),
        ("review_lead", "deduplicate"),
    ]:
        text = load_skill(name)
        assert "{base_ref}" in text
        assert "{head_ref}" in text
        assert marker.lower() in text.lower()


def test_build_llm() -> None:
    llm = build_llm(LLMConfig(model="openai/gpt-4o-mini", base_url="http://localhost:8000/v1"))
    assert llm.model == "gpt-4o-mini"


def _assert_single_agent_crew(crew: Crew) -> None:
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    agent = crew.agents[0]
    task = crew.tasks[0]
    assert agent.tools is not None
    assert len(agent.tools) == 5
    assert task.agent is agent
    assert "{base_ref}" in task.description


def test_build_review_crews(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    scope = _scope()
    llm = build_llm(LLMConfig(model="openai/gpt-4o-mini"))

    for build in (build_code_review_crew, build_test_quality_crew, build_review_guide_crew):
        _assert_single_agent_crew(build(llm, repo, scope))


def test_build_review_pipeline_crew(tmp_path: Path) -> None:
    crew = build_review_pipeline_crew(
        build_llm(LLMConfig(model="openai/gpt-4o-mini")), _repo(tmp_path), _scope()
    )

    assert [agent.role for agent in crew.agents] == [
        "Review Guide",
        "Code Reviewer",
        "Test Quality Reviewer",
        "Review Lead",
    ]
    assert len(crew.tasks) == 4

    guide, code, tests, lead = crew.tasks

    # The guide is the only synchronous entry point.
    assert not guide.async_execution

    # Code and tests run in parallel on the guide's output.
    assert code.async_execution
    assert tests.async_execution
    assert code.context == [guide]
    assert tests.context == [guide]

    # The lead waits for everything.
    assert not lead.async_execution
    assert lead.context == [guide, code, tests]

    # Every agent, including the lead, has the git tools. The lead's tools keep
    # its structured output on CrewAI's tool-present path, which avoids the
    # provider-level response_format request some gateways reject.
    for agent in crew.agents:
        assert agent.tools is not None
        assert len(agent.tools) == 5

    # Role of each agent matches the task it owns.
    assert [task.agent.role for task in crew.tasks] == [
        agent.role for agent in crew.agents
    ]


def test_review_lead_skill_uses_context_not_placeholders() -> None:
    text = load_skill("review_lead")
    assert "{code_report}" not in text
    assert "{tests_report}" not in text
    assert "{guide_report}" not in text
    assert "context" in text


def test_pipeline_passes_guide_output_downstream(tmp_path: Path) -> None:
    """The guide's report must reach the parallel reviewers and the lead.

    Verified without an LLM: crew context is aggregated from ``task.output``,
    so fabricating outputs shows exactly what each downstream task receives.
    """
    crew = build_review_pipeline_crew(
        build_llm(LLMConfig(model="openai/gpt-4o-mini")), _repo(tmp_path), _scope()
    )
    guide, code, tests, lead = crew.tasks

    guide.output = TaskOutput(
        description="d",
        agent="Review Guide",
        raw="INVARIANT: discount must be within [0, 100].",
        output_format="raw",
    )
    code.output = TaskOutput(
        description="d", agent="Code Reviewer", raw="BUG: no validation.", output_format="raw"
    )
    tests.output = TaskOutput(
        description="d", agent="Test Quality", raw="GAP: no empty cart test.", output_format="raw"
    )

    code_context = Crew._get_context(code, [guide.output])
    tests_context = Crew._get_context(tests, [guide.output])
    lead_context = Crew._get_context(lead, [guide.output, code.output, tests.output])

    assert "INVARIANT: discount must be within [0, 100]." in code_context
    assert "INVARIANT: discount must be within [0, 100]." in tests_context
    for marker in ("INVARIANT", "BUG: no validation.", "GAP: no empty cart test."):
        assert marker in lead_context


def test_tasks_request_structured_output(tmp_path: Path) -> None:
    from review_agent_core.schemas import MergedReport, ReviewerFindings

    llm = build_llm(LLMConfig(model="openai/gpt-4o-mini"))
    repo, scope = _repo(tmp_path), _scope()

    # Single-mode reviewers return findings.
    for build in (build_code_review_crew, build_test_quality_crew):
        assert build(llm, repo, scope).tasks[0].output_pydantic is ReviewerFindings

    # The guide stays prose.
    assert build_review_guide_crew(llm, repo, scope).tasks[0].output_pydantic is None

    # In the pipeline only the lead returns a merged report.
    guide, code, tests, lead = build_review_pipeline_crew(llm, repo, scope).tasks
    assert guide.output_pydantic is None
    assert code.output_pydantic is ReviewerFindings
    assert tests.output_pydantic is ReviewerFindings
    assert lead.output_pydantic is MergedReport


def test_skills_describe_meaning_not_json_format() -> None:
    """The schema comes from pydantic via output_pydantic, not from the prompts."""
    for name in ("code_reviewer", "test_quality", "review_lead"):
        text = load_skill(name)
        assert "```json" not in text
        assert "Return **one JSON object" not in text

    # Severity semantics stay in the prompt: they describe content, not format.
    assert "critical:" in load_skill("code_reviewer")
    assert "test_gap" in load_skill("test_quality")
    assert "deduplicate" in load_skill("review_lead").lower()


def test_schema_is_injected_into_the_prompt(tmp_path: Path) -> None:
    """CrewAI appends the pydantic schema to the prompt for output_pydantic tasks.

    This is what makes the hand-written JSON contract in the skills unnecessary
    and why the pydantic models stay the single source of truth.
    """
    from crewai.agent.utils import build_task_prompt_with_schema

    llm = build_llm(LLMConfig(model="openai/gpt-4o-mini"))
    crew = build_review_pipeline_crew(llm, _repo(tmp_path), _scope())
    guide, code, tests, lead = crew.tasks

    for task in (code, tests, lead):
        prompt = build_task_prompt_with_schema(task, task.prompt())
        assert "Format your final answer" in prompt
        for field in ('"path"', '"rule"', '"severity"', '"category"'):
            assert field in prompt

    # The guide returns prose, so no schema is appended.
    guide_prompt = build_task_prompt_with_schema(guide, guide.prompt())
    assert "Format your final answer" not in guide_prompt
