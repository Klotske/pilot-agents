"""Tests for the Crew assembly (structure only, no LLM calls)."""

from __future__ import annotations

from pathlib import Path

from crewai import Crew

from review_agent_core import (
    LLMConfig,
    build_code_review_crew,
    build_llm,
    build_review_guide_crew,
    build_review_lead_crew,
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
        ("code_reviewer", "Bugs and risks"),
        ("test_quality", "Missing scenarios"),
        ("review_guide", "Key invariants"),
    ]:
        text = load_skill(name)
        assert "{base_ref}" in text
        assert "{head_ref}" in text
        assert marker in text


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


def test_build_review_lead_crew() -> None:
    crew = build_review_lead_crew(build_llm(LLMConfig(model="openai/gpt-4o-mini")))
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    lead = crew.agents[0]
    assert not lead.tools
    assert "{code_report}" in crew.tasks[0].description
