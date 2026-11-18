"""Tests for the Crew assembly (structure only, no LLM calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
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
    # The id reaches the endpoint verbatim: a gateway lists its own models, and
    # stripping a leading vendor segment would turn a valid id into a 404.
    assert llm.model == "openai/gpt-4o-mini"
    assert llm.provider == "openai"


def test_build_llm_forwards_context_window() -> None:
    llm = build_llm(
        LLMConfig(model="openai/gpt-4o-mini", context_window_size=128_000)
    )
    assert llm.context_window_size == 128_000


def _assert_single_agent_crew(crew: Crew) -> None:
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    agent = crew.agents[0]
    task = crew.tasks[0]
    assert agent.tools is not None
    assert len(agent.tools) == 6
    # Proactive compaction handles the window; CrewAI's reactive summarizer is
    # only a last-resort net, so it must not be disabled.
    assert agent.respect_context_window is True
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

    # The three reviewers get the full tool set. The lead gets one lookup tool
    # and a two-iteration cap: the reports it merges already contain the diff,
    # and an extra tool round is a chance to end without an answer (which costs
    # an LLM conversion plus a task retry). Any tool at all keeps its structured
    # output on CrewAI's tool-present path, which avoids the provider-level
    # response_format request some gateways reject.
    for agent in crew.agents[:3]:
        assert agent.tools is not None
        assert len(agent.tools) == 6

    lead_agent = crew.agents[3]
    assert lead_agent.tools is not None
    # One cheap tool, present only so the request keeps tool-calling mode.
    assert {tool.name for tool in lead_agent.tools} == {"list_files"}
    assert lead_agent.max_iter == 2
    for agent in crew.agents[:3]:
        assert agent.max_iter == 25  # CrewAI's default, unchanged

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


def test_review_lead_skill_forbids_tool_use() -> None:
    """A tool round is how the lead used to end without a final answer."""
    flat = " ".join(load_skill("review_lead").split())
    assert "Do NOT call read_file, list_files or any other tool" in flat
    assert 'Begin your answer with "{" as the very first character' in flat


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


_GATEWAY_MODEL_IDS = [
    # Vendor prefix that CrewAI would otherwise strip or reject.
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-5.1-codex",
    "anthropic/claude-opus-4.8",
    "google/gemini-3-flash-preview",
    "selectel/qwen3.8-flash-next",
    # A bare id must stay bare as well.
    "qwen3.8-flash-next",
]


@pytest.mark.parametrize("model", _GATEWAY_MODEL_IDS)
def test_build_llm_sends_the_gateway_model_id_verbatim(model: str) -> None:
    llm = build_llm(
        LLMConfig(
            model=model,
            base_url="https://gateway.example/v1",
            api_key="secret",
        )
    )
    # ``llm.model`` is what goes into the request body, so the vendor prefix has
    # to survive; a stripped id is a 404 from the gateway.
    assert llm.model == model
    assert llm.provider == "openai"


def test_build_llm_routes_a_gateway_model_without_litellm() -> None:
    llm = build_llm(
        LLMConfig(
            model="selectel/qwen3.8-flash-next",
            base_url="https://gateway.example/v1",
            api_key="secret",
            context_window_size=262_144,
        )
    )
    assert llm.provider == "openai"
    assert llm.model == "selectel/qwen3.8-flash-next"
    assert llm.context_window_size == 262_144


def test_git_tool_surface(tmp_path: Path) -> None:
    from review_agent_core.tools.git_tools import make_git_tools

    repo = _repo(tmp_path)
    tools = make_git_tools(repo, _scope())

    assert [tool.name for tool in tools] == [
        "get_diff",
        "get_diff_stat",
        "get_changed_files",
        "get_commits",
        "list_files",
        "read_file",
    ]

    list_files = tools[4]
    listed = str(list_files.run(ref="main"))
    assert "a.py" in listed


def test_reviewer_skills_point_at_list_files() -> None:
    for name in ("code_reviewer", "test_quality", "review_guide", "review_lead"):
        assert "list_files" in load_skill(name)


def test_findings_skills_demand_json_only() -> None:
    """`output_pydantic` validates JSON: prose answers cause retries and an extra
    LLM conversion, so the skills must ask for JSON explicitly."""
    for name in ("code_reviewer", "test_quality", "review_lead"):
        flat = " ".join(load_skill(name).split())
        assert "ONE JSON object and nothing else" in flat
        assert "no code fences" in flat
        assert "critical, high, medium, low" in flat


def test_skill_placeholders_survive_interpolation() -> None:
    from crewai.utilities.string_utils import interpolate_only

    inputs = {"repo_path": "/repo", "base_ref": "main", "head_ref": "origin/feature"}
    for name in ("code_reviewer", "test_quality", "review_lead"):
        rendered = interpolate_only(load_skill(name), inputs)
        # The JSON examples must reach the model intact, not be treated as
        # template fields and stripped or raising.
        assert '"findings"' in rendered or '"summary"' in rendered
        assert "{base_ref}" not in rendered
        assert "main" in rendered


def test_make_git_tools_can_be_narrowed(tmp_path: Path) -> None:
    from review_agent_core.tools.git_tools import make_git_tools

    repo = _repo(tmp_path)
    only = make_git_tools(repo, _scope(), only={"read_file", "list_files"})

    assert [tool.name for tool in only] == ["list_files", "read_file"]
    # Unknown names are ignored rather than raising: the set is a filter.
    assert make_git_tools(repo, _scope(), only={"nope"}) == []
