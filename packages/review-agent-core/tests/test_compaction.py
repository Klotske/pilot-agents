"""Tests for proactive history compaction (no LLM calls, no network)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from crewai.hooks import LLMCallHookContext
from pydantic import ValidationError

from review_agent_core import CompactionConfig
from review_agent_core.api import _compactor
from review_agent_core.compaction import CharCounter, Compactor, _Counter, compact_messages
from review_agent_core.config import LLMConfig

Message = dict[str, Any]


class WordCounter:
    """Deterministic stand-in for tiktoken: one token per whitespace word."""

    def count(self, text: str) -> int:
        return len(text.split())


SYSTEM: Message = {"role": "system", "content": "You are a careful reviewer. " * 5}
TASK: Message = {"role": "user", "content": "Review {base_ref}..{head_ref} and report. " * 5}


def _counter() -> _Counter:
    return _Counter(WordCounter())


def _char_counter() -> _Counter:
    """Token estimate proportional to length, for char-budget assertions."""
    return _Counter(CharCounter(chars_per_token=4))


def _assistant(*call_ids: str) -> Message:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "get_diff", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def _tool(content: str, name: str = "get_diff", call_id: str = "c1") -> Message:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _history(steps: int, words: int) -> list[Message]:
    messages: list[Message] = [copy.deepcopy(SYSTEM), copy.deepcopy(TASK)]
    for step in range(steps):
        messages.append(_assistant(f"c{step}"))
        messages.append(_tool(_words(f"line{step}_", words), call_id=f"c{step}"))
    return messages


def test_no_compaction_below_trigger() -> None:
    messages = _history(2, 10)
    original = copy.deepcopy(messages)

    result = compact_messages(
        messages,
        cfg=CompactionConfig(),
        window_tokens=1_000_000,
        counter=_counter(),
    )

    assert messages == original
    assert not result.changed
    assert result.before_tokens == result.after_tokens


def test_keeps_system_task_and_trailing_steps() -> None:
    messages = _history(6, 500)
    tail_before = copy.deepcopy(messages[-4:])

    result = compact_messages(
        messages,
        cfg=CompactionConfig(),
        window_tokens=3_000,
        counter=_counter(),
    )

    assert result.changed
    assert result.after_tokens < result.before_tokens
    assert messages[0] == SYSTEM
    assert messages[1] == TASK
    # Two trailing steps stay verbatim: the answer still needs them.
    assert messages[-4:] == tail_before


def test_middle_tool_results_are_truncated_with_a_marker() -> None:
    messages = _history(6, 500)
    before_chars = len(messages[3]["content"])

    compact_messages(
        messages,
        cfg=CompactionConfig(),
        window_tokens=3_000,
        counter=_counter(),
    )

    content = messages[3]["content"]
    assert len(content) < before_chars
    assert "⟪truncated " in content


def test_compaction_is_idempotent() -> None:
    messages = _history(6, 500)
    cfg = CompactionConfig()
    counter = _counter()

    compact_messages(messages, cfg=cfg, window_tokens=3_000, counter=counter)
    after_first = copy.deepcopy(messages)

    second = compact_messages(messages, cfg=cfg, window_tokens=3_000, counter=counter)

    assert messages == after_first
    assert second.truncated == 0
    assert second.deduped == 0


def test_assistant_tool_pairing_is_preserved() -> None:
    messages = _history(6, 500)
    call_ids_before = [
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    count_before = len(messages)

    compact_messages(
        messages,
        cfg=CompactionConfig(),
        window_tokens=3_000,
        counter=_counter(),
    )

    assert len(messages) == count_before
    assert [
        message["tool_call_id"] for message in messages if message.get("role") == "tool"
    ] == call_ids_before


def test_duplicate_tool_results_are_replaced_by_a_pointer() -> None:
    payload = _words("same_", 200)
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(payload, call_id="c1"),
        _assistant("c2"),
        _tool(payload, call_id="c2"),
    ]
    cfg = CompactionConfig(keep_last_steps=0)

    result = compact_messages(messages, cfg=cfg, window_tokens=600, counter=_counter())

    assert result.deduped == 1
    assert messages[3]["content"] == payload  # the first occurrence stays
    assert messages[5]["content"] == "⟪same result as the earlier get_diff call⟫"
    assert messages[5]["tool_call_id"] == "c2"


def test_dedupe_replaces_the_newest_duplicate() -> None:
    payload = _words("same_", 200)
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(payload, call_id="c1"),
        _assistant("c2"),
        _tool(payload, call_id="c2"),
    ]
    # The duplicate is the freshest step. Replacing it is safe -- the identical
    # text is one message earlier in this very request -- and protecting it
    # would make dedupe a no-op for the case it exists for.
    cfg = CompactionConfig(keep_last_steps=1)

    result = compact_messages(messages, cfg=cfg, window_tokens=600, counter=_counter())

    assert result.deduped == 1
    assert messages[3]["content"] == payload  # the first occurrence stays canonical
    assert messages[5]["content"] == "⟪same result as the earlier get_diff call⟫"
    assert messages[5]["tool_call_id"] == "c2"


def test_structure_aware_keeps_the_diff_map() -> None:
    diff = (
        "x" * 3_000
        + "\ndiff --git a/a.py b/a.py\n@@ -1,2 +1,2 @@\n"
        + "y" * 3_000
    )
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(diff, call_id="c1"),
    ]

    compact_messages(
        messages,
        cfg=CompactionConfig(keep_last_steps=0),
        window_tokens=1_000,
        counter=_char_counter(),
    )

    content = messages[3]["content"]
    assert len(content) < len(diff)
    assert "diff --git a/a.py b/a.py" in content
    assert "@@ -1,2 +1,2 @@" in content
    assert "⟪kept the change map / declarations⟫" in content


def test_structure_lines_can_be_dropped() -> None:
    diff = (
        "x" * 3_000
        + "\ndiff --git a/a.py b/a.py\n@@ -1,2 +1,2 @@\n"
        + "y" * 3_000
    )
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(diff, call_id="c1"),
    ]

    compact_messages(
        messages,
        cfg=CompactionConfig(keep_last_steps=0, structure_aware=False),
        window_tokens=1_000,
        counter=_char_counter(),
    )

    content = messages[3]["content"]
    assert "@@ -1,2 +1,2 @@" not in content


def test_protected_part_larger_than_target_leaves_history_alone() -> None:
    messages = _history(3, 500)
    original = copy.deepcopy(messages)

    # The tail plus prompt already exceed the target, so nothing can be cut safely.
    result = compact_messages(
        messages,
        cfg=CompactionConfig(keep_last_steps=3),
        window_tokens=300,
        counter=_counter(),
    )

    assert messages == original
    assert result.truncated == 0


def test_hook_skips_direct_llm_calls() -> None:
    messages = _history(6, 500)
    original = copy.deepcopy(messages)
    compactor = Compactor(
        CompactionConfig(), model="m", window_tokens=3_000, tokenizer=WordCounter()
    )

    compactor.hook(LLMCallHookContext(messages=messages, agent=None))

    assert messages == original


def test_hook_compacts_agent_history_in_place() -> None:
    messages = _history(6, 500)
    compactor = Compactor(
        CompactionConfig(), model="m", window_tokens=3_000, tokenizer=WordCounter()
    )

    compactor.hook(
        LLMCallHookContext(
            messages=messages, agent=SimpleNamespace(role="Code Reviewer")
        )
    )

    assert any("⟪truncated " in str(message.get("content") or "") for message in messages)


def test_disabled_compactor_does_nothing() -> None:
    messages = _history(6, 500)
    original = copy.deepcopy(messages)
    compactor = Compactor(
        CompactionConfig(enabled=False),
        model="m",
        window_tokens=3_000,
        tokenizer=WordCounter(),
    )

    compactor.hook(
        LLMCallHookContext(
            messages=messages, agent=SimpleNamespace(role="Code Reviewer")
        )
    )

    assert messages == original


def test_compactor_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError):
        Compactor(CompactionConfig(), model="m", window_tokens=0)


def test_api_requires_the_window_for_compaction() -> None:
    with pytest.raises(ValueError, match="context window"):
        _compactor(LLMConfig(model="m"), None)


def test_api_returns_none_when_compaction_is_disabled() -> None:
    assert _compactor(LLMConfig(model="m"), CompactionConfig(enabled=False)) is None


def test_ratios_must_be_consistent() -> None:
    with pytest.raises(ValidationError):
        CompactionConfig(trigger_ratio=0.5, target_ratio=0.6)


def test_char_counter_estimates_tokens() -> None:
    assert CharCounter(chars_per_token=4).count("a" * 400) == 101


def test_scoped_hook_is_dispatched_by_the_crewai_executor() -> None:
    """Prove the wiring, not just the algorithm: crewai resolves scoped hooks."""
    from crewai.hooks import InterceptionPoint
    from crewai.hooks.dispatch import register_scoped, scoped_hooks
    from crewai.utilities.agent_utils import _setup_before_llm_call_hooks

    messages = _history(6, 500)
    executor = SimpleNamespace(
        messages=messages,
        llm=None,
        iterations=0,
        agent=SimpleNamespace(role="Code Reviewer"),
        task=None,
        crew=None,
        before_llm_call_hooks=[],
    )
    compactor = Compactor(
        CompactionConfig(), model="m", window_tokens=3_000, tokenizer=WordCounter()
    )

    with scoped_hooks():
        register_scoped(InterceptionPoint.PRE_MODEL_CALL, compactor.hook)
        assert _setup_before_llm_call_hooks(executor, None, verbose=False) is True  # type: ignore[arg-type]

    assert any("⟪truncated " in str(message.get("content") or "") for message in messages)


def test_scoped_hook_is_gone_after_the_run() -> None:
    from crewai.hooks import InterceptionPoint
    from crewai.hooks.dispatch import get_scoped_hooks

    from review_agent_core.api import _kickoff

    class _StubCrew:
        usage_metrics = None

        def kickoff(self, inputs: dict[str, str]) -> int:
            return len(get_scoped_hooks(InterceptionPoint.PRE_MODEL_CALL))

    compactor = Compactor(
        CompactionConfig(), model="m", window_tokens=3_000, tokenizer=WordCounter()
    )
    repo = SimpleNamespace(path=".")  # type: ignore[arg-type]
    scope = SimpleNamespace(base="main", head="HEAD")  # type: ignore[arg-type]

    assert _kickoff(_StubCrew(), repo, scope, compactor) == 1  # type: ignore[arg-type]
    assert get_scoped_hooks(InterceptionPoint.PRE_MODEL_CALL) == []


def test_tokenizer_falls_back_when_tiktoken_is_unavailable(monkeypatch) -> None:
    import review_agent_core.compaction as compaction

    def boom(model: str, encoding: str | None) -> object:
        raise RuntimeError("no network")

    monkeypatch.setattr(compaction, "TiktokenCounter", boom)

    tokenizer = compaction.build_tokenizer("gateway/model", CompactionConfig())

    assert isinstance(tokenizer, CharCounter)


def test_short_messages_are_never_expanded() -> None:
    small = _words("tiny_", 30)  # well under min_chars_per_message
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(small, call_id="c1"),
        _assistant("c2"),
        _tool(_words("big_", 2_000), call_id="c2"),
    ]
    messages[3]["content"] = small[:350]

    compact_messages(
        messages,
        cfg=CompactionConfig(keep_last_steps=0),
        window_tokens=800,
        counter=_char_counter(),
    )

    assert messages[3]["content"] == small[:350]


def test_already_shortened_history_can_shrink_further() -> None:
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(_words("line_", 3_000), call_id="c1"),
    ]
    cfg = CompactionConfig(keep_last_steps=0)

    compact_messages(messages, cfg=cfg, window_tokens=4_000, counter=_char_counter())
    first_pass = len(messages[3]["content"])

    result = compact_messages(
        messages, cfg=cfg, window_tokens=500, counter=_char_counter()
    )

    assert result.truncated == 1
    assert len(messages[3]["content"]) < first_pass


def test_an_oversized_older_step_loses_its_protection() -> None:
    newest = _words("fresh_", 25)
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(_words("old_", 2_000), call_id="c1"),
        _assistant("c2"),
        _tool(newest, call_id="c2"),
    ]

    result = compact_messages(
        messages,
        cfg=CompactionConfig(keep_last_steps=2),
        window_tokens=1_000,
        counter=_counter(),
    )

    # Protecting both trailing steps would leave nothing to cut, so the older
    # one is compacted instead of silently doing nothing...
    assert result.truncated == 1
    assert "⟪truncated " in messages[3]["content"]
    # ...while the newest step stays intact: the agent has just fetched it.
    assert messages[5]["content"] == newest


def test_describe_states_the_active_budget() -> None:
    compactor = Compactor(
        CompactionConfig(trigger_ratio=0.75, target_ratio=0.5),
        model="m",
        window_tokens=30_000,
        tokenizer=WordCounter(),
    )

    line = compactor.describe()

    assert "window=30000" in line
    assert "trigger=22500" in line
    assert "target=15000" in line


def test_dedupe_runs_without_a_trigger() -> None:
    payload = _words("same_", 200)
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(payload, call_id="c1"),
        _assistant("c2"),
        _tool(payload, call_id="c2"),
    ]

    # The window is huge, so shortening is not warranted -- dedupe still is.
    result = compact_messages(
        messages,
        cfg=CompactionConfig(keep_last_steps=0),
        window_tokens=1_000_000,
        counter=_counter(),
    )

    assert result.deduped == 1
    assert result.truncated == 0
    assert messages[5]["content"] == "⟪same result as the earlier get_diff call⟫"


def test_near_miss_is_explained(monkeypatch) -> None:
    import review_agent_core.compaction as compaction
    from types import SimpleNamespace

    notes: list[str] = []
    monkeypatch.setattr(
        compaction, "trace_note", lambda message, agent=None: notes.append(message)
    )

    messages: list[Message] = [copy.deepcopy(SYSTEM), copy.deepcopy(TASK)]
    compactor = compaction.Compactor(
        CompactionConfig(), model="m", window_tokens=50, tokenizer=WordCounter()
    )

    compactor.hook(
        LLMCallHookContext(messages=messages, agent=SimpleNamespace(role="Review Lead"))
    )

    assert len(notes) == 1
    assert "nothing cut" in notes[0]
    assert "protected" in notes[0]


def test_dedupe_only_run_is_reported(monkeypatch) -> None:
    import review_agent_core.compaction as compaction
    from types import SimpleNamespace

    notes: list[str] = []
    monkeypatch.setattr(
        compaction, "trace_note", lambda message, agent=None: notes.append(message)
    )

    payload = _words("same_", 200)
    messages: list[Message] = [
        copy.deepcopy(SYSTEM),
        copy.deepcopy(TASK),
        _assistant("c1"),
        _tool(payload, call_id="c1"),
        _assistant("c2"),
        _tool(payload, call_id="c2"),
    ]
    compactor = compaction.Compactor(
        CompactionConfig(keep_last_steps=0),
        model="m",
        window_tokens=1_000_000,
        tokenizer=WordCounter(),
    )

    compactor.hook(
        LLMCallHookContext(messages=messages, agent=SimpleNamespace(role="Code Reviewer"))
    )

    assert len(notes) == 1
    assert "dedup=1" in notes[0]
    assert "truncated" not in notes[0]


def test_budget_is_built_from_the_window_and_settings() -> None:
    from review_agent_core.api import _budget
    from review_agent_core.tracing import ContextBudget

    assert _budget(LLMConfig(model="m"), None) is None
    assert _budget(LLMConfig(model="m"), CompactionConfig(trigger_ratio=0.8)) is None
    assert _budget(
        LLMConfig(model="m", context_window_size=6_000), None
    ) == ContextBudget(6_000, 4_500, 3_000)
    # The budget stays useful even with compaction switched off.
    assert _budget(
        LLMConfig(model="m", context_window_size=6_000),
        CompactionConfig(enabled=False),
    ) == ContextBudget(6_000, 4_500, 3_000)
