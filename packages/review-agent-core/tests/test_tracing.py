"""Tests for the tracing usage accounting (no LLM calls)."""

from __future__ import annotations

from crewai.types.usage_metrics import UsageMetrics

from review_agent_core.tracing import (
    _AGENT_PALETTE,
    _agent_color,
    _head,
    _Tracer,
    record_usage,
    usage_summary,
)


def test_head_collapses_and_truncates() -> None:
    assert _head("a\n b\n  c", 100) == "a b c"
    assert _head("x" * 50, 10) == "x" * 10 + "…"


def test_agent_color_is_stable_and_in_palette() -> None:
    assert _agent_color("Code Reviewer") == _agent_color("Code Reviewer")
    for name in (
        "Code Reviewer",
        "Test Quality Reviewer",
        "Review Guide",
        "Review Lead",
    ):
        assert _agent_color(name) in _AGENT_PALETTE


def test_color_for_avoids_collisions() -> None:
    tracer = _Tracer()
    names = [
        "Code Reviewer",
        "Test Quality Reviewer",
        "Review Guide",
        "Review Lead",
    ]
    colors = [tracer._color_for(name) for name in names]
    assert len(set(colors)) == len(names), colors
    # Stable across calls.
    assert [tracer._color_for(name) for name in names] == colors



def test_record_usage_and_summary() -> None:
    record_usage(
        UsageMetrics(
            total_tokens=100,
            prompt_tokens=70,
            completion_tokens=30,
            successful_requests=2,
        )
    )
    summary = usage_summary()
    assert "100 tokens" in summary
    assert "70 prompt" in summary
    assert "30 completion" in summary
    assert "2 requests" in summary


def test_emit_survives_legacy_encoding() -> None:
    """A cp1251 stream must not silently swallow trace lines (no Σ support)."""
    import io

    from rich.console import Console

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1251")
    tracer = _Tracer()
    tracer._stream = stream
    tracer._console = Console(file=stream)

    tracer._emit("[llm] 1(tok) | Σ 5(tok) running", agent="Code Reviewer")
    stream.flush()

    written = raw.getvalue().decode("cp1251")
    assert "[Code Reviewer]" in written
    assert "5(tok) running" in written


def test_context_budget_formats_percent_and_markers() -> None:
    from review_agent_core.tracing import ContextBudget

    budget = ContextBudget.from_window(6_000, 0.75, 0.5)
    assert (budget.trigger, budget.target) == (4_500, 3_000)
    assert budget.percent(1_200) == "20%"
    assert budget.marker(1_200) == ""
    assert budget.marker(4_500).strip() == "⚠trigger"
    assert budget.marker(5_100).strip() == "⚠crewai"
    assert budget.marker(6_000).strip() == "⚠window"
    assert budget.marker(7_128).strip() == "⚠window"

    # A 1M window must not turn into "1000%"-style noise on sub-percent values.
    big = ContextBudget.from_window(1_000_000, 0.75, 0.5)
    assert big.percent(4_982) == "0.5%"
    assert big.marker(4_982) == ""


def test_budget_from_ratios() -> None:
    from review_agent_core.tracing import ContextBudget

    assert ContextBudget.from_window(6_000, 0.6, 0.4) == ContextBudget(6_000, 3_600, 2_400)
    # A degenerate window must not raise from a trace-formatting helper.
    assert ContextBudget(0, 0, 0).percent(10) == "?"
