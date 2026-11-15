"""End-to-end trace test: real CrewAI events through the event bus."""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone

from crewai import Agent
from crewai.events import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

from review_agent_core import LLMConfig, build_llm
from review_agent_core.tracing import _Tracer


def _emit(event) -> None:
    future = crewai_event_bus.emit(None, event)
    if future is not None:
        future.result(timeout=5)


def test_trace_lines_for_agent_llm_tool_and_error() -> None:
    buffer = io.StringIO()
    tracer = _Tracer()
    tracer.enable(stream=buffer)

    agent = Agent(
        role="Code Reviewer",
        goal="g",
        backstory="b",
        tools=[],
        llm=build_llm(LLMConfig(model="openai/gpt-4o-mini")),
    )
    now = datetime.now(timezone.utc)

    _emit(AgentExecutionStartedEvent(agent=agent, task=None, tools=[], task_prompt="x"))
    _emit(
        LLMCallStartedEvent(
            call_id="test-call-1",
            model="deepseek-flash",
            from_agent=agent,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function"}, {"type": "function"}],
            temperature=0.2,
        )
    )
    _emit(ToolUsageStartedEvent(tool_name="get_diff", tool_args={"x": 1}, from_agent=agent))
    _emit(
        ToolUsageFinishedEvent(
            tool_name="get_diff",
            tool_args={"x": 1},
            output="diff --git a/order.py",
            started_at=now,
            finished_at=now + timedelta(milliseconds=34),
            from_agent=agent,
        )
    )
    _emit(
        LLMCallCompletedEvent(
            call_id="test-call-1",
            model="deepseek-flash",
            response="ok",
            call_type=LLMCallType.LLM_CALL,
            usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
            finish_reason="stop",
            from_agent=agent,
            timestamp=now + timedelta(seconds=2),
        )
    )
    _emit(
        ToolUsageErrorEvent(
            tool_name="read_file", tool_args={}, error="file not found", from_agent=agent
        )
    )
    _emit(
        AgentExecutionCompletedEvent(
            agent=agent, task=None, output="done", timestamp=now + timedelta(seconds=9)
        )
    )

    out = buffer.getvalue()
    lines = out.splitlines()
    joined = "\n".join(lines)

    # phase banner (7) with the real agent name, not "unknown agent"
    assert any(
        line.startswith("═══ ") and line.endswith(" ═══") and "Code Reviewer" in line
        for line in lines
    ), lines
    assert not any("unknown agent" in line for line in lines), lines

    # LLM request details (5)
    assert "request: msgs=1 chars=2, tools=2, temp=0.2" in joined

    # tool call with duration (1)
    assert "[tool] get_diff({'x': 1})" in joined
    assert re.search(r"-> diff --git a/order\.py \(\d+ms\)", joined), joined

    # LLM completion: tokens, finish reason, call type, duration (5)
    assert "1200(tok) prompt + 300(tok) completion = 1500(tok)" in joined
    assert "finish=stop" in joined
    assert "call=llm_call" in joined
    assert re.search(r"call=llm_call \d+\.\d+s", joined), joined

    # tool error (2)
    assert "[error] read_file failed: file not found" in joined

    # agent completion with duration (3)
    assert re.search(r"\[agent\] done in \d+\.\d+s", joined), joined
