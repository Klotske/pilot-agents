"""Optional live tracing over the CrewAI event bus.

When enabled, prints a compact, one-line-per-event trace to ``stderr`` so the
Markdown report on ``stdout`` stays clean. Every line is attributed to the agent
that produced it; each agent gets a stable color derived from its name (no color
table to maintain). Colors auto-disable when ``stderr`` is not a terminal.

What is traced:

* phase banner per agent execution (``═══ Code Reviewer ═══``)
* LLM requests: model, message/tool counts, sampling params
* LLM completions: prompt/completion tokens, cumulative total, finish reason,
  call type and duration
* tool calls: arguments, a short head of the output, duration, cache/retry hints
* failures: tool errors, LLM call failures, agent execution errors
"""

from __future__ import annotations

import logging
import sys
import threading
import zlib
from datetime import datetime
from typing import Any, TextIO, cast

from crewai.events import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolFailureDetectedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.types.usage_metrics import UsageMetrics
from rich.console import Console
from rich.text import Text

MAX_ARGS_CHARS = 200
MAX_OUTPUT_HEAD = 120
MAX_ERROR_CHARS = 300

# Distinguishable rich colors; an agent's name is hashed to pick one.
_AGENT_PALETTE = (
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_red",
    "dark_cyan",
    "medium_purple1",
    "dark_sea_green",
    "orange1",
    "spring_green1",
    "deep_pink2",
    "khaki1",
    "steel_blue1",
)

_TAG_COLORS = {
    "[tool]": "cyan",
    "[llm]": "green",
    "[agent]": "blue",
    "[error]": "bold red",
    "[note]": "yellow",
    "[usage]": "bold yellow",
}


def _agent_color(agent: str) -> str:
    """Pick a stable color for ``agent``.

    Uses CRC32 rather than ``hash()`` because the builtin string hash is
    randomized per process (PYTHONHASHSEED), which would change colors between
    runs.
    """
    return _AGENT_PALETTE[zlib.crc32(agent.encode("utf-8")) % len(_AGENT_PALETTE)]


def _next_free_color(start: int, taken: set[str]) -> str:
    """Return the first palette color at or after ``start`` not in ``taken``."""
    for offset in range(len(_AGENT_PALETTE)):
        candidate = _AGENT_PALETTE[(start + offset) % len(_AGENT_PALETTE)]
        if candidate not in taken:
            return candidate
    return _AGENT_PALETTE[start % len(_AGENT_PALETTE)]  # more agents than colors


def _head(text: str, limit: int) -> str:
    """Collapse ``text`` to a single line and return a short head of it."""
    one_line = " ".join(text.split())
    if len(one_line) > limit:
        return one_line[:limit] + "…"
    return one_line


def _agent_name(event: Any) -> str | None:
    """Return the agent behind ``event``.

    LLM and tool events fill ``agent_role`` from ``from_agent``; agent lifecycle
    events instead carry the agent object, so fall back to its ``role``.
    """
    role = getattr(event, "agent_role", None)
    if role:
        return str(role)
    agent = getattr(event, "agent", None)
    fallback = getattr(agent, "role", None)
    return str(fallback) if fallback else None


def _ms(delta_seconds: float) -> str:
    """Format a duration as a compact human-readable string."""
    if delta_seconds < 1:
        return f"{delta_seconds * 1000:.0f}ms"
    if delta_seconds < 60:
        return f"{delta_seconds:.2f}s"
    minutes, seconds = divmod(delta_seconds, 60)
    return f"{int(minutes)}m{seconds:04.1f}s"


def _elapsed(start: datetime | None, end: datetime | None) -> str | None:
    """Return the formatted duration between two event timestamps."""
    if start is None or end is None:
        return None
    try:
        return _ms((end - start).total_seconds())
    except Exception:
        return None


def _messages_summary(messages: Any) -> str:
    """Describe an LLM request's messages by count and total character size."""
    if messages is None:
        return "msgs=0"
    if isinstance(messages, str):
        return f"msgs=1 chars={len(messages)}"
    try:
        count = len(messages)
    except Exception:
        return "msgs=?"
    chars = 0
    for message in messages:
        if isinstance(message, dict):
            item = cast("dict[str, Any]", message)
            chars += len(str(item.get("content", "")))
        else:
            chars += len(str(message))
    return f"msgs={count} chars={chars}"


class _Counters:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.requests = 0


class _Tracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._stream: TextIO = sys.stderr
        self._console = Console(file=self._stream, soft_wrap=True)
        self._registered = False
        self._live = _Counters()                  # running total from events
        self._final = _Counters()                 # authoritative crew usage
        self._per_agent: dict[str, _Counters] = {}
        self._colors: dict[str, str] = {}         # agent -> assigned color
        self._llm_started: dict[str, datetime] = {}     # call_id -> started at
        self._agent_started: dict[str, datetime] = {}   # agent_role -> started at

    # ------------------------------------------------------------------ colors

    def _color_for(self, agent: str) -> str:
        """Return ``agent``'s color, assigning a collision-free one on first use."""
        with self._lock:
            color = self._colors.get(agent)
            if color is None:
                start = _AGENT_PALETTE.index(_agent_color(agent))
                color = _next_free_color(start, set(self._colors.values()))
                self._colors[agent] = color
            return color

    # ------------------------------------------------------------ registration

    def enable(self, stream: TextIO | None = None) -> None:
        if stream is not None:
            self._stream = stream
            self._console = Console(file=stream, soft_wrap=True)
        # Silence CrewAI's "approximate input token count" fallback warning:
        # our trace reports exact usage from the API response instead.
        logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)
        if self._registered:
            return
        self._registered = True

        # ---------------------------------------------------------- phases (7)

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def _on_agent_started(_source: object, event: AgentExecutionStartedEvent) -> None:
            agent = _agent_name(event) or "unknown agent"
            with self._lock:
                self._agent_started[agent] = event.timestamp
            tools = event.tools
            tools_note = f" ({len(tools)} tools)" if tools else " (no tools)"
            self._banner(f"{agent}{tools_note}", agent)

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def _on_agent_completed(
            _source: object, event: AgentExecutionCompletedEvent
        ) -> None:
            agent = _agent_name(event) or "unknown agent"
            self._emit(f"[agent] done{self._agent_duration(agent, event.timestamp)}", agent)

        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def _on_agent_error(_source: object, event: AgentExecutionErrorEvent) -> None:
            self._emit(
                f"[error] agent failed: {_head(event.error, MAX_ERROR_CHARS)}",
                _agent_name(event),
            )

        # ----------------------------------------------------------- LLM (5)

        @crewai_event_bus.on(LLMCallStartedEvent)
        def _on_llm_started(_source: object, event: LLMCallStartedEvent) -> None:
            if event.call_id:
                with self._lock:
                    self._llm_started[event.call_id] = event.timestamp
            details: list[str] = [_messages_summary(event.messages)]
            if event.tools:
                details.append(f"tools={len(event.tools)}")
            for label, value in (
                ("temp", event.temperature),
                ("max_tokens", event.max_tokens),
            ):
                if value is not None:
                    details.append(f"{label}={value}")
            model = event.model or "llm"
            self._emit(f"[llm] {model} request: {', '.join(details)}", event.agent_role)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_completed(_source: object, event: LLMCallCompletedEvent) -> None:
            usage = event.usage or {}
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            total = int(usage.get("total_tokens", prompt + completion))
            agent = event.agent_role or "unknown agent"
            with self._lock:
                self._live.prompt_tokens += prompt
                self._live.completion_tokens += completion
                self._live.total_tokens += total
                self._live.requests += 1
                running = self._live.total_tokens
                per_agent = self._per_agent.setdefault(agent, _Counters())
                per_agent.prompt_tokens += prompt
                per_agent.completion_tokens += completion
                per_agent.total_tokens += total
                per_agent.requests += 1
                started = self._llm_started.pop(event.call_id, None)
            model = event.model or "llm"
            extras: list[str] = []
            if event.finish_reason:
                extras.append(f"finish={event.finish_reason}")
            call_type = getattr(event, "call_type", None)
            if call_type is not None:
                extras.append(f"call={getattr(call_type, 'value', call_type)}")
            duration = _elapsed(started, event.timestamp)
            if duration:
                extras.append(duration)
            suffix = f" | {' '.join(extras)}" if extras else ""
            self._emit(
                f"[llm] {model} {prompt}(tok) prompt + {completion}(tok) completion"
                f" = {total}(tok) this step | Σ {running}(tok) running{suffix}",
                event.agent_role,
            )

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_llm_failed(_source: object, event: LLMCallFailedEvent) -> None:
            with self._lock:
                self._llm_started.pop(event.call_id, None)
            model = event.model or "llm"
            self._emit(
                f"[error] {model} call failed: {_head(event.error, MAX_ERROR_CHARS)}",
                event.agent_role,
            )

        # ---------------------------------------------------------- tools (1)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_tool_started(_source: object, event: ToolUsageStartedEvent) -> None:
            args = str(event.tool_args)
            if len(args) > MAX_ARGS_CHARS:
                args = args[:MAX_ARGS_CHARS] + "…"
            self._emit(f"[tool] {event.tool_name}({args})", agent=event.agent_role)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_finished(_source: object, event: ToolUsageFinishedEvent) -> None:
            out = _head(str(event.output), MAX_OUTPUT_HEAD)
            notes: list[str] = []
            if event.from_cache:
                notes.append("cached")
            if event.run_attempts and event.run_attempts > 1:
                notes.append(f"attempts={event.run_attempts}")
            duration = _elapsed(event.started_at, event.finished_at)
            if duration:
                notes.append(duration)
            suffix = f" ({', '.join(notes)})" if notes else ""
            self._emit(f"[tool] {event.tool_name} -> {out}{suffix}", agent=event.agent_role)

        # --------------------------------------------------------- errors (2)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(_source: object, event: ToolUsageErrorEvent) -> None:
            self._emit(
                f"[error] {event.tool_name} failed: {_head(event.error, MAX_ERROR_CHARS)}",
                event.agent_role,
            )

        @crewai_event_bus.on(ToolFailureDetectedEvent)
        def _on_tool_failure(_source: object, event: ToolFailureDetectedEvent) -> None:
            failure = getattr(event, "failure", None)
            reason = getattr(failure, "error", None) or failure
            self._emit(
                f"[error] {event.tool_name} did not succeed: "
                f"{_head(str(reason), MAX_ERROR_CHARS)}",
                event.agent_role,
            )

    # ------------------------------------------------------------------ timing

    def _agent_duration(self, agent: str, finished_at: datetime) -> str:
        with self._lock:
            started = self._agent_started.pop(agent, None)
        duration = _elapsed(started, finished_at)
        return f" in {duration}" if duration else ""

    # ------------------------------------------------------------------- usage

    def record_usage(self, metrics: UsageMetrics | None) -> None:
        """Accumulate authoritative usage from a finished crew."""
        if metrics is None:
            return
        with self._lock:
            self._final.prompt_tokens += metrics.prompt_tokens
            self._final.completion_tokens += metrics.completion_tokens
            self._final.total_tokens += metrics.total_tokens
            self._final.requests += metrics.successful_requests

    def usage_summary(self) -> str:
        with self._lock:
            c = self._final if self._final.requests else self._live
            return (
                f"[usage] {c.total_tokens} tokens "
                f"({c.prompt_tokens} prompt + {c.completion_tokens} completion), "
                f"{c.requests} requests"
            )

    def emit_note(self, message: str, agent: str | None = None) -> None:
        """Print a diagnostic note line."""
        self._emit(f"[note] {message}", agent)

    def emit_usage_summary(self) -> None:
        """Print the usage summary, plus a per-agent breakdown when useful."""
        self._emit(self.usage_summary())
        with self._lock:
            breakdown = [
                (agent, c.total_tokens, c.requests)
                for agent, c in self._per_agent.items()
            ]
        if len(breakdown) < 2:
            return
        for agent, tokens, requests in sorted(
            breakdown, key=lambda item: item[1], reverse=True
        ):
            self._emit(
                f"[usage] {agent}: {tokens} tokens, {requests} requests", agent=agent
            )

    # ------------------------------------------------------------------ output

    def _banner(self, title: str, agent: str | None = None) -> None:
        """Print a phase separator line."""
        with self._print_lock:
            color = self._color_for(agent) if agent else "white"
            try:
                self._console.print(
                    Text(f"═══ {title} ═══", style=f"bold {color}"), highlight=False
                )
            except UnicodeEncodeError:
                self._write_safe(f"=== {title} ===")
            except Exception:
                pass

    def _emit(self, line: str, agent: str | None = None) -> None:
        # Pipeline tasks run in parallel, so writes from different agents must
        # not interleave: hold a dedicated print lock around build + write.
        with self._print_lock:
            try:
                text = Text()
                if agent:
                    text.append(f"[{agent}]", style=f"bold {self._color_for(agent)}")
                    text.append(" ")
                tag = next((t for t in _TAG_COLORS if line.startswith(t)), None)
                if tag is None:
                    text.append(line)
                else:
                    text.append(tag, style=_TAG_COLORS[tag])
                    text.append(line[len(tag):])
            except Exception:
                return
            try:
                self._console.print(text, highlight=False)
            except UnicodeEncodeError:
                # A redirected stream may use a legacy encoding (e.g. cp1251) that
                # cannot represent characters like Σ or …; downgrade to ASCII.
                self._write_safe(text.plain)
            except Exception:
                pass

    def _write_safe(self, plain: str) -> None:
        """Write ``plain`` with characters unrepresentable in the stream dropped."""
        try:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe = plain.encode(encoding, "replace").decode(encoding, "replace")
            self._stream.write(safe + "\n")
            self._stream.flush()
        except Exception:
            pass


_tracer = _Tracer()


def enable_tracing(stream: TextIO | None = None) -> None:
    """Enable the compact stderr trace (idempotent)."""
    _tracer.enable(stream)


def record_usage(metrics: UsageMetrics | None) -> None:
    """Accumulate authoritative usage from a finished crew (for the summary)."""
    _tracer.record_usage(metrics)


def usage_summary() -> str:
    """Return a one-line token usage summary."""
    return _tracer.usage_summary()


def print_usage_summary() -> None:
    """Print the usage summary (and per-agent breakdown) to the trace stream."""
    _tracer.emit_usage_summary()


def trace_note(message: str, agent: str | None = None) -> None:
    """Print a diagnostic note line to the trace stream."""
    _tracer.emit_note(message, agent)
