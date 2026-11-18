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
from dataclasses import dataclass
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

# Label for LLM calls made without an agent (CrewAI's structured-output
# conversion, internal analysis): they cost real tokens and must be visible.
# Labels for LLM calls made without an agent. CrewAI uses direct calls for two
# very different jobs, and they cost very different amounts:
#   * ``direct`` -- converting an answer into the ``output_pydantic`` schema;
#   * ``direct/recovery`` -- a forced final answer after ``max_iter`` was used up,
#     which continues the agent's own history (it contains ``tool`` messages).
# Keeping them apart makes a run readable without reconstructing it from msgs=N.
DIRECT_AGENT = "direct"
DIRECT_RECOVERY = "direct/recovery"

# CrewAI summarizes a request on its own once it reaches 85% of the window.
CREWAI_PREFLIGHT_RATIO = 0.85


@dataclass(frozen=True)
class ContextBudget:
    """The window a run was budgeted against, for the per-call ``ctx`` stats."""

    window: int
    trigger: int
    target: int

    @classmethod
    def from_window(cls, window: int, trigger_ratio: float, target_ratio: float) -> ContextBudget:
        """Build a budget from a raw model window and the compaction ratios."""
        return cls(
            window=window,
            trigger=int(window * trigger_ratio),
            target=int(window * target_ratio),
        )

    def percent(self, tokens: int) -> str:
        """Format ``tokens`` as a share of the window (``83%``, ``0.5%``)."""
        if self.window <= 0:
            return "?"
        percent = 100 * tokens / self.window
        return f"{percent:.1f}%" if percent < 10 else f"{percent:.0f}%"

    def marker(self, tokens: int) -> str:
        """Return the warning marker for a request of ``tokens`` input tokens."""
        if tokens >= self.window:
            return " ⚠window"
        if tokens >= self.window * CREWAI_PREFLIGHT_RATIO:
            return " ⚠crewai"
        if tokens >= self.trigger:
            return " ⚠trigger"
        return ""

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
    "[compact]": "bold magenta",
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


def _direct_label(messages: Any) -> str:
    """Label an agent-less LLM call by what it is actually doing.

    CrewAI's ``output_pydantic`` conversion sends exactly two messages (an
    instruction and the answer). A forced final answer after ``max_iter`` resumes
    the agent's own history, which by definition contains ``tool`` messages.
    """
    if isinstance(messages, list):
        for message in cast("list[Any]", messages):
            if not isinstance(message, dict):
                continue
            entry = cast("dict[str, Any]", message)
            if entry.get("role") == "tool":
                return DIRECT_RECOVERY
    return DIRECT_AGENT


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
        self._budget: ContextBudget | None = None
        self._last_prompt: dict[str, int] = {}          # agent -> last prompt tokens
        self._peak_prompt: dict[str, int] = {}          # agent -> peak prompt tokens
        self._call_label: dict[str, str] = {}           # call_id -> trace label

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
            agent = event.agent_role or _direct_label(event.messages)
            if event.call_id:
                with self._lock:
                    self._llm_started[event.call_id] = event.timestamp
                    if event.agent_role is None:
                        self._call_label[event.call_id] = agent
            details: list[str] = [_messages_summary(event.messages)]
            if event.tools:
                details.append(f"tools={len(event.tools)}")
            for label, value in (
                ("temp", event.temperature),
                ("max_tokens", event.max_tokens),
            ):
                if value is not None:
                    details.append(f"{label}={value}")
            with self._lock:
                last = self._last_prompt.get(agent)
            if last is not None:
                # Calibration for the compaction estimate: it counts message
                # content only, the provider also bills the tool schemas.
                details.append(f"last actual {last}(tok)")
            model = event.model or "llm"
            self._emit(f"[llm] {model} request: {', '.join(details)}", agent)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_completed(_source: object, event: LLMCallCompletedEvent) -> None:
            usage = event.usage or {}
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            total = int(usage.get("total_tokens", prompt + completion))
            agent = event.agent_role or DIRECT_AGENT
            if event.agent_role is None:
                with self._lock:
                    agent = self._call_label.pop(event.call_id, DIRECT_AGENT)
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
                self._last_prompt[agent] = prompt
                self._peak_prompt[agent] = max(self._peak_prompt.get(agent, 0), prompt)
                started = self._llm_started.pop(event.call_id, None)
                budget = self._budget
            model = event.model or "llm"
            extras: list[str] = []
            if budget is not None:
                extras.append(
                    f"ctx {prompt}/{budget.window} {budget.percent(prompt)}"
                    f"{budget.marker(prompt)}"
                )
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
                agent,
            )

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_llm_failed(_source: object, event: LLMCallFailedEvent) -> None:
            with self._lock:
                self._llm_started.pop(event.call_id, None)
                label = self._call_label.pop(event.call_id, None)
            model = event.model or "llm"
            self._emit(
                f"[error] {model} call failed: {_head(event.error, MAX_ERROR_CHARS)}",
                event.agent_role or label,
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

    def set_budget(self, budget: ContextBudget | None) -> None:
        """Budget the run's requests against, and reset the per-run peaks."""
        with self._lock:
            self._budget = budget
            self._last_prompt.clear()
            self._peak_prompt.clear()

    def last_prompt_tokens(self, agent: str) -> int | None:
        """Input tokens of ``agent``'s most recent request, when known."""
        with self._lock:
            return self._last_prompt.get(agent)

    def record_usage(self, metrics: UsageMetrics | None) -> None:
        """Cross-check CrewAI's crew rollup against the per-call events.

        The rollup is not authoritative: on a multi-task pipeline it reports the
        same calls once per task (seen as exactly xN). The event counters are
        exact, so the rollup is only compared against them.
        """
        if metrics is None:
            return
        with self._lock:
            self._final.prompt_tokens += metrics.prompt_tokens
            self._final.completion_tokens += metrics.completion_tokens
            self._final.total_tokens += metrics.total_tokens
            self._final.requests += metrics.successful_requests
            live_total = self._live.total_tokens
            live_requests = self._live.requests
        if not live_total or not metrics.total_tokens:
            return
        drift = abs(metrics.total_tokens - live_total) / live_total
        if drift <= 0.05:
            return
        self._emit(
            f"[usage] note: crewai rolled up {metrics.total_tokens} tok / "
            f"{metrics.successful_requests} requests, events say {live_total} / "
            f"{live_requests} — using the event counts"
        )

    def usage_summary(self) -> str:
        with self._lock:
            # Event counters are exact; the rollup is only a fallback for runs
            # where tracing was never enabled and no events were recorded.
            c = self._live if self._live.requests else self._final
            budget = self._budget
            peak = max(self._peak_prompt.values(), default=0)
        summary = (
            f"[usage] {c.total_tokens} tokens "
            f"({c.prompt_tokens} prompt + {c.completion_tokens} completion), "
            f"{c.requests} requests"
        )
        if budget is not None and peak:
            summary += (
                f", peak ctx {peak}/{budget.window} {budget.percent(peak)}"
                f"{budget.marker(peak)}"
            )
        return summary

    def emit_note(self, message: str, agent: str | None = None) -> None:
        """Print a diagnostic note line.

        A message that already starts with a known tag (``[compact]``, ...) is
        printed as is, so notes can carry their own category.
        """
        tagged = any(message.startswith(tag) for tag in _TAG_COLORS)
        self._emit(message if tagged else f"[note] {message}", agent)

    def emit_usage_summary(self) -> None:
        """Print the usage summary, plus a per-agent breakdown when useful."""
        self._emit(self.usage_summary())
        with self._lock:
            breakdown = [
                (agent, c.total_tokens, c.requests, self._peak_prompt.get(agent, 0))
                for agent, c in self._per_agent.items()
            ]
            budget = self._budget
        if len(breakdown) < 2:
            return
        for agent, tokens, requests, peak in sorted(
            breakdown, key=lambda item: item[1], reverse=True
        ):
            peak_note = ""
            if budget is not None and peak:
                peak_note = (
                    f", peak ctx {peak}/{budget.window} {budget.percent(peak)}"
                    f"{budget.marker(peak)}"
                )
            self._emit(
                f"[usage] {agent}: {tokens} tokens, {requests} requests{peak_note}",
                agent=agent,
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


def set_context_budget(budget: ContextBudget | None) -> None:
    """Set the window a run is budgeted against and reset its peaks."""
    _tracer.set_budget(budget)


def last_prompt_tokens(agent: str) -> int | None:
    """Input tokens of ``agent``'s most recent request, when tracing saw it."""
    return _tracer.last_prompt_tokens(agent)


def record_usage(metrics: UsageMetrics | None) -> None:
    """Cross-check a finished crew's rollup against the per-call events."""
    _tracer.record_usage(metrics)


def usage_summary() -> str:
    """Return a one-line token and context-window usage summary."""
    return _tracer.usage_summary()


def print_usage_summary() -> None:
    """Print the usage summary (and per-agent breakdown) to the trace stream."""
    _tracer.emit_usage_summary()


def trace_note(message: str, agent: str | None = None) -> None:
    """Print a diagnostic note line to the trace stream."""
    _tracer.emit_note(message, agent)
