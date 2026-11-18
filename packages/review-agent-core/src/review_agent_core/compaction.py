"""Proactive, deterministic compaction of an agent's LLM message history.

CrewAI's built-in compaction (``Agent.respect_context_window=True``) triggers
at 85% of the window -- or when the provider rejects a call -- and then replaces
*every* non-system message with a single summary, including the task prompt and
the structured-output schema. For a review agent that is destructive: the
reviewer loses its instructions and the answer stops validating against
``Task.output_pydantic``.

This module compacts *before* the call instead, keeping the invariants the
pipeline depends on:

* all ``system`` messages and the first ``user`` message (task description,
  ``Task.context`` reports, pydantic schema) are never touched;
* the last ``keep_last_steps`` steps stay verbatim, so the final answer still
  matches the requested format;
* ``assistant``/``tool`` pairing is preserved: no message is dropped or merged,
  only ``content`` is shortened. A tool message must keep the ``tool_call_id``
  of the assistant message that produced it, or OpenAI-compatible providers
  reject the request with HTTP 400;
* identical tool results are deduplicated, which is usually the biggest win --
  reviewers tend to call ``get_diff`` more than once;
* shortening is idempotent: content that already carries the truncation marker
  is never shortened again.

The hook is installed per crew run via
:func:`crewai.hooks.dispatch.scoped_hooks`, not globally: a module-level
``@before_llm_call`` would affect every crew in the process and is snapshotted
into executors at construction time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import re
import threading
from typing import Any, Protocol, cast

from crewai.hooks import LLMCallHookContext

from .config import CompactionConfig
from .tracing import last_prompt_tokens, trace_note

Message = dict[str, Any]

_MARKER_TEMPLATE = "⟪truncated {removed} chars⟫"
_MARKER_PREFIX = "⟪truncated "
_STRUCTURE_MARKER = "⟪kept the change map / declarations⟫"
_DUPLICATE_TEMPLATE = "⟪same result as the earlier {name} call⟫"

# Chat formatting adds separators and role tokens per message; counting them
# keeps the estimate from drifting low on long histories.
_MESSAGE_OVERHEAD_TOKENS = 4
_DEDUPE_MIN_CHARS = 400
_MAX_CACHE_ENTRIES = 4096
_STRUCTURE_MAX_LINES = 300

# Model families that use the o200k BPE. Everything else (gateway aliases,
# DeepSeek/Qwen/Kimi ids, local models) is approximated with cl100k_base.
_O200K_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")

# Lines that are cheap to keep and expensive to lose: the map of a diff or of a
# source file stays intact while the bodies -- what actually bloats the prompt --
# are shortened.
_STRUCTURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "get_diff": re.compile(
        r"^(diff --git |index |@@ |Binary files |new file mode|deleted file mode|rename (from|to))"
    ),
    "read_file": re.compile(
        r"^\s*(def |class |async def |func |type |interface |export |public |private |protected )"
    ),
}


class Tokenizer(Protocol):
    """Counts tokens in a string."""

    def count(self, text: str) -> int: ...


def _encoding_name(model: str, override: str | None) -> str:
    """Pick a tiktoken encoding for ``model``, which may be a gateway alias."""
    if override:
        return override
    bare = model.rsplit("/", 1)[-1].lower()
    return "o200k_base" if bare.startswith(_O200K_PREFIXES) else "cl100k_base"


class TiktokenCounter:
    """Exact token counts with the ``tiktoken`` BPE chosen for ``model``."""

    def __init__(self, model: str, encoding: str | None = None) -> None:
        import tiktoken  # imported here so the fallback path never needs it

        self._encoding = tiktoken.get_encoding(_encoding_name(model, encoding))

    def count(self, text: str) -> int:
        if not text:
            return 0
        # ``disallowed_special`` must be empty: diffs and sources legitimately
        # contain literals such as ``<|endoftext|>`` and must not raise.
        return len(self._encoding.encode(text, disallowed_special=()))


class CharCounter:
    """Cheap estimate used when ``tiktoken`` cannot be built (offline, no cache)."""

    def __init__(self, chars_per_token: float) -> None:
        self._chars_per_token = max(1.0, chars_per_token)

    def count(self, text: str) -> int:
        return int(len(text) / self._chars_per_token) + 1


def build_tokenizer(model: str, cfg: CompactionConfig) -> Tokenizer:
    """Return an exact tokenizer, degrading to an estimate when unavailable.

    ``tiktoken`` downloads its BPE file on first use, which may be impossible in
    an air-gapped environment. A review must not fail because of a tokenizer, so
    the character heuristic takes over and the trace says so.
    """
    try:
        return TiktokenCounter(model, cfg.tiktoken_encoding)
    except Exception as error:  # noqa: BLE001 - any tiktoken failure means "no BPE"
        trace_note(f"[compact] tiktoken unavailable ({error}); using a char estimate")
        return CharCounter(cfg.chars_per_token)


@dataclass(frozen=True)
class CompactionResult:
    """What a compaction pass did (and what it was up against), for the trace."""

    before_tokens: int
    after_tokens: int
    limit_tokens: int
    target_tokens: int = 0
    protected_tokens: int = 0
    truncated: int = 0
    deduped: int = 0

    @property
    def changed(self) -> bool:
        return self.truncated > 0 or self.deduped > 0

    @property
    def near_limit(self) -> bool:
        """Whether the estimate is close enough to the trigger to be worth logging."""
        return self.limit_tokens > 0 and self.before_tokens >= 0.8 * self.limit_tokens


class _Counter:
    """Message token counter with a bounded cache keyed by content hash.

    The same history is measured on every LLM call of a task; hashing is
    microseconds where re-encoding tens of kilobytes is milliseconds.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._cache: dict[bytes, int] = {}
        self._lock = threading.Lock()

    def text(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        tokens = self._tokenizer.count(text)
        with self._lock:
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                self._cache.clear()
            self._cache[key] = tokens
        return tokens

    def message(self, message: Mapping[str, Any]) -> int:
        total = _MESSAGE_OVERHEAD_TOKENS
        content = message.get("content")
        if isinstance(content, str):
            total += self.text(content)
        elif content is not None:
            total += self.text(str(content))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total += self.text(json.dumps(tool_calls, ensure_ascii=False, default=str))
        if message.get("name"):
            total += _MESSAGE_OVERHEAD_TOKENS
        return total


def _steps(messages: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Group messages into steps: an assistant turn plus the tool results it produced."""
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        start = index
        role = messages[index].get("role")
        index += 1
        if role == "assistant":
            while index < len(messages) and messages[index].get("role") == "tool":
                index += 1
        groups.append((start, index))
    return groups


def _protected(
    messages: Sequence[Mapping[str, Any]],
    keep_last_steps: int,
    *,
    counter: _Counter,
    budget: int,
) -> set[int]:
    """Return indices that must never be shortened.

    That is every ``system`` message, the first ``user`` message (the task
    prompt, the upstream reports and the output schema all live there), and up to
    ``keep_last_steps`` trailing steps the agent still needs to answer.

    Trailing steps are protected newest first, and an older one is dropped from
    protection as soon as it would leave nothing to cut: on a small window a
    single ``get_diff`` result can fill most of it, and protecting that step too
    silently turns compaction into a no-op (the request then exceeds the window
    and CrewAI's built-in summarizer takes over). The newest step always stays
    protected when ``keep_last_steps > 0``: the agent has just asked for it.
    """
    protected = {
        index for index, message in enumerate(messages) if message.get("role") == "system"
    }
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            protected.add(index)
            break
    if keep_last_steps <= 0:
        return protected

    used = sum(counter.message(messages[index]) for index in protected)
    kept = 0
    for start, end in reversed(_steps(messages)):
        if kept >= keep_last_steps:
            break
        step = sum(counter.message(messages[index]) for index in range(start, end))
        if kept and used + step > budget:
            break
        protected.update(range(start, end))
        used += step
        kept += 1
    return protected


def _tool_name(message: Mapping[str, Any]) -> str | None:
    name = message.get("name")
    return str(name) if name else None


def _content_hash(content: str) -> bytes:
    return hashlib.blake2b(content.encode("utf-8", "replace"), digest_size=16).digest()


def _dedupe(messages: list[Message]) -> int:
    """Replace repeated tool results with a pointer to the first occurrence.

    An occurrence is identified by tool name plus content: two ``get_diff``
    calls over the same scope return byte-identical text, and the same file read
    at two refs that did not change returns the same bytes, so keeping both
    costs the window for nothing. The first occurrence in message order stays
    authoritative.

    A duplicate is usually the *newest* message -- the agent has just asked for
    something it already has. Replacing it loses nothing: the identical text is
    still earlier in the very same request. Protecting it instead (as
    ``keep_last_steps`` does for shortening) would make deduplication a no-op
    exactly when it matters.
    """
    seen: dict[tuple[str, bytes], int] = {}
    deduped = 0
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) < _DEDUPE_MIN_CHARS:
            continue
        if _MARKER_PREFIX in content:
            continue
        key = (_tool_name(message) or "", _content_hash(content))
        first = seen.setdefault(key, index)
        if first == index:
            continue
        messages[index] = {
            **message,
            "content": _DUPLICATE_TEMPLATE.format(name=_tool_name(message) or "tool"),
        }
        deduped += 1
    return deduped


def _structure_block(name: str | None, content: str, limit_chars: int) -> str:
    """Return the diff headers or declaration lines of ``content``.

    Bounded by ``limit_chars``: a large source file has enough declarations to
    fill the whole budget, which would defeat the point of shortening it.
    """
    pattern = _STRUCTURE_PATTERNS.get(name or "")
    if pattern is None or limit_chars <= 0:
        return ""
    selected: list[str] = []
    used = 0
    for line in content.splitlines():
        if not pattern.match(line):
            continue
        if used + len(line) + 1 > limit_chars or len(selected) >= _STRUCTURE_MAX_LINES:
            break
        selected.append(line)
        used += len(line) + 1
    return "\n".join(selected)


def _shorten(
    content: str,
    *,
    budget: int,
    name: str | None,
    cfg: CompactionConfig,
) -> str:
    """Shorten ``content`` to about ``budget`` chars, keeping its structure."""
    if len(content) <= budget or len(content) <= cfg.min_chars_per_message:
        return content
    structure = ""
    if cfg.structure_aware:
        structure = _structure_block(name, content, int(budget * 0.4))
    available = max(cfg.min_chars_per_message, budget - len(structure))
    head_share = cfg.tool_result_head / (cfg.tool_result_head + cfg.tool_result_tail)
    head_chars = int(available * head_share)
    tail_chars = available - head_chars
    removed = len(content) - head_chars - tail_chars
    parts = [content[:head_chars]]
    if structure:
        parts.append(f"\n{_STRUCTURE_MARKER}\n{structure}\n")
    parts.append(f"\n{_MARKER_TEMPLATE.format(removed=max(removed, 0))}\n")
    if tail_chars:
        parts.append(content[-tail_chars:])
    shortened = "".join(parts)
    # The structure block and the marker can outweigh a short message; never
    # trade a result that is longer than the input for it.
    return shortened if len(shortened) < len(content) else content


def _shorten_reasoning(content: str, *, budget: int, cfg: CompactionConfig) -> str:
    """Keep the first line of an assistant message and drop the rest."""
    if budget < cfg.min_chars_per_message or len(content) <= budget:
        return content
    first_line = content.splitlines()[0] if content.splitlines() else content
    kept = first_line[:budget]
    removed = len(content) - len(kept)
    return f"{kept}\n{_MARKER_TEMPLATE.format(removed=removed)}"


def compact_messages(
    messages: list[Message],
    *,
    cfg: CompactionConfig,
    window_tokens: int,
    counter: _Counter,
) -> CompactionResult:
    """Deduplicate repeated tool results and shorten the history in place.

    Deduplication runs on every call: a repeated ``get_diff`` costs the window
    for nothing, whether or not the history is close to the trigger. Shortening
    only starts once the estimate exceeds ``trigger_ratio`` of ``window_tokens``
    and then aims at ``target_ratio``, so following calls do not re-trigger.
    """
    limit = int(window_tokens * cfg.trigger_ratio)
    target = int(window_tokens * cfg.target_ratio)
    before = sum(counter.message(message) for message in messages)

    # Deduplicate first: the protection decision below is size-based, and a
    # pointer is much smaller than the copy it replaces.
    deduped = _dedupe(messages)
    protected = _protected(
        messages, cfg.keep_last_steps, counter=counter, budget=target
    )
    protected_tokens = sum(counter.message(messages[index]) for index in protected)
    after_dedup = sum(counter.message(message) for message in messages)

    def result(**kwargs: int) -> CompactionResult:
        return CompactionResult(
            before,
            after_dedup,
            limit,
            target_tokens=target,
            protected_tokens=protected_tokens,
            deduped=deduped,
            **kwargs,
        )

    if after_dedup <= limit:
        return result()

    editable = [index for index in range(len(messages)) if index not in protected]
    middle = sum(counter.message(messages[index]) for index in editable)
    fixed = after_dedup - middle
    allowed = target - fixed
    if allowed <= 0 or middle <= 0:
        # The protected part alone exceeds the target: nothing sensible to cut.
        return result()

    scale = allowed / middle
    truncated = 0
    for index in editable:
        message = messages[index]
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        share = int(len(content) * scale)
        if message.get("role") == "tool":
            budget = min(share, cfg.tool_result_head + cfg.tool_result_tail)
            shortened = _shorten(
                content, budget=budget, name=_tool_name(message), cfg=cfg
            )
        else:
            shortened = _shorten_reasoning(
                content, budget=min(share, cfg.step_head), cfg=cfg
            )
        if shortened == content:
            continue
        messages[index] = {**message, "content": shortened}
        truncated += 1

    after = sum(counter.message(message) for message in messages)
    return CompactionResult(
        before,
        after,
        limit,
        target_tokens=target,
        protected_tokens=protected_tokens,
        truncated=truncated,
        deduped=deduped,
    )


class Compactor:
    """Stateful ``before_llm_call`` hook that compacts one crew run."""

    def __init__(
        self,
        cfg: CompactionConfig,
        *,
        model: str,
        window_tokens: int,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if window_tokens <= 0:
            raise ValueError("window_tokens must be a positive number of tokens")
        self._cfg = cfg
        self._window = window_tokens
        self._counter = _Counter(tokenizer or build_tokenizer(model, cfg))
        # Guards against re-entering the hook from a call made *by* a hook.
        self._inside: ContextVar[bool] = ContextVar(
            f"review_agent_compacting_{id(self)}", default=False
        )

    def hook(self, context: LLMCallHookContext) -> None:
        """Compact the pending request, in place, then let it proceed."""
        if not self._cfg.enabled or self._inside.get():
            return
        agent = getattr(context, "agent", None)
        if agent is None:
            # Direct LLM calls (memory analysis, a summarizer) carry no agent
            # history worth compacting, and compacting them could recurse.
            return
        role = str(getattr(agent, "role", "agent"))
        token = self._inside.set(True)
        try:
            result = compact_messages(
                cast("list[Message]", context.messages),
                cfg=self._cfg,
                window_tokens=self._window,
                counter=self._counter,
            )
        except Exception as error:  # noqa: BLE001 - never break a review over compaction
            trace_note(f"[compact] failed, continuing uncompacted: {error}", role)
            return
        finally:
            self._inside.reset(token)
        if result.changed:
            trace_note(self._describe(result), role)
        elif result.near_limit:
            # A silent compactor is indistinguishable from a broken one: on a
            # near miss say why nothing was cut.
            trace_note(self._explain(result, role), role)

    def describe(self) -> str:
        """Describe the active budget, so a silent run can be explained."""
        return (
            f"[compact] enabled: window={self._window} tok, "
            f"trigger={int(self._window * self._cfg.trigger_ratio)}, "
            f"target={int(self._window * self._cfg.target_ratio)}, "
            f"keep_last_steps={self._cfg.keep_last_steps}"
        )

    @staticmethod
    def _describe(result: CompactionResult) -> str:
        parts = [
            f"{result.before_tokens}->{result.after_tokens} tok "
            f"(trigger {result.limit_tokens}, target {result.target_tokens})"
        ]
        if result.deduped:
            parts.append(f"dedup={result.deduped}")
        if result.truncated:
            parts.append(f"truncated={result.truncated}")
        return "[compact] " + ", ".join(parts)

    @staticmethod
    def _explain(result: CompactionResult, role: str) -> str:
        """Explain a near miss: the estimate is close, yet nothing was cut."""
        parts = [
            f"nothing cut: est {result.before_tokens}/{result.limit_tokens} "
            f"({100 * result.before_tokens // max(result.limit_tokens, 1)}%)"
        ]
        actual = last_prompt_tokens(role)
        if actual and result.before_tokens:
            drift = round(100 * (actual / result.before_tokens - 1))
            parts.append(f"last actual {actual} ({drift:+d}%)")
        if result.protected_tokens >= result.target_tokens:
            parts.append(
                f"protected {result.protected_tokens} ≥ target "
                f"{result.target_tokens} (system, task, trailing steps)"
            )
        else:
            parts.append("every remaining message is already short")
        return "[compact] " + ", ".join(parts)
