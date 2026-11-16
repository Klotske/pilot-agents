"""Structured review output.

The reviewers return findings, not free-form prose, so that merging is a
deterministic operation on data instead of text matching in an LLM prompt.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError


class Severity(StrEnum):
    """How much attention a finding deserves."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


class Category(StrEnum):
    """What kind of problem a finding describes."""

    BUG = "bug"
    TEST_GAP = "test_gap"
    INVARIANT = "invariant"
    RISK = "risk"
    STYLE = "style"


class Finding(BaseModel):
    """A single actionable observation about the change."""

    path: str = Field(description="Repository-relative path of the affected file.")
    rule: str = Field(
        description="Short title of the problem or the violated rule, e.g. "
        "'discount must be within [0, 100]'."
    )
    summary: str = Field(description="What is wrong and why it matters.")
    severity: Severity = Field(description="critical, high, medium or low.")
    category: Category = Field(description="bug, test_gap, invariant, risk or style.")
    line_start: int | None = Field(
        default=None, description="First affected line in the new file, if known."
    )
    line_end: int | None = Field(
        default=None, description="Last affected line, if it spans several lines."
    )
    suggestion: str | None = Field(
        default=None, description="Concrete fix or the test that should be added."
    )

    def location(self) -> str:
        """Return ``path`` or ``path:line`` for display."""
        if self.line_start is None:
            return self.path
        if self.line_end is not None and self.line_end != self.line_start:
            return f"{self.path}:{self.line_start}-{self.line_end}"
        return f"{self.path}:{self.line_start}"

    def dedupe_key(self) -> tuple[str, str]:
        """Identity of a finding, used to merge duplicates across reviewers."""
        rule = " ".join(self.rule.lower().split())
        return (self.path.lstrip("/"), rule)


class ReviewerFindings(BaseModel):
    """What a single reviewer returns."""

    findings: list[Finding] = Field(default_factory=list[Finding])
    notes: list[str] = Field(
        default_factory=list[str],
        description="Lower-priority observations that are not worth a finding.",
    )


class MergedReport(BaseModel):
    """What the review lead returns."""

    summary: str = Field(
        description="What the change does and the overall verdict, in a few sentences."
    )
    findings: list[Finding] = Field(
        default_factory=list[Finding],
        description="All findings across reviewers, deduplicated and ordered by severity.",
    )
    remaining_notes: list[str] = Field(
        default_factory=list[str],
        description="Lower-priority observations worth a reviewer's attention.",
    )


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Order findings by severity, then by path and line."""
    return sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER.get(f.severity, len(SEVERITY_ORDER)),
            f.path,
            f.line_start or 0,
        ),
    )


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Drop repeated findings, keeping the first (highest-severity) occurrence."""
    seen: set[tuple[str, str]] = set()
    unique: list[Finding] = []
    for finding in sort_findings(findings):
        key = finding.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _json_candidate(text: str) -> str | None:
    """Return the outermost JSON object found in ``text``, if any."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_as[T: (ReviewerFindings, MergedReport)](
    model: type[T], text: str, required_keys: frozenset[str]
) -> T | None:
    """Parse ``text`` into ``model``, tolerating surrounding prose/fences.

    Every field of these models has a default, so any JSON object would validate
    into "no findings". ``required_keys`` guards against silently treating an
    unrelated object as an empty report.
    """
    for candidate in (text.strip(), _json_candidate(text)):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not required_keys <= payload.keys():
            continue
        try:
            return model.model_validate(payload)
        except ValidationError:
            continue
    return None


def parse_findings(text: str) -> ReviewerFindings | None:
    """Parse a reviewer's raw answer into findings, or ``None`` if it is not JSON."""
    return _parse_as(ReviewerFindings, text, frozenset({"findings"}))


def parse_merged(text: str) -> MergedReport | None:
    """Parse the lead's raw answer into a merged report, or ``None``."""
    return _parse_as(MergedReport, text, frozenset({"summary", "findings"}))
