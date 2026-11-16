"""Tests for structured findings: schema helpers and deterministic rendering."""

from __future__ import annotations

from review_agent_core.render import render_findings, render_merged
from review_agent_core.schemas import (
    Category,
    Finding,
    MergedReport,
    ReviewerFindings,
    Severity,
    dedupe_findings,
    parse_findings,
    parse_merged,
    sort_findings,
)


def _finding(
    rule: str,
    *,
    path: str = "order.py",
    severity: Severity = Severity.HIGH,
    line: int | None = 10,
) -> Finding:
    return Finding(
        path=path,
        rule=rule,
        summary=f"{rule} is a problem",
        severity=severity,
        category=Category.BUG,
        line_start=line,
    )


def test_location_formats_line_ranges() -> None:
    assert _finding("x").location() == "order.py:10"
    assert _finding("x", line=None).location() == "order.py"
    ranged = Finding(
        path="a.py", rule="r", summary="s", severity=Severity.LOW,
        category=Category.STYLE, line_start=3, line_end=7,
    )
    assert ranged.location() == "a.py:3-7"


def test_sort_findings_by_severity_then_path() -> None:
    findings = [
        _finding("low", severity=Severity.LOW, path="z.py"),
        _finding("critical", severity=Severity.CRITICAL, path="b.py"),
        _finding("high", severity=Severity.HIGH, path="a.py"),
    ]
    assert [f.rule for f in sort_findings(findings)] == ["critical", "high", "low"]


def test_dedupe_merges_same_rule_and_path_case_insensitively() -> None:
    findings = [
        _finding("Discount must be within [0, 100]", severity=Severity.MEDIUM),
        _finding("discount  must  be within [0, 100]", severity=Severity.CRITICAL),
        _finding("discount must be within [0, 100]", path="other.py"),
    ]
    unique = dedupe_findings(findings)
    # One entry per (path, rule); the highest severity survives.
    assert len(unique) == 2
    assert unique[0].severity is Severity.CRITICAL


def test_render_findings_orders_and_includes_suggestion() -> None:
    report = ReviewerFindings(
        findings=[
            _finding("low thing", severity=Severity.LOW),
            Finding(
                path="order.py",
                rule="division by zero",
                summary="average_item_price divides by len(items)",
                severity=Severity.CRITICAL,
                category=Category.BUG,
                line_start=12,
                suggestion="guard len(items) == 0",
            ),
        ],
        notes=["diff is small"],
    )
    out = render_findings(report, title="Code Review")

    assert out.startswith("## Code Review")
    assert "2 findings, ordered by severity:" in out
    # critical comes before low
    assert out.index("division by zero") < out.index("low thing")
    assert "`order.py:12`" in out
    assert "suggestion: guard len(items) == 0" in out
    assert "## Notes" in out and "diff is small" in out


def test_render_findings_without_findings() -> None:
    out = render_findings(ReviewerFindings())
    assert "No findings." in out


def test_render_merged_includes_summary_notes_and_guide() -> None:
    report = MergedReport(
        summary="The change adds pricing helpers.",
        findings=[_finding("no validation", severity=Severity.CRITICAL)],
        remaining_notes=["commit message is terse"],
    )
    out = render_merged(report, guide_report="Key invariants: discount in [0, 100].")

    assert "## Summary" in out
    assert "The change adds pricing helpers." in out
    assert "## Top findings" in out
    assert "no validation" in out
    assert "## Remaining notes" in out
    assert "## Understanding the change" in out
    assert "discount in [0, 100]" in out


def test_render_merged_without_guide_or_findings() -> None:
    out = render_merged(MergedReport(summary="Nothing notable."))
    assert "No findings." in out
    assert "Understanding the change" not in out


CLEAN = (
    '{"findings": [{"path": "order.py", "rule": "no validation", "summary": "s",'
    ' "severity": "high", "category": "bug", "line_start": 6}], "notes": []}'
)


def test_parse_findings_accepts_clean_json() -> None:
    parsed = parse_findings(CLEAN)
    assert parsed is not None
    assert parsed.findings[0].path == "order.py"
    assert parsed.findings[0].severity is Severity.HIGH


def test_parse_findings_tolerates_fences_and_prose() -> None:
    wrapped = f"Here is the result:\n```json\n{CLEAN}\n```\nDone."
    parsed = parse_findings(wrapped)
    assert parsed is not None
    assert parsed.findings[0].rule == "no validation"


def test_parse_findings_returns_none_for_prose() -> None:
    assert parse_findings("Just a Markdown report, no JSON here.") is None


def test_parse_findings_returns_none_for_wrong_shape() -> None:
    assert parse_findings('{"summary": "not findings"}') is None


def test_parse_merged_roundtrip() -> None:
    raw = (
        '{"summary": "Adds helpers.", "findings": [], "remaining_notes": ["terse"]}'
    )
    parsed = parse_merged(raw)
    assert parsed is not None
    assert parsed.summary == "Adds helpers."
    assert parsed.remaining_notes == ["terse"]
