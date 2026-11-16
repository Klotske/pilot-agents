"""Render structured review data as Markdown.

Rendering is deterministic: the same findings always produce the same report, so
the final document does not depend on how a model phrased its answer.
"""

from __future__ import annotations

from .schemas import Finding, MergedReport, ReviewerFindings, sort_findings


def render_finding(finding: Finding) -> list[str]:
    """Render one finding as Markdown lines."""
    lines = [
        f"- **[{finding.severity}]** `{finding.location()}` — {finding.rule}",
        f"  - category: {finding.category}",
        f"  - {finding.summary}",
    ]
    if finding.suggestion:
        lines.append(f"  - suggestion: {finding.suggestion}")
    return lines


def _finding_count(count: int) -> str:
    """Return ``N finding(s), ordered by severity`` with correct pluralization."""
    noun = "finding" if count == 1 else "findings"
    return f"{count} {noun}, ordered by severity:"


def render_findings(report: ReviewerFindings, *, title: str = "Findings") -> str:
    """Render a single reviewer's findings."""
    ordered = sort_findings(report.findings)
    lines = [f"## {title}"]

    if not ordered:
        lines.append("No findings.")
    else:
        lines.append(_finding_count(len(ordered)))
        lines.append("")
        for finding in ordered:
            lines.extend(render_finding(finding))

    if report.notes:
        lines.append("")
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in report.notes)

    return "\n".join(lines)


def render_merged(report: MergedReport, *, guide_report: str | None = None) -> str:
    """Render the consolidated report produced by the review lead."""
    lines = ["## Summary", "", report.summary.strip() or "No summary produced.", ""]

    ordered = sort_findings(report.findings)
    lines.append("## Top findings")
    lines.append("")
    if not ordered:
        lines.append("No findings.")
    else:
        lines.append(_finding_count(len(ordered)))
        lines.append("")
        for finding in ordered:
            lines.extend(render_finding(finding))

    if report.remaining_notes:
        lines.append("")
        lines.append("## Remaining notes")
        lines.extend(f"- {note}" for note in report.remaining_notes)

    if guide_report and guide_report.strip():
        lines.append("")
        lines.append("## Understanding the change")
        lines.append("")
        lines.append(guide_report.strip())

    return "\n".join(lines)
