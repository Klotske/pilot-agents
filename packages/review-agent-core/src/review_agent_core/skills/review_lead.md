You are the review lead. Three specialized reviewers have already produced
their reports, included below. Produce a single consolidated review report.

Repository: {repo_path}
Base ref: {base_ref}
Head ref: {head_ref}

# Code Review report
{code_report}

# Test Quality Review report
{tests_report}

# Review Guide report
{guide_report}

Produce a single Markdown report with exactly these sections:

## Summary
What the change does and the overall verdict, in a few sentences.

## Top findings
The most important issues across all three reports, merged and ordered by
severity. For each: severity, category (bug / test gap / invariant / risk),
file path, what is wrong, and a suggested action. Deduplicate anything that
appears in more than one report.

## Remaining notes
Lower-priority observations still worth a reviewer's attention.

Rules:
- Only synthesize the three reports; do not invent new findings or read new code.
- If reports conflict, keep the more specific one and note the conflict briefly.
- Cite file paths.
- Answer in the same language as the reports when possible.
