# review-agent-cli

Command-line entry point for `review-agent-core`.

## Install

```bash
uv sync
```

## Usage

```bash
# review the current branch against main (all three modes)
review-agent review

# review a specific branch, only the Code Reviewer
review-agent review feature --code

# review tests only, against an explicit base
review-agent review feature --base develop --tests

# combined with more control
review-agent review feature --code --guide
```

Flags are additive: `--code`, `--tests`, `--guide` select modes (each runs its
own single-agent crew). `--all` (or no flags) runs the full pipeline as one
Crew: Review Guide → (Code Reviewer ∥ Test Quality Reviewer) → Review Lead,
where the guide's output is passed to the parallel reviewers and the lead
merges all three reports.
`--verbose` additionally turns on CrewAI's full rich console output.

A compact, colorized trace is printed to **stderr**, attributed to the agent
that produced each line:

```
═══ Code Reviewer (5 tools) ═══
[Code Reviewer] [llm] deepseek-flash request: msgs=3 chars=5120, tools=5, temp=0.2
[Code Reviewer] [tool] get_diff_stat -> order.py | 13 +++- (34ms)
[Code Reviewer] [llm] deepseek-flash 1200(tok) prompt + 300(tok) completion = 1500(tok) this step | Σ 1500(tok) running | finish=stop call=llm_call 2.10s
[Code Reviewer] [error] read_file failed: path not found
[Code Reviewer] [agent] done in 8.40s
[usage] 42000 tokens (38000 prompt + 4000 completion), 9 requests
```

so the Markdown report on stdout stays clean and can be redirected:

```bash
review-agent review feature > report.md
```

The CLI loads `.env` from the current directory (or any parent) via
`python-dotenv`. Environment variables already set in the shell take precedence.

| Variable | Required | Notes |
| --- | --- | --- |
| `LLM_MODEL` | yes | model name, e.g. `openai/gpt-4o-mini` |
| `LLM_BASE_URL` | no | OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1` |
| `LLM_API_KEY` | no | API key (not required for local endpoints) |
| `LLM_TEMPERATURE` | no | float |
| `GIT_REPO` | no | path to the repository (default: current directory) |
| `GIT_BASE` | no | base ref (default: main, then master) |

## Tests

```bash
uv run pytest -q
```
