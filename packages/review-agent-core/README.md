# review-agent-core

A [CrewAI](https://docs.crewai.com/)-based agent core that reviews changes in a
local git repository. Three review modes are provided, each as a single-agent
Crew over read-only git tools:

- **Code Reviewer** — finds potential bugs, risks, and complex areas.
- **Test Quality Reviewer** — analyzes tests, finds untested scenarios and weak tests.
- **Review Guide** — explains business logic, highlights invariants, prepares review comments.
- **Review Lead** (`review_all`) — merges the three reports into one deduplicated, ranked report.

Reviewers return **structured findings** (`Finding`: path, lines, severity,
category, rule, summary, suggestion) rather than free-form prose: the tasks set
`output_pydantic`, so CrewAI appends the pydantic schema to the prompt and
validates the answer. The lead merges them, and the final Markdown is rendered
deterministically from that data (`review_agent_core.render`). If a model's answer
cannot be validated, the raw answer text is parsed and, failing that, printed
as-is instead of breaking the run.

**Gateway caveat.** CrewAI only skips the provider-level `response_format`
request when the agent has at least one tool (`effective_response_model = None if
self.original_tools else ...`). Without tools it sends a `json_schema`
response_format, which some OpenAI-compatible gateways reject with HTTP 400 —
that is why the lead also gets the (read-only) git tools.

`review_all` runs them as a single Crew wired as a DAG:

```
Review Guide ──> Code Reviewer   ┐
             └─> Test Quality  ──┴─> Review Lead
```

The guide's report is passed to the parallel reviewers through `Task.context`,
and `Task.async_execution=True` makes the code and test reviewers run
concurrently while the lead waits for both.

## Install

```bash
uv sync
```

## Usage

```python
from review_agent_core import GitConfig, LLMConfig, review_all, review_code

git = GitConfig(
    repo_path="/path/to/repo",   # optional; defaults to the current directory
    base_ref="main",             # optional; defaults to main, then master
    head_ref="feature",          # optional; defaults to HEAD
)

llm = LLMConfig(
    model="openai/gpt-4o-mini",          # any OpenAI-compatible model
    base_url="http://localhost:8000/v1", # optional; defaults to api.openai.com
    api_key="...",
)

report = review_code(git=git, llm=llm)  # Markdown
full = review_all(git=git, llm=llm)     # guide -> (code, tests) -> lead pipeline
```

## Layout

```
src/review_agent_core/
  config.py           # GitConfig, LLMConfig (pydantic)
  git_repo.py         # read-only git access + BranchDiffScope
  tools/git_tools.py  # 5 read-only CrewAI tools over git
  skills/*.md         # prompt modules: code_reviewer, test_quality, review_guide, review_lead
  crew.py             # build_llm / build_*_crew / build_review_pipeline_crew
  schemas.py          # Finding / ReviewerFindings / MergedReport + merge helpers
  render.py           # deterministic Markdown rendering
  api.py              # review_code / review_tests / review_guide / review_all
```

## Adding a new capability

Add a `skills/<name>.md` prompt module and a `build_<name>_crew(...)` factory in
`crew.py`, reusing `make_git_tools(repo, scope)`. Reviewers that produce findings
should set `output_pydantic=ReviewerFindings`.

## Tests

```bash
uv run pytest -q
```

## Notes

- Read-only: the agent never writes to the repository or publishes comments.
- The first review scope is "branch vs base" (merge-base diff). Other scopes
  (uncommitted changes, explicit range, single commit) can be added later as
  sibling implementations of the same surface.
- A CLI (`apps/review-agent-cli`) provides the `review-agent review` command.
- Optional observability: `enable_tracing()` prints a compact, colorized stderr
  trace attributed to each agent: phase banners, LLM requests (messages/tools/
  sampling), token usage with finish reason and duration, tool calls with
  duration, and failures. `print_usage_summary()` prints the totals plus a
  per-agent breakdown. Colors auto-disable when stderr is redirected.
