# review-agent-core

A [CrewAI](https://docs.crewai.com/)-based agent core that reviews changes in a
local git repository. Three review modes are provided, each as a single-agent
Crew over read-only git tools:

- **Code Reviewer** — finds potential bugs, risks, and complex areas.
- **Test Quality Reviewer** — analyzes tests, finds untested scenarios and weak tests.
- **Review Guide** — explains business logic, highlights invariants, prepares review comments.
- **Review Lead** (`review_all`) — merges the three reports into one deduplicated, ranked report.

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
full = review_all(git=git, llm=llm)     # three reviewers + Review Lead merge
```

## Layout

```
src/review_agent_core/
  config.py           # GitConfig, LLMConfig (pydantic)
  git_repo.py         # read-only git access + BranchDiffScope
  tools/git_tools.py  # 5 read-only CrewAI tools over git
  skills/*.md         # prompt modules: code_reviewer, test_quality, review_guide, review_lead
  crew.py             # build_llm / build_*_crew
  api.py              # review_code / review_tests / review_guide / review_all
```

## Adding a new capability

Add a `skills/<name>.md` prompt module and a `build_<name>_crew(...)` factory in
`crew.py`, reusing `make_git_tools(repo, scope)`.

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
