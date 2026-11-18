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
validates the answer. The skills therefore end with an explicit output contract —
*one JSON object and nothing else, no Markdown, no code fences* — because a prose
answer fails validation: CrewAI then re-asks and/or runs a separate LLM
conversion pass (visible in the trace as a `[direct]` call), which costs tokens
and latency without improving the report.

Markdown is produced by code, not by a model: `review_agent_core.render` turns
the validated data into the final document deterministically, so the same
findings always render the same way. Prose survives only where it is authored on
purpose — the guide's report and the lead's `summary` field. If a model's answer
cannot be validated, the raw answer text is parsed and, failing that, printed
as-is instead of breaking the run.

**Gateway caveat.** CrewAI only skips the provider-level `response_format`
request when the agent has at least one tool (`effective_response_model = None if
self.original_tools else ...`). Without tools it sends a `json_schema`
response_format, which some OpenAI-compatible gateways reject with HTTP 400 —
that is why the lead also gets a tool — but only one cheap one (`list_files`)
and at most two iterations, while its skill forbids calling it outright. The
reports it merges already carry the diff, so a wider tool set just made it
re-fetch everything: in one run a single tool round ended without a final
answer, which cost an LLM conversion plus a full task retry (~26k tokens, ~60 s).
`max_iter` is the seatbelt: CrewAI then asks for a final answer instead of
dropping the task.

**Model ids.** CrewAI reads `prefix/model` as a provider route and falls back to
LiteLLM for prefixes it does not know (it is not a dependency here), so a
gateway id such as `selectel/qwen3.8-flash-next` fails before the request is
sent. `build_llm` wraps an unknown prefix as `openai/<id>`, which selects the
OpenAI-compatible provider and puts the id after the prefix into the request
body unchanged. Whether the vendor prefix belongs on the wire is the gateway's
call — its `/v1/models` listing is the source of truth; pass the bare model id
if the listing has no prefix.

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

## History compaction

A reviewer loops over git tools, and `get_diff`/`read_file` easily push a
message history past the model's context window. CrewAI's built-in handling
fires at 85% of the window (or when the provider rejects a call) and then
replaces every non-system message with a single summary — losing the task prompt
and the output schema, so the answer stops validating. This package compacts
**before** the call, first, and much earlier
(`compaction.py`, installed as an execution-scoped `before_llm_call` hook):

- all `system` messages and the first `user` message are never touched;
- the last `keep_last_steps` steps stay verbatim;
- `assistant`/`tool` pairing is preserved — messages are shortened, never
dropped, so providers keep accepting the request;
- identical tool results are replaced by a pointer to the first occurrence,
  on **every** call rather than only near the limit (reviewers tend to call
  `get_diff` twice — usually the biggest win);
- shortening is structure-aware: `diff --git`/`@@` headers and code
declarations survive while file bodies are cut;
- it is idempotent: already-shortened content is not shortened again.

Token counts come from `tiktoken` (encoding picked from the model name, or
`CompactionConfig.tiktoken_encoding`); if the BPE cannot be loaded offline the
estimator degrades to `chars / chars_per_token` and the trace says so.

The window is **required**, never guessed — CrewAI would assume 4096 tokens for
an unknown gateway alias and compact far too early:

```bash
export REVIEW_AGENT_CONTEXT_WINDOW=262144   # the model's raw window in tokens
review-agent review --all
```

The CLI also accepts `--context-window N` and `--no-compact`. From Python pass
`LLMConfig(context_window_size=...)`; tuning lives in `CompactionConfig`
(`trigger_ratio=0.75`, `target_ratio=0.5`, budgets, `enabled=False` to opt out).
Every compaction prints a `[compact] 128k->61k tok (trigger 750k, target 500k),
dedup=2, truncated=9` trace line to stderr. When the estimate is close to the
trigger but nothing was cut, the compactor says why instead of staying silent:
`[compact] nothing cut: est 7160/4500 (159%), protected 7160 ≥ target 3000`.

`Agent.respect_context_window` stays `True` as a last-resort net for a
mis-estimated window; it never fires while the compactor keeps the history in
bounds.

## Layout

```
src/review_agent_core/
  config.py           # GitConfig, LLMConfig (pydantic)
  git_repo.py         # read-only git access + BranchDiffScope
  tools/git_tools.py  # 6 read-only CrewAI tools over git
  skills/*.md         # prompt modules: code_reviewer, test_quality, review_guide, review_lead
  crew.py             # build_llm / build_*_crew / build_review_pipeline_crew
  compaction.py       # proactive history compaction (before_llm_call hook)
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
- `REVIEW_AGENT_CONTEXT_WINDOW` (tokens) is required by the CLI unless
  `--no-compact` is passed; the library requires
  `LLMConfig.context_window_size` when compaction is enabled.
- Optional observability: `enable_tracing()` prints a compact, colorized stderr
  trace attributed to each agent: phase banners, LLM requests (messages/tools/
  sampling), token usage with finish reason and duration, tool calls with
  duration, and failures. `print_usage_summary()` prints the totals plus a
  per-agent breakdown. Colors auto-disable when stderr is redirected.
- Context-window telemetry: every LLM answer carries `ctx <prompt>/<window>
  <pct>%` with a marker — `⚠trigger` (compaction threshold), `⚠crewai` (CrewAI's
  own 85% preflight), `⚠window` (over the window). Requests show the previous
  call's exact `last actual` tokens, so an estimate bias is visible, and the
  summary reports the peak per agent. Calls made without an agent are labelled
  by what they are, not hidden: `[direct]` is CrewAI's structured-output
  conversion, `[direct/recovery]` is the forced final answer after `max_iter` was
  used up (it resumes the agent history, so it carries `tool` messages).
- Usage numbers come from the per-call events, not from `crew.usage_metrics`:
  on a multi-task pipeline that rollup repeats the same calls once per task. The
  rollup is still compared against the events and a note is printed when the two
  disagree by more than 5%.
