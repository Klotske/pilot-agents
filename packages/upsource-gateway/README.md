# upsource-gateway

A lightweight, synchronous client for the [Upsource](https://www.jetbrains.com/upsource/)
RPC API. Ships full type coverage: 370 message DTOs as pydantic `BaseModel`, 33
enums as `StrEnum`/`IntEnum`, and 175 typed RPC methods — all generated from the
official API descriptor.

## Install

```bash
uv sync
```

## Quick start

```python
from upsource_gateway import UpsourceClient

client = UpsourceClient(
    base_url="https://review-pilot.ascon.net",
    login="user",
    password="secret",
    project_id="demo-project",
)

review = client.get_review_details(review_id="CR-123")   # ReviewDescriptorDTO
diff = client.get_review_summary_changes(review_id="CR-123")  # ReviewSummaryChangesResponseDTO
comments = client.get_review_summary_discussions(review_id="CR-123")  # DiscussionsInFilesDTO
file = client.get_file_content(revision_id="abc123", file_name="/src/main.py")
client.close_review(review_id="CR-123", is_flagged=True)  # CloseReviewResponseDTO
```

Generated methods are keyword-only, snake_case, and return typed pydantic models
(`None` for methods with no return value). Methods whose request carries a
top-level `projectId` accept an optional `project_id=` that overrides the client
default; nested `ReviewIdDTO` arguments are built for you from a plain
`review_id=` + the resolved project id.

## Raw RPC access

```python
# any method, JSON in / JSON out
result = client.call("getReviews", {"projectId": "demo-project", "limit": 50, "query": "..."})
# read-only methods can use GET
result = client.call("getReviews", {...}, use_get=True)
```

## Validation

- **Responses are parsed leniently**: unknown fields are ignored, unknown enum
  values are kept as raw strings, scalar values are coerced.
- **Requests are validated strictly**: missing required fields and wrong scalar
  types raise `pydantic.ValidationError`.

Read-only methods (detected by name: `get*`, `find*`, `query*`, ...) are sent as
GET and retried (up to `max_retries=3`, exponential backoff) on transient
failures; write methods are POST and never retried.

## Errors

- `UpsourceRpcError` — structured `{"error": {"code", "message"}}` from Upsource,
  with `.code`, `.name`, `.message`, `.http_status`. Subclasses:
  `AuthenticationError`, `AccessDeniedError`, `NotFoundError`, `InvalidRequestError`.
- `UpsourceTransportError` — network/timeout/HTTP-level failures.
- Both derive from `UpsourceError`.

## Regenerating models & client

`src/upsource_gateway/models.py` and `src/upsource_gateway/_generated.py` are
generated from `tools/api.json` (a snapshot of the Upsource API descriptor) and
committed to the repo.

```bash
python tools/codegen.py                     # regenerate both files
python tools/codegen.py --input /path/to/fresh/api.json   # from a refreshed dump
```

A couple of overrides live at the top of `tools/codegen.py` (rather than in the
snapshot): `WRAPPED_ARGS` for the few methods that wrap their argument DTO under
a named key, and `READ_ONLY_OVERRIDES` / `WRITE_OVERRIDES` to pin the HTTP verb
of specific methods.

## Tests

```bash
uv run pytest
```
