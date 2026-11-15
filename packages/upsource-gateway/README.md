# upsource-gateway

A lightweight, synchronous client for the [Upsource](https://www.jetbrains.com/upsource/)
RPC API. Ships full type coverage: 370 message DTOs as `TypedDict` and 33 enums as
`StrEnum`/`IntEnum`, generated from the official API descriptor.

## Install

```bash
uv sync
```

## Quick start

```python
from upsource_gateway import UpsourceClient, UpsourceError

client = UpsourceClient(
    base_url="https://review-pilot.ascon.net",
    login="user",
    password="secret",
    project_id="demo-project",
)

review = client.get_review("CR-123")          # ReviewDescriptorDTO
diff = client.get_review_diff("CR-123")       # ReviewSummaryChangesResponseDTO
comments = client.get_review_comments("CR-123")  # list[CommentDTO]
text = client.read_repository_file("/src/main.py", "abc123")  # str
reviews = client.get_coverage_report("abc123")  # list[ShortReviewInfoDTO]
discussion = client.publish_review_comment("CR-123", "LGTM")
```

All high-level methods accept an optional `project_id=` to override the client default.

## Raw RPC access

```python
# any method, JSON in / JSON out
result = client.call("getReviews", {"projectId": "demo-project", "limit": 50, "query": "..."})
# read-only methods can use GET
result = client.call("getReviews", {...}, use_get=True)
```

## Errors

- `UpsourceRpcError` — structured `{"error": {"code", "message"}}` from Upsource,
  with `.code`, `.name`, `.message`, `.http_status`. Subclasses:
  `AuthenticationError`, `AccessDeniedError`, `NotFoundError`, `InvalidRequestError`.
- `UpsourceTransportError` — network/timeout/HTTP-level failures.
- Both derive from `UpsourceError`.

Read-only GET calls are retried (up to `max_retries=3`, exponential backoff) on
transient failures; POST calls are never retried.

## Regenerating models

`src/upsource_gateway/models.py` is generated from `tools/api.json` (a snapshot of
the Upsource API descriptor) and committed to the repo.

```bash
python tools/codegen.py                     # regenerate from tools/api.json
python tools/codegen.py --input /path/to/fresh/api.json   # from a refreshed dump
```

## Tests

```bash
uv run pytest
```
