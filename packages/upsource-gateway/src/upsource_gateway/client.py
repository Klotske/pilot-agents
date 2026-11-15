"""Synchronous Upsource RPC API client."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from . import models
from .errors import UpsourceError, UpsourceTransportError, build_error

_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


class _RetryableError(Exception):
    """Internal marker for a transient failure that may be retried on GET."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class UpsourceClient:
    """A lightweight client for the Upsource RPC API.

    Args:
        base_url: Upsource server root, e.g. ``https://review.example.com``
            (the ``/~rpc`` prefix is added automatically).
        login: Optional Basic-auth login. When omitted, requests run as guest.
        password: Optional Basic-auth password.
        project_id: Default project used by the high-level helpers; can be
            overridden per call.
        timeout: Per-request timeout in seconds.
        max_retries: Extra attempts for idempotent GET requests on transient
            failures (transport errors, 429/5xx).
        retry_backoff: Initial backoff in seconds (doubles on each retry).
    """

    def __init__(
        self,
        base_url: str,
        login: str | None = None,
        password: str | None = None,
        project_id: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        auth = httpx.BasicAuth(login, password) if login is not None else None
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            auth=auth,
            transport=transport,
        )

    # ------------------------------------------------------------------ transport

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "UpsourceClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def call(self, method: str, params: dict | None = None, *, use_get: bool = False) -> Any:
        """Invoke any RPC method and return its ``result`` payload.

        ``method`` is the bare RPC name (e.g. ``"getReviewDetails"``); ``params``
        is the JSON request object. Set ``use_get=True`` for read-only calls.
        """
        attempts = (self.max_retries + 1) if use_get else 1
        last: _RetryableError | None = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
            try:
                return self._do_request(method, params, use_get)
            except _RetryableError as exc:
                last = exc
        assert last is not None
        raise UpsourceTransportError(str(last)) from last.cause

    def _do_request(self, method: str, params: dict | None, use_get: bool) -> Any:
        url = f"/~rpc/{method}"
        try:
            if use_get:
                response = self._client.get(
                    url, params={"params": json.dumps(params or {})}
                )
            else:
                response = self._client.post(url, json=params)
        except httpx.TransportError as exc:
            raise _RetryableError(f"transport error: {exc}", cause=exc) from exc

        if response.status_code in _RETRYABLE_HTTP_STATUS:
            raise _RetryableError(f"HTTP {response.status_code}")

        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise UpsourceTransportError(
                f"non-JSON response (HTTP {response.status_code}): "
                f"{response.text[:200]!r}"
            )

        if isinstance(data, dict) and "error" in data:
            raise build_error(data["error"])

        if response.is_error:
            response.raise_for_status()

        if isinstance(data, dict):
            return data.get("result", data)
        return data

    # ------------------------------------------------------------------ helpers

    def _project(self, project_id: str | None) -> str:
        pid = project_id if project_id is not None else self.project_id
        if pid is None:
            raise UpsourceError(
                "project_id must be provided either at construction or per call"
            )
        return pid

    def _review_id(self, review_id: str, project_id: str | None) -> models.ReviewIdDTO:
        return {"projectId": self._project(project_id), "reviewId": review_id}

    # ----------------------------------------------------------- high-level API

    def get_review(
        self, review_id: str, project_id: str | None = None
    ) -> models.ReviewDescriptorDTO:
        """Return review details (title, state, participants, labels, ...)."""
        return self.call("getReviewDetails", {"reviewId": self._review_id(review_id, project_id)})

    def get_review_diff(
        self, review_id: str, project_id: str | None = None
    ) -> models.ReviewSummaryChangesResponseDTO:
        """Return the summary diff (sum of all revisions) of a review."""
        return self.call(
            "getReviewSummaryChanges", {"reviewId": self._review_id(review_id, project_id)}
        )

    def get_review_comments(
        self, review_id: str, project_id: str | None = None
    ) -> list[models.CommentDTO]:
        """Return a flat list of comments across all discussions in a review."""
        data = self.call(
            "getReviewSummaryDiscussions", {"reviewId": self._review_id(review_id, project_id)}
        )
        comments: list[models.CommentDTO] = []
        for item in data.get("discussions", []) or []:
            discussion = item.get("discussionInFile") or {}
            comments.extend(discussion.get("comments", []) or [])
        return comments

    def read_repository_file(
        self, path: str, revision: str, project_id: str | None = None
    ) -> str:
        """Return the text content of ``path`` at ``revision``.

        Raises :class:`UpsourceError` when no text is returned (e.g. the file is
        binary or too large to display).
        """
        request: models.FileInRevisionDTO = {
            "projectId": self._project(project_id),
            "revisionId": revision,
            "fileName": path,
        }
        data = self.call("getFileContent", request)
        content = data.get("fileContent") or {}
        text = content.get("text")
        if text is None:
            raise UpsourceError(f"no text content returned for {path!r} at {revision!r}")
        return text

    def get_coverage_report(
        self, revision: str, project_id: str | None = None
    ) -> list[models.ShortReviewInfoDTO]:
        """Return reviews that cover the given revision."""
        request: models.RevisionListDTO = {
            "projectId": self._project(project_id),
            "revisionId": [revision],
        }
        data = self.call("getRevisionReviewInfo", request)
        result: list[models.ShortReviewInfoDTO] = []
        for item in data.get("reviewInfo", []) or []:
            info = item.get("reviewInfo")
            if info:
                result.append(info)
        return result

    def publish_review_comment(
        self, review_id: str, comment: str, project_id: str | None = None
    ) -> models.DiscussionInFileDTO:
        """Post a top-level comment (a new discussion) on a review."""
        request: models.CreateDiscussionRequestDTO = {
            "projectId": self._project(project_id),
            "reviewId": self._review_id(review_id, project_id),
            "text": comment,
            "anchor": {},
        }
        return self.call("createDiscussion", request)
