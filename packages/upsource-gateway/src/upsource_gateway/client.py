"""Synchronous Upsource RPC API client."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ._generated import UpsourceRpcMethods
from .errors import UpsourceError, UpsourceTransportError, build_error

_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


class _RetryableError(Exception):
    """Internal marker for a transient failure that may be retried on GET."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class UpsourceClient(UpsourceRpcMethods):
    """A lightweight client for the Upsource RPC API.

    The typed RPC methods (one per Upsource API method, in snake_case) are
    generated into :class:`UpsourceRpcMethods` and mixed in here.

    Args:
        base_url: Upsource server root, e.g. ``https://review.example.com``
            (the ``/~rpc`` prefix is added automatically).
        login: Optional Basic-auth login. When omitted, requests run as guest.
        password: Optional Basic-auth password.
        project_id: Default project used by the generated methods; can be
            overridden per call via ``project_id=``.
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
