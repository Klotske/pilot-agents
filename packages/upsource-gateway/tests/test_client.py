"""Tests for UpsourceClient using httpx.MockTransport."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from upsource_gateway import (
    AuthenticationError,
    NotFoundError,
    UpsourceClient,
    UpsourceError,
    UpsourceRpcError,
    UpsourceTransportError,
)

BASE = "https://example.test"


def make_client(
    handler,
    *,
    login: str | None = None,
    password: str | None = None,
    project_id: str | None = "demo",
    max_retries: int = 0,
    retry_backoff: float = 0.0,
) -> UpsourceClient:
    return UpsourceClient(
        base_url=BASE,
        login=login,
        password=password,
        project_id=project_id,
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def json_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode("utf-8"))


def test_post_url_and_auth_header() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json_body(request)
        return httpx.Response(200, json={"result": {"ok": True}})

    client = make_client(handler, login="user", password="pass")
    result = client.call("getReviewDetails", {"x": 1})

    assert result == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["path"] == "/~rpc/getReviewDetails"
    assert captured["body"] == {"x": 1}
    expected = "Basic " + base64.b64encode(b"user:pass").decode()
    assert captured["auth"] == expected


def test_get_uses_query_params() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["params"] = request.url.params.get("params")
        return httpx.Response(200, json={"result": {"n": 1}})

    client = make_client(handler)
    client.call("getRevisionsList", {"projectId": "demo"}, use_get=True)

    assert captured["method"] == "GET"
    assert json.loads(captured["params"]) == {"projectId": "demo"}


def test_rpc_error_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": {"code": 106, "message": "not found"}}
        )

    client = make_client(handler)
    with pytest.raises(NotFoundError) as exc_info:
        client.call("getReviewDetails", {})

    err = exc_info.value
    assert err.code == 106
    assert err.name == "NotFound"
    assert err.http_status == 404
    assert isinstance(err, UpsourceRpcError)


def test_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 101, "message": "denied"}})

    client = make_client(handler)
    with pytest.raises(AuthenticationError):
        client.call("getReviewDetails", {})


def test_retry_on_transient_get() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"result": {"ok": True}})

    client = make_client(handler, max_retries=2, retry_backoff=0.0)
    result = client.call("getRevisionsList", {}, use_get=True)

    assert result == {"ok": True}
    assert attempts["n"] == 2


def test_post_is_not_retried() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500)

    client = make_client(handler, max_retries=3, retry_backoff=0.0)
    with pytest.raises(UpsourceTransportError):
        client.call("createReview", {})

    assert attempts["n"] == 1


def test_get_review() -> None:
    captured: dict[str, Any] = {}
    payload = {"reviewId": {"projectId": "demo", "reviewId": "CR-1"}, "title": "T"}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json_body(request)
        return httpx.Response(200, json={"result": payload})

    client = make_client(handler)
    result = client.get_review("CR-1")

    assert captured["body"] == {"reviewId": {"projectId": "demo", "reviewId": "CR-1"}}
    assert result == payload


def test_get_review_comments_flattens() -> None:
    payload = {
        "discussions": [
            {
                "fileName": "/a.py",
                "discussionInFile": {
                    "discussionId": "d1",
                    "comments": [
                        {"commentId": "c1", "text": "one"},
                        {"commentId": "c2", "text": "two"},
                    ],
                },
            },
            {
                "fileName": "/b.py",
                "discussionInFile": {
                    "discussionId": "d2",
                    "comments": [{"commentId": "c3", "text": "three"}],
                },
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": payload})

    client = make_client(handler)
    comments = client.get_review_comments("CR-1")

    assert [c["commentId"] for c in comments] == ["c1", "c2", "c3"]


def test_read_repository_file() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json_body(request)
        return httpx.Response(
            200, json={"result": {"fileContent": {"text": "print('hi')\n"}}}
        )

    client = make_client(handler)
    text = client.read_repository_file("/src/main.py", "abc123")

    assert text == "print('hi')\n"
    assert captured["body"] == {
        "projectId": "demo",
        "revisionId": "abc123",
        "fileName": "/src/main.py",
    }


def test_read_repository_file_no_text_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"contentType": {}}})

    client = make_client(handler)
    with pytest.raises(UpsourceError):
        client.read_repository_file("/binary.bin", "abc123")


def test_get_coverage_report() -> None:
    payload = {
        "reviewInfo": [
            {"reviewInfo": {"reviewId": {"projectId": "demo", "reviewId": "CR-1"}}},
            {"reviewInfo": None},
            {"reviewInfo": {"reviewId": {"projectId": "demo", "reviewId": "CR-2"}}},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": payload})

    client = make_client(handler)
    reviews = client.get_coverage_report("abc123")

    assert [r["reviewId"]["reviewId"] for r in reviews] == ["CR-1", "CR-2"]


def test_publish_review_comment() -> None:
    captured: dict[str, Any] = {}
    payload = {"discussionId": "d1"}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json_body(request)
        return httpx.Response(200, json={"result": payload})

    client = make_client(handler)
    result = client.publish_review_comment("CR-1", "LGTM")

    assert captured["body"] == {
        "projectId": "demo",
        "reviewId": {"projectId": "demo", "reviewId": "CR-1"},
        "text": "LGTM",
        "anchor": {},
    }
    assert result == payload


def test_missing_project_id_raises() -> None:
    client = make_client(lambda r: httpx.Response(200, json={}), project_id=None)
    with pytest.raises(UpsourceError):
        client.get_review("CR-1")


def test_per_call_project_override() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json_body(request)
        return httpx.Response(200, json={"result": {}})

    client = make_client(handler)
    client.get_review("CR-1", project_id="other")

    assert captured["body"]["reviewId"] == {"projectId": "other", "reviewId": "CR-1"}
