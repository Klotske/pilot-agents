"""Tests for UpsourceClient using httpx.MockTransport."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from upsource_gateway import (
    AuthenticationError,
    NotFoundError,
    UpsourceClient,
    UpsourceError,
    UpsourceRpcError,
    UpsourceTransportError,
    models,
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


# ------------------------------------------------------------------ transport


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


# --------------------------------------------------------- generated methods


def test_get_review_details_wrapped_get() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["params"] = json.loads(request.url.params.get("params"))
        return httpx.Response(
            200,
            json={
                "result": {
                    "reviewId": {"projectId": "demo", "reviewId": "CR-1"},
                    "title": "T",
                    "participants": [],
                    "state": "Open",
                    "branch": [],
                    "issue": [],
                    "createdAt": 1,
                    "updatedAt": 2,
                    "labels": [],
                }
            },
        )

    client = make_client(handler)
    result = client.get_review_details(review_id="CR-1")

    assert captured["method"] == "GET"
    assert captured["params"] == {"reviewId": {"projectId": "demo", "reviewId": "CR-1"}}
    assert isinstance(result, models.ReviewDescriptorDTO)
    assert result.reviewId.reviewId == "CR-1"
    assert result.state == models.ReviewStateEnum.Open


def test_close_review_nested_review_id_post() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["body"] = json_body(request)
        return httpx.Response(200, json={"result": {}})

    client = make_client(handler)
    result = client.close_review(review_id="CR-1", is_flagged=True)

    assert captured["method"] == "POST"
    assert captured["body"] == {
        "reviewId": {"projectId": "demo", "reviewId": "CR-1"},
        "isFlagged": True,
    }
    assert isinstance(result, models.CloseReviewResponseDTO)
    assert result.syncResult is None


def test_get_user_agreement_text_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"text": "hello"}})

    client = make_client(handler)
    result = client.get_user_agreement_text()

    assert isinstance(result, models.UserAgreementTextDTO)
    assert result.text == "hello"


def test_accept_user_agreement_void() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json_body(request)
        return httpx.Response(200, json={"result": {}})

    client = make_client(handler)
    result = client.accept_user_agreement()

    assert result is None
    assert captured["body"] == {}


def test_lenient_response_unknown_enum_and_extra_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"result": {"syncResult": "SomeNewValue", "extraField": 42}}
        )

    client = make_client(handler)
    result = client.close_review(review_id="CR-1", is_flagged=True)

    # Unknown enum value stays a raw str; extra fields are ignored.
    assert result.syncResult == "SomeNewValue"
    assert not hasattr(result, "extraField")


def test_strict_request_rejects_wrong_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {}})

    client = make_client(handler)
    with pytest.raises(ValidationError):
        client.close_review(review_id=123, is_flagged=True)  # type: ignore[arg-type]


def test_per_call_project_override() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = json.loads(request.url.params.get("params"))
        return httpx.Response(200, json={"result": {}})

    client = make_client(handler)
    try:
        client.get_review_details(review_id="CR-1", project_id="other")
    except ValidationError:
        pass  # empty result does not satisfy ReviewDescriptorDTO

    assert captured["params"] == {"reviewId": {"projectId": "other", "reviewId": "CR-1"}}


def test_missing_project_id_raises() -> None:
    client = make_client(lambda r: httpx.Response(200, json={}), project_id=None)
    with pytest.raises(UpsourceError):
        client.get_review_details(review_id="CR-1")
