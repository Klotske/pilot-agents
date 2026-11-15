"""Tests for the codegen and the generated models/client modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import get_type_hints

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SRC = PKG / "src"
TOOLS = PKG / "tools"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TOOLS))

import codegen  # noqa: E402
from upsource_gateway import models  # noqa: E402

API = TOOLS / "api.json"


def load_api() -> dict:
    return json.loads(API.read_text(encoding="utf-8"))


def test_every_message_and_enum_is_generated() -> None:
    data = load_api()
    for name in data["messages"]:
        assert getattr(models, name, None) is not None, f"missing message {name}"
    for name in data["enums"]:
        assert getattr(models, name, None) is not None, f"missing enum {name}"


def test_all_method_types_resolve() -> None:
    data = load_api()
    for service in data["services"].values():
        for method in service["methods"]:
            for kind in ("argumentType", "returnType"):
                t = codegen.strip_package(method[kind])
                assert (
                    t in codegen.SCALARS
                    or t in data["messages"]
                    or t in data["enums"]
                ), f"{method['name']}.{kind} -> {t}"


def test_generated_files_are_up_to_date() -> None:
    data = load_api()
    models_rendered = codegen.render_models(data["messages"], data["enums"])
    client_rendered = codegen.render_client(data, data["messages"], data["enums"])
    committed_models = (SRC / "upsource_gateway" / "models.py").read_text(encoding="utf-8")
    committed_client = (SRC / "upsource_gateway" / "_generated.py").read_text(encoding="utf-8")
    assert models_rendered == committed_models, "models.py is stale; run tools/codegen.py"
    assert client_rendered == committed_client, "_generated.py is stale; run tools/codegen.py"


def test_key_dto_shape() -> None:
    review = models.ReviewIdDTO(projectId="p", reviewId="r")
    assert review.projectId == "p"
    assert review.reviewId == "r"
    assert review.model_dump() == {"projectId": "p", "reviewId": "r"}


def test_keyword_field_names_use_aliases() -> None:
    assert "from_" in models.RangeMappingDTO.__annotations__
    assert "from_" in models.ReviewersGraphLinkDTO.__annotations__
    assert models.RangeMappingDTO.model_fields["from_"].alias == "from"
    assert models.RevisionStateEnum.None_.value == "None"


def test_cyclic_message_types_resolve() -> None:
    for name in (
        "FileAnnotationSectionDTO",
        "FileAnnotationResponseDTO",
        "FileInlineDiffResponseDTO",
    ):
        get_type_hints(getattr(models, name))


def test_enum_field_accepts_unknown_value_as_str() -> None:
    m = models.CloseReviewResponseDTO.model_validate({"syncResult": "TotallyNew"})
    assert m.syncResult == "TotallyNew"
    assert isinstance(m.syncResult, str)


def test_response_ignores_extra_fields() -> None:
    m = models.UserAgreementTextDTO.model_validate({"text": "x", "extra": 1})
    assert m.text == "x"
    assert not hasattr(m, "extra")


def test_request_is_strict() -> None:
    from pydantic import ValidationError

    try:
        models.ReviewIdDTO.model_validate({"projectId": "p", "reviewId": 123}, strict=True)
    except ValidationError:
        pass
    else:
        raise AssertionError("strict validation should reject int for str field")
