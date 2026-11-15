"""Tests for the codegen and the generated models module."""

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


def test_generated_file_is_up_to_date() -> None:
    data = load_api()
    rendered = codegen.render(data["messages"], data["enums"])
    committed = (SRC / "upsource_gateway" / "models.py").read_text(encoding="utf-8")
    assert rendered == committed, "models.py is stale; run tools/codegen.py"


def test_key_dtos_shape() -> None:
    review_hints = get_type_hints(models.ReviewIdDTO)
    file_hints = get_type_hints(models.FileInRevisionDTO)
    assert review_hints["projectId"] == str
    assert review_hints["reviewId"] == str
    assert file_hints["revisionId"] == str
    assert file_hints["fileName"] == str


def test_keyword_field_names_are_handled() -> None:
    assert "from" in models.RangeMappingDTO.__annotations__
    assert "from" in models.ReviewersGraphLinkDTO.__annotations__
    assert models.RevisionStateEnum.None_.value == "None"


def test_cyclic_message_types_resolve() -> None:
    for name in (
        "FileAnnotationResponseDTO",
        "FileAnnotationSectionDTO",
        "FileInlineDiffResponseDTO",
    ):
        get_type_hints(getattr(models, name))
