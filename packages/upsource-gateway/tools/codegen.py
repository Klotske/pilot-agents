#!/usr/bin/env python3
"""Generate pydantic models + typed RPC client from the Upsource API dump.

Reads a machine-readable API descriptor (``api.json``) and emits:

* ``src/upsource_gateway/models.py``   - one ``BaseModel`` per message/DTO,
  one ``StrEnum`` per enum (``IntEnum`` for ``RPCErrors``);
* ``src/upsource_gateway/_generated.py`` - an ``UpsourceRpcMethods`` mixin with
  one typed snake_case method per RPC method.

Usage::

    python tools/codegen.py [--input api.json]
                            [--models-out src/upsource_gateway/models.py]
                            [--client-out src/upsource_gateway/_generated.py]

The generated files are deterministic and committed to the repo; re-run this
script only after refreshing ``tools/api.json`` from a newer Upsource.
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "api.json"
DEFAULT_MODELS_OUT = HERE.parent / "src" / "upsource_gateway" / "models.py"
DEFAULT_CLIENT_OUT = HERE.parent / "src" / "upsource_gateway" / "_generated.py"

# Protobuf scalar -> Python type.
SCALARS = {
    "String": "str",
    "Int32": "int",
    "Int64": "int",
    "Bool": "bool",
    "Float": "float",
    "Double": "float",
    "Bytes": "bytes",
}

# Enums serialized on the wire as numeric codes (error codes) -> IntEnum;
# everything else is sent/received by name -> StrEnum.
INT_ENUMS = {"RPCErrors"}

# ---------------------------------------------------------------------------
# Request-shape and read-only overrides.
#
# ``api.json`` records each method's *argument type* but not the argument's
# *name*. For the vast majority of methods the request body is the argument DTO
# spread at the top level (``{"projectId": ..., "limit": ...}``). A few methods
# wrap the DTO under a named key instead. Add such methods here.
#
# method name -> wrapper key, e.g. getReviewDetails sends
# ``{"reviewId": {"projectId": ..., "reviewId": ...}}``.
WRAPPED_ARGS: dict[str, str] = {
    "getReviewDetails": "reviewId",
}

# Read-only methods are sent as GET and retried on transient failures. By
# default this is derived from the method name; use these sets to override.
READ_ONLY_PREFIXES = ("get", "find", "query", "list", "search", "check", "has", "is")
READ_ONLY_OVERRIDES: set[str] = set()  # force GET even if the name says otherwise
WRITE_OVERRIDES: set[str] = set()  # force POST even if the name looks read-only


def strip_package(name: str) -> str:
    """Drop the ``messages.`` / ``enums.`` prefix used at the method level."""
    return name.split(".", 1)[1] if "." in name else name


def sanitize_ident(name: str) -> str:
    """Make a name a valid Python identifier (keywords get a trailing ``_``)."""
    return name + "_" if keyword.iskeyword(name) else name


def to_snake(name: str) -> str:
    """``getReviewDetails`` -> ``get_review_details``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def is_read_only(method_name: str) -> bool:
    if method_name in READ_ONLY_OVERRIDES:
        return True
    if method_name in WRITE_OVERRIDES:
        return False
    return method_name.startswith(READ_ONLY_PREFIXES)


def clean_desc(s: str | None) -> str:
    return " ".join((s or "").split())


def load(api_path: Path) -> dict:
    return json.loads(api_path.read_text(encoding="utf-8"))


def validate(data: dict, messages: dict, enums: dict) -> None:
    """Fail the build if any RPC method references an unknown type."""
    problems = []
    for service in data.get("services", {}).values():
        for method in service.get("methods", []):
            for kind in ("argumentType", "returnType"):
                t = strip_package(method[kind])
                if t not in SCALARS and t not in messages and t not in enums:
                    problems.append((method["name"], kind, t))
    if problems:
        for name, kind, t in problems:
            print(f"unresolved {kind} for method {name}: {t}", file=sys.stderr)
        raise SystemExit(1)

    # Method names must stay unique after snake_case conversion.
    seen: dict[str, str] = {}
    for service in data.get("services", {}).values():
        for method in service.get("methods", []):
            snake = to_snake(method["name"])
            if snake in seen:
                raise SystemExit(
                    f"method name collision: {seen[snake]} vs {method['name']} -> {snake}"
                )
            seen[snake] = method["name"]


# ---------------------------------------------------------------- models


def field_type(f: dict, enums: dict) -> str:
    """Annotation text for a pydantic model field."""
    t = f["type"]
    if t in SCALARS:
        base = SCALARS[t]
    elif t in enums:
        # Lenient enum: known values validate to the enum, unknown ones stay str.
        base = f"{t} | str"
    else:
        base = t  # message forward reference
    if f["label"] == "repeated":
        return f"list[{base}]"
    return base


def render_models(messages: dict, enums: dict) -> str:
    out: list[str] = []
    A = out.append

    A("# AUTO-GENERATED FILE - DO NOT EDIT.")
    A("# Generated by tools/codegen.py from tools/api.json.")
    A(f"# {len(messages)} messages (pydantic BaseModel), {len(enums)} enums.")
    A("")
    A("from __future__ import annotations")
    A("")
    A("import enum")
    A("from pydantic import BaseModel, ConfigDict, Field")
    A("")
    A("# ------------------------------------------------------------- enums")
    A("")

    for name in sorted(enums):
        e = enums[name]
        desc = clean_desc(e.get("description"))
        if desc:
            A(f"# {desc}")
        base = "enum.IntEnum" if name in INT_ENUMS else "enum.StrEnum"
        A(f"class {name}({base}):")
        values = e.get("values") or e.get("constants") or []
        if not values:
            A("    pass")
        for v in values:
            member = sanitize_ident(v["name"])
            if name in INT_ENUMS:
                A(f"    {member} = {v['number']}")
            else:
                A(f'    {member} = "{v["name"]}"')
        A("")

    A("# ---------------------------------------------------------- messages")
    A("")
    A("")
    A("class _DTO(BaseModel):")
    A('    model_config = ConfigDict(extra="ignore", populate_by_name=True)')
    A("")
    A("")

    for name in sorted(messages):
        m = messages[name]
        desc = clean_desc(m.get("description"))
        if desc:
            A(f"# {desc}")
        A(f"class {name}(_DTO):")
        fields = m.get("fields", [])
        if not fields:
            A("    pass")
        for f in fields:
            fname = f["name"]
            fd = clean_desc(f.get("description"))
            suffix = f"  # {fd}" if fd else ""
            ftype = field_type(f, enums)
            if keyword.iskeyword(fname):
                attr = sanitize_ident(fname)
                if f["label"] == "optional":
                    A(f'    {attr}: {ftype} | None = Field(default=None, alias="{fname}"){suffix}')
                else:
                    A(f'    {attr}: {ftype} = Field(alias="{fname}"){suffix}')
            elif f["label"] == "optional":
                A(f"    {fname}: {ftype} | None = None{suffix}")
            else:
                A(f"    {fname}: {ftype}{suffix}")
        A("")

    A("# Resolve forward references (including self-references).")
    A("_MODELS = [")
    for name in sorted(messages):
        A(f'    "{name}",')
    A("]")
    A("for _name in _MODELS:")
    A("    globals()[_name].model_rebuild()")
    A("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- client


def return_info(rt: str, enums: dict) -> tuple[str, str]:
    """Return ``(annotation, return_model_expr)`` for a method return type."""
    if rt == "VoidMessage":
        return "None", "None"
    if rt in SCALARS:
        return SCALARS[rt], "None"
    if rt in enums:
        return f"models.{rt} | str", "None"
    return f"models.{rt}", f"models.{rt}"


def param_type(f: dict, enums: dict) -> str:
    t = f["type"]
    if t in SCALARS:
        base = SCALARS[t]
    elif t in enums:
        base = f"models.{t} | str"
    else:
        base = f"models.{t}"
    if f["label"] == "repeated":
        return f"list[{base}]"
    return base


def render_client(data: dict, messages: dict, enums: dict) -> str:
    service = next(iter(data.get("services", {}).values()))
    methods = service.get("methods", [])

    out: list[str] = []
    A = out.append

    A("# AUTO-GENERATED FILE - DO NOT EDIT.")
    A("# Generated by tools/codegen.py from tools/api.json.")
    A(f"# {len(methods)} typed RPC methods.")
    A("")
    A("from __future__ import annotations")
    A("")
    A("from typing import Any")
    A("")
    A("from pydantic import BaseModel")
    A("")
    A("from . import models")
    A("")
    A("")
    A("class UpsourceRpcMethods:")
    A('    """Typed RPC methods mixed into :class:`UpsourceClient`."""')
    A("")
    A("    def _call_rpc(")
    A("        self,")
    A("        method: str,")
    A("        raw: dict[str, Any],")
    A("        request_model: type[BaseModel] | None,")
    A("        return_model: type[BaseModel] | None,")
    A("        *,")
    A("        read_only: bool,")
    A("        wrap_key: str | None = None,")
    A("    ) -> Any:")
    A("        \"\"\"Strictly validate the request, call the RPC, parse the response.\"\"\"")
    A("        if request_model is not None:")
    A("            request = request_model.model_validate(raw, strict=True)")
    A("            body: Any = request.model_dump(mode=\"json\", by_alias=True, exclude_none=True)")
    A("        else:")
    A("            body = raw")
    A("        if wrap_key is not None:")
    A("            body = {wrap_key: body}")
    A("        data = self.call(method, body, use_get=read_only)")
    A("        if return_model is None:")
    A("            return None")
    A("        return return_model.model_validate(data)")
    A("")

    for m in methods:
        rpc = m["name"]
        snake = to_snake(rpc)
        at = strip_package(m["argumentType"])
        rt = strip_package(m["returnType"])
        read_only = is_read_only(rpc)
        wrap_key = WRAPPED_ARGS.get(rpc)
        ret_ann, ret_expr = return_info(rt, enums)

        params: list[str] = []
        body_items: list[tuple[str, str]] = []
        needs_pid = False
        has_pid_param = False

        if at != "VoidMessage":
            for f in messages[at].get("fields", []):
                pname = sanitize_ident(to_snake(f["name"]))
                ptype = param_type(f, enums)

                # projectId is auto-injected from the client default (scalar) or
                # simply passed through as an optional list (repeated).
                if f["name"] == "projectId" and f["type"] == "String":
                    if f["label"] == "repeated":
                        params.append(f"{pname}: list[str] | None = None")
                        body_items.append(("projectId", pname))
                    else:
                        params.append(f"{pname}: str | None = None")
                        body_items.append(("projectId", "pid"))
                        needs_pid = True
                        has_pid_param = True
                    continue

                # reviewId: ReviewIdDTO is auto-built from a plain review_id + project id.
                if f["name"] == "reviewId" and f["type"] == "ReviewIdDTO":
                    if f["label"] == "optional":
                        params.append(f"{pname}: str | None = None")
                        body_items.append(
                            (
                                "reviewId",
                                f'{{"projectId": pid, "reviewId": {pname}}} if {pname} is not None else None',
                            )
                        )
                    else:
                        params.append(f"{pname}: str")
                        body_items.append(("reviewId", f'{{"projectId": pid, "reviewId": {pname}}}'))
                    needs_pid = True
                    continue

                if f["label"] == "optional":
                    params.append(f"{pname}: {ptype} | None = None")
                else:
                    params.append(f"{pname}: {ptype}")
                body_items.append((f["name"], pname))

        if params:
            sig = ", ".join(["self", "*"] + params)
        else:
            sig = "self"
        A(f"    def {snake}({sig}) -> {ret_ann}:")

        desc = clean_desc(m.get("description"))
        if desc:
            A(f'        """{desc}"""')

        if needs_pid:
            arg = "project_id" if has_pid_param else "None"
            A(f"        pid = self._project({arg})")
        A("        raw: dict[str, Any] = {")
        for key, expr in body_items:
            A(f'            "{key}": {expr},')
        A("        }")

        wrap_expr = f', wrap_key="{wrap_key}"' if wrap_key else ""
        req_expr = f"models.{at}" if at != "VoidMessage" else "None"
        A(
            f'        return self._call_rpc("{rpc}", raw, {req_expr}, {ret_expr},'
            f" read_only={read_only}{wrap_expr})"
        )
        A("")

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="api.json path")
    parser.add_argument(
        "--models-out", type=Path, default=DEFAULT_MODELS_OUT, help="models.py path"
    )
    parser.add_argument(
        "--client-out", type=Path, default=DEFAULT_CLIENT_OUT, help="_generated.py path"
    )
    args = parser.parse_args(argv)

    data = load(args.input)
    messages = data.get("messages", {})
    enums = data.get("enums", {})
    validate(data, messages, enums)

    models_text = render_models(messages, enums)
    args.models_out.write_text(models_text, encoding="utf-8")
    print(f"wrote {args.models_out} ({len(messages)} messages, {len(enums)} enums)")

    client_text = render_client(data, messages, enums)
    args.client_out.write_text(client_text, encoding="utf-8")
    n_methods = len(next(iter(data.get("services", {}).values())).get("methods", []))
    print(f"wrote {args.client_out} ({n_methods} methods)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
