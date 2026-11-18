"""Tests for the CLI argument parsing (no LLM calls)."""

from __future__ import annotations

from review_agent_cli.cli import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["review", "feature"])
    assert args.command == "review"
    assert args.branch == "feature"
    assert args.base is None
    assert not args.code and not args.tests and not args.guide and not args.all


def test_parser_mode_flags() -> None:
    args = build_parser().parse_args(["review", "--code", "--tests"])
    assert args.code and args.tests
    assert not args.guide and not args.all


def test_parser_all_flag() -> None:
    args = build_parser().parse_args(["review", "--all", "--code"])
    assert args.all and args.code


def test_parser_compaction_flags() -> None:
    args = build_parser().parse_args(["review"])
    assert args.context_window is None
    assert not args.no_compact

    args = build_parser().parse_args(
        ["review", "--context-window", "131072", "--no-compact"]
    )
    assert args.context_window == 131_072
    assert args.no_compact
