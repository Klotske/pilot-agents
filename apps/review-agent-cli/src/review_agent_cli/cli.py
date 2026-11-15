"""Command-line interface for the git review agent."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from dotenv import load_dotenv
from review_agent_core import (
    GitConfig,
    LLMConfig,
    enable_tracing,
    print_usage_summary,
    review_all,
    review_code,
    review_guide,
    review_tests,
)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-agent",
        description="Review changes in a local git repository.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="Review a branch/ref against a base ref")
    review.add_argument(
        "branch", nargs="?", default=None,
        help="Branch/ref to review (default: HEAD)",
    )
    review.add_argument(
        "--base", default=None,
        help="Base ref to diff against (default: main, then master)",
    )
    review.add_argument(
        "--repo", default=None,
        help="Path to the git repository (default: current directory)",
    )
    review.add_argument("--code", action="store_true", help="Run the Code Reviewer")
    review.add_argument("--tests", action="store_true", help="Run the Test Quality Reviewer")
    review.add_argument("--guide", action="store_true", help="Run the Review Guide")
    review.add_argument("--all", action="store_true", help="Run all review modes (default)")
    review.add_argument(
        "--verbose", action="store_true",
        help="Also show CrewAI's full rich console output",
    )
    return parser


def _llm_config() -> LLMConfig:
    model = _env("LLM_MODEL")
    if not model:
        raise SystemExit("missing required environment variable: LLM_MODEL")
    temperature = _env("LLM_TEMPERATURE")
    return LLMConfig(
        model=model,
        base_url=_env("LLM_BASE_URL"),
        api_key=_env("LLM_API_KEY"),
        temperature=float(temperature) if temperature else None,
    )


def _git_config(args: argparse.Namespace) -> GitConfig:
    return GitConfig(
        repo_path=args.repo or _env("GIT_REPO"),
        base_ref=args.base or _env("GIT_BASE"),
        head_ref=args.branch,
    )


def _run(args: argparse.Namespace) -> int:
    git = _git_config(args)
    llm = _llm_config()
    verbose = args.verbose

    selected = [
        mode
        for mode, flag in (("code", args.code), ("tests", args.tests), ("guide", args.guide))
        if flag
    ]
    if args.all or not selected:
        print(review_all(git=git, llm=llm, verbose=verbose))
    else:
        for mode in selected:
            if mode == "code":
                print(review_code(git=git, llm=llm, verbose=verbose))
            elif mode == "tests":
                print(review_tests(git=git, llm=llm, verbose=verbose))
            else:
                print(review_guide(git=git, llm=llm, verbose=verbose))
            print()

    print_usage_summary()
    return 0


def _configure_io() -> None:
    """Make stdout/stderr tolerate non-ASCII output on legacy code pages.

    Windows consoles can be cp1251/cp866, where characters like ``Σ``, ``…`` or
    the emoji CrewAI prints would raise ``UnicodeEncodeError`` when the streams
    are redirected. Forcing UTF-8 with ``errors="replace"`` keeps reports and
    the trace flowing instead of crashing or silently dropping lines.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - best effort only
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_io()
    load_dotenv()
    enable_tracing()
    args = build_parser().parse_args(argv)
    return _run(args)
