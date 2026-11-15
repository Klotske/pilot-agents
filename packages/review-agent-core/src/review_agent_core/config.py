"""Configuration models for the review agent."""

from __future__ import annotations

from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Settings for the LLM backing the agent (OpenAI-compatible endpoint)."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None


class GitConfig(BaseModel):
    """Settings for the local git repository under review."""

    repo_path: str | None = None  # default: current working directory
    base_ref: str | None = None   # default: main, then master
    head_ref: str | None = None   # default: HEAD
