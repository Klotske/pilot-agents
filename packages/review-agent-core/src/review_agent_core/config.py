"""Configuration models for the review agent."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    """Settings for the LLM backing the agent (OpenAI-compatible endpoint)."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    context_window_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "The model's full context window in tokens. Required when history "
            "compaction is enabled; the CLI fills it from "
            "REVIEW_AGENT_CONTEXT_WINDOW. Also forwarded to CrewAI so its "
            "built-in handling does not fall back to a guessed window."
        ),
    )


class GitConfig(BaseModel):
    """Settings for the local git repository under review."""

    repo_path: str | None = None  # default: current working directory
    base_ref: str | None = None   # default: main, then master
    head_ref: str | None = None   # default: HEAD


class CompactionConfig(BaseModel):
    """Proactive compaction of an agent's message history before an LLM call.

    CrewAI's built-in compaction only reacts to a provider error and replaces
    every non-system message with one summary, which destroys the task prompt
    and the structured-output schema. This compactor runs before the call and
    keeps the assistant/tool pairing, the task prompt, and the last steps
    intact (see :mod:`review_agent_core.compaction`).
    """

    enabled: bool = True
    trigger_ratio: float = Field(
        default=0.75, gt=0, le=1,
        description="Compact once the estimate exceeds this share of the window.",
    )
    target_ratio: float = Field(
        default=0.5, gt=0, le=1,
        description="Aim for this share of the window after compacting.",
    )
    keep_last_steps: int = Field(
        default=2, ge=0,
        description="Number of trailing steps (assistant turn + tool results) kept verbatim.",
    )
    tool_result_head: int = Field(default=1_200, gt=0, description="Chars kept from a tool result's start.")
    tool_result_tail: int = Field(default=400, ge=0, description="Chars kept from a tool result's end.")
    step_head: int = Field(default=600, gt=0, description="Max chars kept from an assistant reasoning message.")
    min_chars_per_message: int = Field(
        default=300, gt=0,
        description="Lower bound for a shortened message, so nothing is reduced to nothing.",
    )
    chars_per_token: float = Field(
        default=3.5, gt=0,
        description="Fallback estimate used when tiktoken is unavailable.",
    )
    tiktoken_encoding: str | None = Field(
        default=None,
        description="Override the tiktoken encoding (defaults derive from the model name).",
    )
    structure_aware: bool = Field(
        default=True,
        description="Keep diff file/hunk headers and code declarations when shortening.",
    )

    @model_validator(mode="after")
    def _check_ratios(self) -> CompactionConfig:
        if self.target_ratio >= self.trigger_ratio:
            raise ValueError("target_ratio must be lower than trigger_ratio")
        return self
