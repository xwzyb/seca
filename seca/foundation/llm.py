"""
seca.foundation.llm — Unified LLM interface (Amazon Bedrock)
==============================================================

Provides a single ``generate`` coroutine that calls Claude models via
**Amazon Bedrock** using ``boto3``.  Also retains OpenAI and direct-Anthropic
as fallback providers.

Provider auto-detection order:
  1. ``BEDROCK`` — if ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` is set (default)
  2. ``ANTHROPIC`` — if ``ANTHROPIC_API_KEY`` is set
  3. ``OPENAI`` — if ``OPENAI_API_KEY`` is set

AWS credentials are resolved by boto3's standard chain:
  env vars → ~/.aws/credentials → IAM role → instance profile
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMConfig(BaseModel):
    """Runtime configuration for the LLM layer."""

    provider: LLMProvider | None = Field(
        default=None,
        description="Force a specific provider. ``None`` means auto-detect.",
    )
    # Bedrock
    bedrock_model_id: str = Field(
        default="global.anthropic.claude-opus-4-6-v1",
        description="Bedrock model ID (cross-region inference profile or base ID)",
    )
    bedrock_region: str = Field(
        default="ap-northeast-1",
        description="AWS region for Bedrock calls",
    )
    # Direct Anthropic (fallback)
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model name (direct API)",
    )
    # OpenAI (fallback)
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    # Common
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    anthropic_version: str = Field(
        default="bedrock-2023-05-31",
        description="Anthropic API version for Bedrock",
    )


# ---------------------------------------------------------------------------
# Bedrock provider (primary)
# ---------------------------------------------------------------------------

async def _generate_bedrock(
    prompt: str,
    system: str,
    tools: list[dict[str, Any]] | None,
    config: LLMConfig,
) -> str:
    """Call Claude via Amazon Bedrock's Converse API (async via run_in_executor)."""
    import asyncio
    import boto3

    def _sync_call() -> str:
        region = os.environ.get(
            "AWS_REGION",
            os.environ.get("AWS_DEFAULT_REGION", config.bedrock_region),
        )
        client = boto3.client("bedrock-runtime", region_name=region)

        # Build messages
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        # Build kwargs for converse()
        kwargs: dict[str, Any] = dict(
            modelId=config.bedrock_model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": config.max_tokens,
                "temperature": config.temperature,
            },
        )

        if system:
            kwargs["system"] = [{"text": system}]

        if tools:
            # Convert tools to Bedrock Converse toolConfig format
            tool_specs = []
            for t in tools:
                tool_spec = _convert_tool_to_bedrock(t)
                if tool_spec:
                    tool_specs.append(tool_spec)
            if tool_specs:
                kwargs["toolConfig"] = {"tools": tool_specs}

        response = client.converse(**kwargs)

        # Parse response
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        parts: list[str] = []
        for block in content_blocks:
            if "text" in block:
                parts.append(block["text"])
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                parts.append(json.dumps(
                    {
                        "id": tool_use.get("toolUseId", ""),
                        "function": tool_use.get("name", ""),
                        "arguments": tool_use.get("input", {}),
                    },
                    ensure_ascii=False,
                ))

        return "\n".join(parts) if parts else ""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_call)


def _convert_tool_to_bedrock(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a tool definition to Bedrock Converse toolSpec format.

    Accepts both OpenAI-style and Anthropic-style tool definitions.
    """
    # OpenAI style: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    if "function" in tool:
        func = tool["function"]
        return {
            "toolSpec": {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "inputSchema": {
                    "json": func.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        }

    # Anthropic style: {"name": ..., "description": ..., "input_schema": ...}
    if "name" in tool and "input_schema" in tool:
        return {
            "toolSpec": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": {
                    "json": tool["input_schema"],
                },
            }
        }

    # Already Bedrock format
    if "toolSpec" in tool:
        return tool

    return None


# ---------------------------------------------------------------------------
# OpenAI provider (fallback)
# ---------------------------------------------------------------------------

async def _generate_openai(
    prompt: str,
    system: str,
    tools: list[dict[str, Any]] | None,
    config: LLMConfig,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = dict(
        model=config.openai_model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    if tools:
        kwargs["tools"] = tools

    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]

    if choice.message.tool_calls:
        return json.dumps(
            [
                {
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.message.tool_calls
            ],
            ensure_ascii=False,
        )

    return choice.message.content or ""


# ---------------------------------------------------------------------------
# Direct Anthropic provider (fallback)
# ---------------------------------------------------------------------------

async def _generate_anthropic(
    prompt: str,
    system: str,
    tools: list[dict[str, Any]] | None,
    config: LLMConfig,
) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    kwargs: dict[str, Any] = dict(
        model=config.anthropic_model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    response = await client.messages.create(**kwargs)

    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "tool_use":
            parts.append(json.dumps(
                {"id": block.id, "function": block.name, "arguments": block.input},
                ensure_ascii=False,
            ))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _detect_provider() -> LLMProvider:
    """Pick a provider based on available credentials.

    Priority: Bedrock (AWS) > Anthropic (direct) > OpenAI
    """
    # Bedrock: check for AWS region or explicit credentials
    if (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    ):
        return LLMProvider.BEDROCK

    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC

    if os.environ.get("OPENAI_API_KEY"):
        return LLMProvider.OPENAI

    # Default to Bedrock — boto3 can resolve credentials from instance profile
    return LLMProvider.BEDROCK


_default_config = LLMConfig()


def configure(
    provider: LLMProvider | str | None = None,
    model_id: str | None = None,
    region: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMConfig:
    """Update the default LLM configuration. Returns the new config.

    Examples
    --------
    >>> configure(provider="bedrock", model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
    >>> configure(provider="bedrock", model_id="us.anthropic.claude-opus-4-6-v1:0", region="us-west-2")
    >>> configure(provider="anthropic")  # fall back to direct API
    """
    global _default_config

    updates: dict[str, Any] = {}
    if provider is not None:
        updates["provider"] = LLMProvider(provider)
    if model_id is not None:
        updates["bedrock_model_id"] = model_id
        # Also set anthropic_model for fallback compatibility
        updates["anthropic_model"] = model_id
    if region is not None:
        updates["bedrock_region"] = region
    if temperature is not None:
        updates["temperature"] = temperature
    if max_tokens is not None:
        updates["max_tokens"] = max_tokens

    _default_config = _default_config.model_copy(update=updates)
    return _default_config


async def generate(
    prompt: str,
    system: str = "",
    tools: list[dict[str, Any]] | None = None,
    *,
    config: LLMConfig | None = None,
) -> str:
    """Generate a completion from the configured LLM provider.

    Parameters
    ----------
    prompt:
        The user message / prompt.
    system:
        An optional system message.
    tools:
        An optional list of tool definitions (any format — auto-converted).
    config:
        Override the default ``LLMConfig``.

    Returns
    -------
    str
        The model's textual response (or serialised tool calls).
    """
    cfg = config or _default_config
    provider = cfg.provider or _detect_provider()

    if provider == LLMProvider.BEDROCK:
        return await _generate_bedrock(prompt, system, tools, cfg)
    elif provider == LLMProvider.OPENAI:
        return await _generate_openai(prompt, system, tools, cfg)
    else:
        return await _generate_anthropic(prompt, system, tools, cfg)
