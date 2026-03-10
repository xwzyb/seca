"""
seca.cognitive.analogy — Analogy engine
========================================

Finds structurally similar past experiences in episodic memory, uses the LLM
to extract abstract patterns, and transfers strategies from old situations to
the current one.

Key function: ``find_analogies(current_situation) -> list[Analogy]``
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from seca.foundation.llm import generate
from seca.foundation.memory import EpisodicMemory, Episode


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class StructuralMapping(BaseModel):
    """Maps elements of a source situation to the target situation."""

    source_element: str
    target_element: str
    relationship: str = Field(description="What structural role this element plays")


class Analogy(BaseModel):
    """A discovered analogy between a past episode and the current situation."""

    source_episode: Episode
    similarity_score: float = Field(ge=0.0, le=1.0)
    structural_mappings: list[StructuralMapping] = Field(default_factory=list)
    abstract_pattern: str = Field(
        default="",
        description="The high-level abstract pattern shared by both situations",
    )
    transferred_strategy: str = Field(
        default="",
        description="The adapted strategy for the current situation",
    )


# ---------------------------------------------------------------------------
# Analogy Engine
# ---------------------------------------------------------------------------

class AnalogyEngine:
    """Retrieves and maps analogies from episodic memory."""

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self.memory = episodic_memory

    async def find_analogies(
        self,
        current_situation: str,
        top_k: int = 3,
    ) -> list[Analogy]:
        """Return the best analogies from episodic memory for *current_situation*.

        Steps:
        1. Retrieve candidate episodes via textual similarity.
        2. Use the LLM to evaluate structural similarity and build mappings.
        3. Transfer the old strategy to the new context.
        """
        candidates = self.memory.search(current_situation, top_k=top_k * 2)
        if not candidates:
            return []

        analogies: list[Analogy] = []
        for episode in candidates:
            analogy = await self._evaluate_analogy(current_situation, episode)
            if analogy and analogy.similarity_score >= 0.3:
                analogies.append(analogy)

        # Sort by similarity (descending) and keep top_k
        analogies.sort(key=lambda a: a.similarity_score, reverse=True)
        return analogies[:top_k]

    async def _evaluate_analogy(
        self,
        current_situation: str,
        episode: Episode,
    ) -> Analogy | None:
        """Use the LLM to judge structural similarity and build mappings."""
        prompt = (
            f"Current situation:\n{current_situation}\n\n"
            f"Past experience:\n"
            f"  Situation: {episode.situation}\n"
            f"  Strategy used: {episode.strategy_used}\n"
            f"  Outcome: {episode.outcome}\n"
            f"  Reflection: {episode.reflection}\n\n"
            "Analyze the structural similarity between these two situations.\n"
            "Return a JSON object with:\n"
            '  "similarity_score": float between 0 and 1,\n'
            '  "abstract_pattern": "the shared abstract pattern",\n'
            '  "structural_mappings": [\n'
            '    {"source_element": "...", "target_element": "...", "relationship": "..."}\n'
            "  ],\n"
            '  "transferred_strategy": "how the old strategy can be adapted to the new situation"\n'
            "\nReply ONLY with valid JSON, no markdown fences."
        )
        system = (
            "You are a structural analogy engine. You find deep structural "
            "similarities between situations, beyond surface-level resemblance. "
            "Reply ONLY with valid JSON."
        )

        raw = await generate(prompt, system=system)

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned.strip())

            mappings = [
                StructuralMapping(**m)
                for m in data.get("structural_mappings", [])
            ]

            return Analogy(
                source_episode=episode,
                similarity_score=float(data.get("similarity_score", 0.0)),
                structural_mappings=mappings,
                abstract_pattern=data.get("abstract_pattern", ""),
                transferred_strategy=data.get("transferred_strategy", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Could not parse LLM output — skip this candidate
            return None

    async def transfer_strategy(
        self,
        analogy: Analogy,
        current_situation: str,
    ) -> str:
        """Refine the transferred strategy with additional context."""
        if analogy.transferred_strategy:
            prompt = (
                f"Current situation:\n{current_situation}\n\n"
                f"Proposed strategy (transferred from analogy):\n{analogy.transferred_strategy}\n\n"
                f"Abstract pattern: {analogy.abstract_pattern}\n\n"
                "Refine this strategy to better fit the current situation. "
                "Be specific and actionable."
            )
            system = "You are a strategy adaptation engine. Make transferred strategies concrete and applicable."
            return await generate(prompt, system=system)
        return ""

    def to_context(self, analogies: list[Analogy]) -> list[dict[str, Any]]:
        """Serialize analogies for inclusion in reasoning context."""
        return [
            {
                "source_situation": a.source_episode.situation,
                "source_strategy": a.source_episode.strategy_used,
                "source_outcome": a.source_episode.outcome,
                "similarity_score": a.similarity_score,
                "abstract_pattern": a.abstract_pattern,
                "transferred_strategy": a.transferred_strategy,
                "mappings": [m.model_dump() for m in a.structural_mappings],
            }
            for a in analogies
        ]
