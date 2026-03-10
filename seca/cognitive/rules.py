"""
seca.cognitive.rules — Cognitive rule system
=============================================

The **heart** of SECA.  Every piece of strategic knowledge is encoded as a
``CognitiveRule`` — a structured, human-readable JSON object that tells the
agent *when* to do *what*.

The ``RuleEngine`` loads rules from disk, matches them against the current
situation (via an LLM call), executes the selected strategy, and records
usage statistics.

Rules file: ``data/rules/cognitive_rules.json``
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from seca.foundation.llm import generate


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CognitiveRule(BaseModel):
    """A single cognitive rule — the atom of the agent's strategic knowledge."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = Field(description="Short human-readable rule name")
    condition: str = Field(description="Natural-language description of WHEN this rule triggers")
    strategy: str = Field(description="Natural-language description of WHAT to do")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How confident the agent is in this rule (0–1)",
    )
    usage_count: int = Field(default=0, description="How many times the rule has been invoked")
    success_count: int = Field(default=0, description="How many times it led to a positive outcome")
    created_by: Literal["human", "self"] = Field(
        default="human",
        description="Who authored this rule — a human designer or the agent itself",
    )
    source_episode: str | None = Field(
        default=None,
        description="ID of the episodic memory entry that inspired this rule (if created_by == 'self')",
    )
    deprecated: bool = Field(default=False, description="Soft-deleted flag")

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class RuleMatchResult(BaseModel):
    """Outcome of matching rules against a situation."""

    matched_rules: list[CognitiveRule]
    rule_gap: bool = Field(
        default=False,
        description="True when no rules matched — signals a learning opportunity",
    )
    match_reasoning: str = Field(default="", description="LLM's explanation of why rules matched")


# ---------------------------------------------------------------------------
# Default rules (shipped with SECA)
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "id": "rule_contradictions",
        "name": "Contradiction Resolution",
        "condition": "When the input contains contradictory or conflicting information",
        "strategy": (
            "1. List all contradictory points explicitly. "
            "2. For each contradiction, evaluate the source reliability. "
            "3. Propose the most likely resolution with reasoning. "
            "4. Note remaining uncertainty."
        ),
        "confidence": 0.85,
        "usage_count": 0,
        "success_count": 0,
        "created_by": "human",
        "source_episode": None,
        "deprecated": False,
    },
    {
        "id": "rule_clarification",
        "name": "Task Clarification",
        "condition": "When the task description is vague, ambiguous, or missing key details",
        "strategy": (
            "1. Identify what is unclear or ambiguous. "
            "2. Formulate specific clarifying questions. "
            "3. State the assumptions you would make if no clarification is available. "
            "4. Proceed with the most reasonable interpretation while flagging uncertainty."
        ),
        "confidence": 0.90,
        "usage_count": 0,
        "success_count": 0,
        "created_by": "human",
        "source_episode": None,
        "deprecated": False,
    },
    {
        "id": "rule_decomposition",
        "name": "Complex Task Decomposition",
        "condition": "When the task involves multiple distinct sub-problems or steps",
        "strategy": (
            "1. Break the task into independent sub-tasks. "
            "2. Identify dependencies between sub-tasks. "
            "3. Solve each sub-task in dependency order. "
            "4. Integrate partial results into a coherent final answer."
        ),
        "confidence": 0.88,
        "usage_count": 0,
        "success_count": 0,
        "created_by": "human",
        "source_episode": None,
        "deprecated": False,
    },
    {
        "id": "rule_verification",
        "name": "Self-Verification",
        "condition": "When the task requires a factual or quantitative answer",
        "strategy": (
            "1. Produce an initial answer. "
            "2. Check the answer by approaching the problem from a different angle. "
            "3. If the two approaches disagree, investigate the discrepancy. "
            "4. Report confidence level alongside the final answer."
        ),
        "confidence": 0.82,
        "usage_count": 0,
        "success_count": 0,
        "created_by": "human",
        "source_episode": None,
        "deprecated": False,
    },
    {
        "id": "rule_analogy",
        "name": "Analogical Transfer",
        "condition": "When the current problem resembles a previously solved problem in structure",
        "strategy": (
            "1. Retrieve similar past experiences from episodic memory. "
            "2. Identify the structural mapping between old and new situations. "
            "3. Adapt the old strategy to the new context. "
            "4. Verify the adapted strategy makes sense in the new domain."
        ),
        "confidence": 0.75,
        "usage_count": 0,
        "success_count": 0,
        "created_by": "human",
        "source_episode": None,
        "deprecated": False,
    },
]


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

class RuleEngine:
    """Loads, matches, and manages cognitive rules."""

    _path: Path
    _rules: list[CognitiveRule]

    def __init__(self, path: Path | None = None) -> None:
        rules_dir = Path(__file__).resolve().parent.parent.parent / "data" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        self._path = path or (rules_dir / "cognitive_rules.json")
        self._rules = []
        self._load()

    # -- persistence --

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._rules = [CognitiveRule(**r) for r in raw]
        else:
            # First run — seed with defaults
            self._rules = [CognitiveRule(**r) for r in DEFAULT_RULES]
            self._save()

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                [r.model_dump() for r in self._rules],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- accessors --

    def all_rules(self, include_deprecated: bool = False) -> list[CognitiveRule]:
        if include_deprecated:
            return list(self._rules)
        return [r for r in self._rules if not r.deprecated]

    def get_rule(self, rule_id: str) -> CognitiveRule | None:
        for r in self._rules:
            if r.id == rule_id:
                return r
        return None

    # -- matching --

    async def match(self, situation: str) -> RuleMatchResult:
        """Use the LLM to decide which rules apply to *situation*."""
        active_rules = self.all_rules()
        if not active_rules:
            return RuleMatchResult(matched_rules=[], rule_gap=True, match_reasoning="No rules available.")

        rules_text = "\n".join(
            f"- [{r.id}] {r.name} (confidence={r.confidence:.2f})\n"
            f"  Condition: {r.condition}\n"
            f"  Strategy: {r.strategy}"
            for r in active_rules
        )

        prompt = (
            f"Given the following situation:\n\n{situation}\n\n"
            f"And the following cognitive rules:\n\n{rules_text}\n\n"
            "Which rules are relevant to this situation? "
            "Return a JSON object with:\n"
            '  "matched_ids": [list of rule ids that apply],\n'
            '  "reasoning": "why these rules match"\n'
            "If no rules match, return an empty list for matched_ids."
        )
        system = (
            "You are the rule-matching subsystem of a cognitive agent. "
            "Analyze the situation and select all applicable rules. "
            "Reply ONLY with valid JSON, no markdown fences."
        )

        raw = await generate(prompt, system=system)

        # Parse LLM response
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
            data = json.loads(cleaned)
            matched_ids: list[str] = data.get("matched_ids", [])
            reasoning: str = data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat the whole response as reasoning, no matches
            matched_ids = []
            reasoning = raw

        matched = [r for r in active_rules if r.id in matched_ids]
        return RuleMatchResult(
            matched_rules=matched,
            rule_gap=len(matched) == 0,
            match_reasoning=reasoning,
        )

    # -- mutation --

    def record_usage(self, rule_id: str, success: bool) -> None:
        """Increment usage (and optionally success) counters for a rule."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.usage_count += 1
            if success:
                rule.success_count += 1
            self._save()

    def add_rule(self, rule: CognitiveRule) -> None:
        self._rules.append(rule)
        self._save()

    def update_rule(self, rule_id: str, **updates: Any) -> CognitiveRule | None:
        rule = self.get_rule(rule_id)
        if rule is None:
            return None
        for k, v in updates.items():
            if hasattr(rule, k):
                setattr(rule, k, v)
        self._save()
        return rule

    def deprecate_rule(self, rule_id: str) -> bool:
        rule = self.get_rule(rule_id)
        if rule:
            rule.deprecated = True
            self._save()
            return True
        return False

    def remove_rule(self, rule_id: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.id != rule_id]
        if len(self._rules) < before:
            self._save()
            return True
        return False

    def __len__(self) -> int:
        return len(self.all_rules())

    def __repr__(self) -> str:
        return f"RuleEngine({len(self.all_rules())} active rules)"
