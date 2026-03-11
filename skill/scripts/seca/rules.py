"""
seca — Cognitive rule system
==============================
Pure data layer. No LLM calls — the agent does all reasoning.
Rules are human-readable JSON: condition → strategy, with confidence + usage stats.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class CognitiveRule:
    """A single cognitive rule — the atom of strategic knowledge."""

    __slots__ = (
        "id", "name", "condition", "strategy", "confidence",
        "usage_count", "success_count", "created_by", "source_episode", "deprecated",
    )

    def __init__(
        self,
        name: str,
        condition: str,
        strategy: str,
        confidence: float = 0.8,
        usage_count: int = 0,
        success_count: int = 0,
        created_by: str = "human",
        source_episode: str | None = None,
        deprecated: bool = False,
        id: str | None = None,
    ):
        self.id = id or uuid.uuid4().hex[:12]
        self.name = name
        self.condition = condition
        self.strategy = strategy
        self.confidence = confidence
        self.usage_count = usage_count
        self.success_count = success_count
        self.created_by = created_by
        self.source_episode = source_episode
        self.deprecated = deprecated

    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "condition": self.condition,
            "strategy": self.strategy, "confidence": self.confidence,
            "usage_count": self.usage_count, "success_count": self.success_count,
            "created_by": self.created_by, "source_episode": self.source_episode,
            "deprecated": self.deprecated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CognitiveRule":
        return cls(**d)


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
    },
    {
        "id": "rule_clarification",
        "name": "Task Clarification",
        "condition": "When the task description is vague, ambiguous, or missing key details",
        "strategy": (
            "1. Identify what is unclear or ambiguous. "
            "2. Formulate specific clarifying questions. "
            "3. State assumptions if no clarification is available. "
            "4. Proceed with the most reasonable interpretation while flagging uncertainty."
        ),
        "confidence": 0.90,
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
    },
    {
        "id": "rule_verification",
        "name": "Self-Verification",
        "condition": "When the task requires a factual or quantitative answer",
        "strategy": (
            "1. Produce an initial answer. "
            "2. Check the answer by approaching from a different angle. "
            "3. If the two approaches disagree, investigate the discrepancy. "
            "4. Report confidence level alongside the final answer."
        ),
        "confidence": 0.82,
    },
    {
        "id": "rule_analogy",
        "name": "Analogical Transfer",
        "condition": "When the current problem resembles a previously solved problem in structure",
        "strategy": (
            "1. Retrieve similar past experiences from memory. "
            "2. Identify the structural mapping between old and new situations. "
            "3. Adapt the old strategy to the new context. "
            "4. Verify the adapted strategy makes sense in the new domain."
        ),
        "confidence": 0.75,
    },
]


class RuleEngine:
    """Loads and manages cognitive rules. Pure data — no LLM calls."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rules: list[CognitiveRule] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._rules = [CognitiveRule.from_dict(r) for r in raw]
        else:
            self._rules = [CognitiveRule.from_dict(r) for r in DEFAULT_RULES]
            self._save()

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([r.to_dict() for r in self._rules], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def all_rules(self, include_deprecated: bool = False) -> list[CognitiveRule]:
        if include_deprecated:
            return list(self._rules)
        return [r for r in self._rules if not r.deprecated]

    def get_rule(self, rule_id: str) -> CognitiveRule | None:
        for r in self._rules:
            if r.id == rule_id:
                return r
        return None

    def record_usage(self, rule_id: str, success: bool) -> bool:
        rule = self.get_rule(rule_id)
        if rule:
            rule.usage_count += 1
            if success:
                rule.success_count += 1
            self._save()
            return True
        return False

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

    def __len__(self) -> int:
        return len(self.all_rules())
