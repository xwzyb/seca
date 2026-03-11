"""
seca — Rule evolution engine (Strange Loop core)
==================================================
Pure data layer for rule mutations. The agent does all reasoning — this module
just handles persistence and logging.

Actions: refine / create / merge / deprecate / update-meta.
Every mutation logged to evolution_log.json.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rules import CognitiveRule, RuleEngine


class MetaRule:
    __slots__ = ("id", "name", "condition", "action", "threshold")

    def __init__(self, name: str, condition: str, action: str, threshold: float = 0.5, id: str | None = None):
        self.id = id or uuid.uuid4().hex[:12]
        self.name = name
        self.condition = condition
        self.action = action
        self.threshold = threshold

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "condition": self.condition, "action": self.action, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetaRule":
        return cls(**d)


DEFAULT_META_RULES: list[dict[str, Any]] = [
    {"id": "meta_low_success", "name": "Refine on low success rate", "condition": "When a rule's success rate drops below threshold after >=3 uses", "action": "refine_rule", "threshold": 0.4},
    {"id": "meta_rule_gap", "name": "Create rule on gap", "condition": "When no rules match a situation", "action": "create_rule", "threshold": 0.0},
    {"id": "meta_high_similarity", "name": "Merge similar rules", "condition": "When two+ rules have overlapping conditions and similar strategies", "action": "merge_rules", "threshold": 0.8},
    {"id": "meta_deprecate_unused", "name": "Deprecate stale rules", "condition": "When a rule has been used >5 times with success rate below threshold", "action": "deprecate_rule", "threshold": 0.2},
]


class EvolutionEntry:
    def __init__(self, action: str, target_rule_ids: list[str], reason: str,
                 before: Any = None, after: Any = None, trigger: str = "auto",
                 id: str | None = None, timestamp: str | None = None):
        self.id = id or uuid.uuid4().hex[:12]
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.action = action
        self.target_rule_ids = target_rule_ids
        self.before = before
        self.after = after
        self.reason = reason
        self.trigger = trigger

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "timestamp": self.timestamp, "action": self.action,
            "target_rule_ids": self.target_rule_ids, "before": self.before,
            "after": self.after, "reason": self.reason, "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvolutionEntry":
        return cls(**d)


class RuleEvolver:
    """Manages rule mutations and evolution logging. No LLM — agent does reasoning."""

    def __init__(self, rule_engine: RuleEngine, data_dir: Path) -> None:
        self.rule_engine = rule_engine
        rules_dir = data_dir / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = rules_dir / "evolution_log.json"
        self._meta_path = rules_dir / "meta_rules.json"
        self._log: list[EvolutionEntry] = []
        self._meta_rules: list[MetaRule] = []
        self._load()

    def _load(self) -> None:
        if self._log_path.exists():
            raw = json.loads(self._log_path.read_text(encoding="utf-8"))
            self._log = [EvolutionEntry.from_dict(e) for e in raw]
        if self._meta_path.exists():
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._meta_rules = [MetaRule.from_dict(m) for m in raw]
        else:
            self._meta_rules = [MetaRule.from_dict(m) for m in DEFAULT_META_RULES]
            self._save_meta()

    def _save_log(self) -> None:
        self._log_path.write_text(
            json.dumps([e.to_dict() for e in self._log], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps([m.to_dict() for m in self._meta_rules], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _record(self, entry: EvolutionEntry) -> None:
        self._log.append(entry)
        self._save_log()

    # --- core actions (all pure data, no LLM) ---

    def refine_rule(self, rule_id: str, condition: str, strategy: str,
                    confidence: float, reason: str) -> dict[str, Any]:
        """Update a rule with agent-provided new values."""
        rule = self.rule_engine.get_rule(rule_id)
        if rule is None:
            return {"status": "error", "message": f"Rule {rule_id} not found"}

        before = rule.to_dict()
        self.rule_engine.update_rule(rule_id, condition=condition, strategy=strategy, confidence=confidence)
        updated = self.rule_engine.get_rule(rule_id)
        after = updated.to_dict() if updated else None

        self._record(EvolutionEntry(
            action="refine", target_rule_ids=[rule_id],
            before=before, after=after, reason=reason,
        ))
        return {"status": "ok", "before": before, "after": after}

    def create_rule(self, name: str, condition: str, strategy: str,
                    confidence: float = 0.6, source_episode_id: str | None = None,
                    reason: str = "") -> dict[str, Any]:
        """Create a new rule with agent-provided values."""
        new_rule = CognitiveRule(
            name=name, condition=condition, strategy=strategy,
            confidence=confidence, created_by="self",
            source_episode=source_episode_id,
        )
        self.rule_engine.add_rule(new_rule)
        self._record(EvolutionEntry(
            action="create", target_rule_ids=[new_rule.id],
            after=new_rule.to_dict(), reason=reason or f"New rule: {name}",
        ))
        return {"status": "ok", "rule": new_rule.to_dict()}

    def merge_rules(self, rule_ids: list[str], name: str, condition: str,
                    strategy: str, confidence: float = 0.7,
                    reason: str = "") -> dict[str, Any]:
        """Merge rules: create a new one and deprecate the originals."""
        rules = [self.rule_engine.get_rule(rid) for rid in rule_ids]
        rules = [r for r in rules if r is not None]
        if len(rules) < 2:
            return {"status": "error", "message": "Need at least 2 valid rules to merge"}

        merged = CognitiveRule(
            name=name, condition=condition, strategy=strategy,
            confidence=confidence, created_by="self",
        )
        self.rule_engine.add_rule(merged)

        befores = {r.id: r.to_dict() for r in rules}
        for r in rules:
            self.rule_engine.deprecate_rule(r.id)

        self._record(EvolutionEntry(
            action="merge",
            target_rule_ids=[r.id for r in rules] + [merged.id],
            before=befores, after=merged.to_dict(),
            reason=reason or f"Merged {len(rules)} rules",
        ))
        return {"status": "ok", "merged_rule": merged.to_dict(), "deprecated": [r.id for r in rules]}

    def deprecate_rule(self, rule_id: str, reason: str) -> dict[str, Any]:
        rule = self.rule_engine.get_rule(rule_id)
        if rule is None:
            return {"status": "error", "message": f"Rule {rule_id} not found"}
        before = rule.to_dict()
        self.rule_engine.deprecate_rule(rule_id)
        self._record(EvolutionEntry(
            action="deprecate", target_rule_ids=[rule_id],
            before=before, reason=reason,
        ))
        return {"status": "ok", "deprecated": before}

    def update_meta_rules(self, meta_rules: list[dict[str, Any]], reason: str = "") -> dict[str, Any]:
        """Replace meta-rules with agent-provided new set."""
        before = [m.to_dict() for m in self._meta_rules]
        self._meta_rules = [MetaRule.from_dict(m) for m in meta_rules]
        self._save_meta()
        after = [m.to_dict() for m in self._meta_rules]
        self._record(EvolutionEntry(
            action="evolve_meta",
            target_rule_ids=[m.id for m in self._meta_rules],
            before={"meta_rules": before}, after={"meta_rules": after},
            reason=reason or "Meta-rule update",
        ))
        return {"status": "ok", "before": before, "after": after}

    @property
    def evolution_log(self) -> list[EvolutionEntry]:
        return list(self._log)

    @property
    def meta_rules(self) -> list[MetaRule]:
        return list(self._meta_rules)
