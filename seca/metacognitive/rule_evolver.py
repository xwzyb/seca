"""
seca.metacognitive.rule_evolver — Rule evolution engine (Strange Loop core)
===========================================================================

This is where the **strange loop** lives.  ``RuleEvolver`` can:

* refine existing rules based on feedback
* create brand-new rules from experience
* merge similar rules into more general ones
* deprecate underperforming rules
* **evolve its own meta-rules** — the rules that govern *when and how* to
  evolve other rules.  This self-referential loop is the computational
  analogue of Hofstadter's Strange Loop.

Every mutation is logged in ``data/rules/evolution_log.json``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from seca.foundation.llm import generate
from seca.cognitive.rules import CognitiveRule, RuleEngine
from seca.metacognitive.monitor import MonitorReport, SuggestedAction

console = Console()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EvolutionEntry(BaseModel):
    """One logged mutation in the rule base."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = Field(description="refine | create | merge | deprecate | evolve_meta")
    target_rule_ids: list[str] = Field(default_factory=list)
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    reason: str = ""
    trigger: str = Field(default="auto", description="auto | manual")


class MetaRule(BaseModel):
    """A rule about *when and how* to evolve cognitive rules."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    condition: str = Field(description="When this meta-rule triggers an evolution action")
    action: str = Field(description="What evolution action to take")
    threshold: float = Field(default=0.5, description="Numeric threshold (meaning depends on condition)")

    
DEFAULT_META_RULES: list[dict[str, Any]] = [
    {
        "id": "meta_low_success",
        "name": "Refine on low success rate",
        "condition": "When a rule's success rate drops below threshold after at least 3 uses",
        "action": "refine_rule",
        "threshold": 0.4,
    },
    {
        "id": "meta_rule_gap",
        "name": "Create rule on gap",
        "condition": "When the monitor detects a rule gap (no matching rules for a situation)",
        "action": "create_rule",
        "threshold": 0.0,
    },
    {
        "id": "meta_high_similarity",
        "name": "Merge similar rules",
        "condition": "When two or more rules have overlapping conditions and similar strategies",
        "action": "merge_rules",
        "threshold": 0.8,
    },
    {
        "id": "meta_deprecate_unused",
        "name": "Deprecate stale rules",
        "condition": "When a rule has been used more than 5 times with success rate below threshold",
        "action": "deprecate_rule",
        "threshold": 0.2,
    },
]


# ---------------------------------------------------------------------------
# Rule Evolver
# ---------------------------------------------------------------------------

class RuleEvolver:
    """Mutates the cognitive rule base based on experience and meta-cognitive feedback."""

    def __init__(
        self,
        rule_engine: RuleEngine,
        data_dir: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.rule_engine = rule_engine
        self.verbose = verbose

        base = data_dir or Path(__file__).resolve().parent.parent.parent / "data"
        rules_dir = base / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = rules_dir / "evolution_log.json"
        self._meta_path = rules_dir / "meta_rules.json"
        self._log: list[EvolutionEntry] = []
        self._meta_rules: list[MetaRule] = []
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        if self._log_path.exists():
            raw = json.loads(self._log_path.read_text(encoding="utf-8"))
            self._log = [EvolutionEntry(**e) for e in raw]
        if self._meta_path.exists():
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._meta_rules = [MetaRule(**m) for m in raw]
        else:
            self._meta_rules = [MetaRule(**m) for m in DEFAULT_META_RULES]
            self._save_meta()

    def _save_log(self) -> None:
        self._log_path.write_text(
            json.dumps([e.model_dump() for e in self._log], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps([m.model_dump() for m in self._meta_rules], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _record(self, entry: EvolutionEntry) -> None:
        self._log.append(entry)
        self._save_log()
        if self.verbose:
            console.print(Panel(
                f"[bold]{entry.action.upper()}[/bold] → {entry.reason}\n"
                f"Target: {entry.target_rule_ids}",
                title="🧬 Rule Evolution",
                border_style="magenta",
            ))

    # -- core evolution actions ------------------------------------------------

    async def refine_rule(self, rule_id: str, feedback: str) -> CognitiveRule | None:
        """Refine an existing rule's condition/strategy based on feedback."""
        rule = self.rule_engine.get_rule(rule_id)
        if rule is None:
            return None

        before = rule.model_dump()

        prompt = (
            f"Current cognitive rule:\n"
            f"  Name: {rule.name}\n"
            f"  Condition: {rule.condition}\n"
            f"  Strategy: {rule.strategy}\n"
            f"  Success rate: {rule.success_rate:.0%} ({rule.success_count}/{rule.usage_count})\n\n"
            f"Feedback from experience:\n{feedback}\n\n"
            "Suggest an improved version of this rule. Return JSON with:\n"
            '  "condition": "updated condition",\n'
            '  "strategy": "updated strategy",\n'
            '  "confidence": float 0-1,\n'
            '  "reasoning": "why you made these changes"\n'
            "Reply ONLY with valid JSON, no markdown fences."
        )
        system = (
            "You are a rule refinement engine. Improve cognitive rules based on "
            "empirical feedback. Keep changes minimal but impactful. Reply ONLY with JSON."
        )
        raw = await generate(prompt, system=system)
        data = self._parse_json(raw)

        if data:
            self.rule_engine.update_rule(
                rule_id,
                condition=data.get("condition", rule.condition),
                strategy=data.get("strategy", rule.strategy),
                confidence=float(data.get("confidence", rule.confidence)),
            )
            updated = self.rule_engine.get_rule(rule_id)
            self._record(EvolutionEntry(
                action="refine",
                target_rule_ids=[rule_id],
                before=before,
                after=updated.model_dump() if updated else None,
                reason=data.get("reasoning", feedback),
            ))
            return updated
        return None

    async def create_rule(
        self,
        situation: str,
        learned_strategy: str,
        source_episode_id: str | None = None,
    ) -> CognitiveRule:
        """Create a new rule from experience."""
        prompt = (
            f"A cognitive agent encountered this situation:\n{situation}\n\n"
            f"It learned this strategy works:\n{learned_strategy}\n\n"
            "Formulate a reusable cognitive rule. Return JSON with:\n"
            '  "name": "short descriptive name",\n'
            '  "condition": "when this rule should trigger",\n'
            '  "strategy": "step-by-step strategy to follow",\n'
            '  "confidence": float 0-1\n'
            "Make the condition general enough to apply to similar situations, "
            "but specific enough to be useful. Reply ONLY with valid JSON."
        )
        system = "You are a rule creation engine. Formulate clear, reusable cognitive rules. Reply ONLY with JSON."
        raw = await generate(prompt, system=system)
        data = self._parse_json(raw) or {}

        new_rule = CognitiveRule(
            name=data.get("name", "Auto-generated rule"),
            condition=data.get("condition", situation),
            strategy=data.get("strategy", learned_strategy),
            confidence=float(data.get("confidence", 0.6)),
            created_by="self",
            source_episode=source_episode_id,
        )
        self.rule_engine.add_rule(new_rule)

        self._record(EvolutionEntry(
            action="create",
            target_rule_ids=[new_rule.id],
            after=new_rule.model_dump(),
            reason=f"Rule gap: {situation[:200]}",
        ))
        return new_rule

    async def merge_rules(self, rule_ids: list[str]) -> CognitiveRule | None:
        """Merge similar rules into a single, more general rule."""
        rules = [self.rule_engine.get_rule(rid) for rid in rule_ids]
        rules = [r for r in rules if r is not None]
        if len(rules) < 2:
            return None

        rules_text = "\n".join(
            f"- {r.name}: condition='{r.condition}', strategy='{r.strategy}'"
            for r in rules
        )
        prompt = (
            f"These cognitive rules are similar and should be merged:\n\n{rules_text}\n\n"
            "Create a single, more general rule that covers all cases. Return JSON with:\n"
            '  "name": "merged rule name",\n'
            '  "condition": "generalized condition",\n'
            '  "strategy": "unified strategy",\n'
            '  "confidence": float 0-1\n'
            "Reply ONLY with valid JSON."
        )
        system = "You are a rule merging engine. Create general rules from specific ones. Reply ONLY with JSON."
        raw = await generate(prompt, system=system)
        data = self._parse_json(raw) or {}

        merged = CognitiveRule(
            name=data.get("name", "Merged rule"),
            condition=data.get("condition", " OR ".join(r.condition for r in rules)),
            strategy=data.get("strategy", rules[0].strategy),
            confidence=float(data.get("confidence", 0.7)),
            created_by="self",
        )
        self.rule_engine.add_rule(merged)

        # Deprecate originals
        befores = {}
        for r in rules:
            befores[r.id] = r.model_dump()
            self.rule_engine.deprecate_rule(r.id)

        self._record(EvolutionEntry(
            action="merge",
            target_rule_ids=[r.id for r in rules] + [merged.id],
            before=befores,
            after=merged.model_dump(),
            reason=f"Merged {len(rules)} similar rules into one general rule",
        ))
        return merged

    async def deprecate_rule(self, rule_id: str, reason: str) -> bool:
        """Soft-delete a rule that is underperforming."""
        rule = self.rule_engine.get_rule(rule_id)
        if rule is None:
            return False

        before = rule.model_dump()
        self.rule_engine.deprecate_rule(rule_id)

        self._record(EvolutionEntry(
            action="deprecate",
            target_rule_ids=[rule_id],
            before=before,
            reason=reason,
        ))
        return True

    async def evolve_meta_rules(self) -> list[MetaRule]:
        """
        🌀 THE STRANGE LOOP — modify the rules that govern rule evolution itself.

        Examines the evolution log to assess whether the current meta-rules
        are effective, then adjusts them.
        """
        recent_log = self._log[-20:] if self._log else []
        if not recent_log:
            if self.verbose:
                console.print("[dim]No evolution history yet — skipping meta-rule evolution.[/dim]")
            return self._meta_rules

        log_summary = "\n".join(
            f"- [{e.action}] {e.reason[:120]} (target: {e.target_rule_ids})"
            for e in recent_log
        )
        meta_text = "\n".join(
            f"- [{m.id}] {m.name}: condition='{m.condition}', action='{m.action}', threshold={m.threshold}"
            for m in self._meta_rules
        )

        prompt = (
            f"You are reviewing the META-RULES of a self-evolving cognitive agent.\n\n"
            f"Current meta-rules (rules about when/how to evolve cognitive rules):\n{meta_text}\n\n"
            f"Recent evolution history:\n{log_summary}\n\n"
            "Based on the evolution history, are the meta-rules working well?\n"
            "Should any thresholds be adjusted? Should new meta-rules be added?\n"
            "Should any meta-rules be removed?\n\n"
            "Return a JSON object with:\n"
            '  "assessment": "brief assessment of meta-rule effectiveness",\n'
            '  "changes": [\n'
            '    {"meta_rule_id": "...", "field": "threshold|condition|action", "new_value": ...},\n'
            "    ...\n"
            "  ],\n"
            '  "new_meta_rules": [\n'
            '    {"name": "...", "condition": "...", "action": "...", "threshold": float},\n'
            "    ...\n"
            "  ],\n"
            '  "remove_ids": ["ids to remove"]\n'
            "Reply ONLY with valid JSON, no markdown fences."
        )
        system = (
            "You are a meta-cognitive evolution engine — you improve the rules that "
            "govern how an AI agent improves its own rules. This is a strange loop: "
            "you are modifying yourself. Be thoughtful and conservative — bad meta-rule "
            "changes can cascade. Reply ONLY with JSON."
        )

        raw = await generate(prompt, system=system)
        data = self._parse_json(raw) or {}

        before_snapshot = [m.model_dump() for m in self._meta_rules]

        # Apply changes to existing meta-rules
        for change in data.get("changes", []):
            mid = change.get("meta_rule_id", "")
            field = change.get("field", "")
            new_val = change.get("new_value")
            for mr in self._meta_rules:
                if mr.id == mid and hasattr(mr, field) and new_val is not None:
                    setattr(mr, field, new_val)

        # Add new meta-rules
        for new_mr in data.get("new_meta_rules", []):
            self._meta_rules.append(MetaRule(
                name=new_mr.get("name", "New meta-rule"),
                condition=new_mr.get("condition", ""),
                action=new_mr.get("action", ""),
                threshold=float(new_mr.get("threshold", 0.5)),
            ))

        # Remove meta-rules
        remove_ids = set(data.get("remove_ids", []))
        if remove_ids:
            self._meta_rules = [m for m in self._meta_rules if m.id not in remove_ids]

        self._save_meta()

        self._record(EvolutionEntry(
            action="evolve_meta",
            target_rule_ids=[m.id for m in self._meta_rules],
            before={"meta_rules": before_snapshot},
            after={"meta_rules": [m.model_dump() for m in self._meta_rules]},
            reason=data.get("assessment", "Periodic meta-rule review"),
        ))

        if self.verbose:
            console.print(Panel(
                f"[bold yellow]🌀 STRANGE LOOP[/bold yellow]\n"
                f"Assessment: {data.get('assessment', 'N/A')}\n"
                f"Changes: {len(data.get('changes', []))}\n"
                f"New meta-rules: {len(data.get('new_meta_rules', []))}\n"
                f"Removed: {len(remove_ids)}",
                title="Meta-Rule Evolution",
                border_style="yellow",
            ))

        return self._meta_rules

    # -- auto-evolve (main entry point) ----------------------------------------

    async def auto_evolve(
        self,
        report: MonitorReport,
        episode_context: dict[str, Any] | None = None,
    ) -> list[EvolutionEntry]:
        """
        High-level entry: read the monitor report, check meta-rules, and
        perform whatever evolution actions are warranted.
        """
        actions_taken: list[EvolutionEntry] = []
        log_len_before = len(self._log)

        # 1. Process suggested actions from monitor
        for suggestion in report.suggested_actions:
            if suggestion.action_type == "refine_rule" and suggestion.target_rule_id:
                result = await self.refine_rule(suggestion.target_rule_id, suggestion.details)
                if result:
                    if self.verbose:
                        console.print(f"  [green]✓[/green] Refined rule: {result.name}")

            elif suggestion.action_type == "create_rule":
                situation = episode_context.get("task", report.task) if episode_context else report.task
                strategy = suggestion.details
                ep_id = episode_context.get("episode_id") if episode_context else None
                new_rule = await self.create_rule(situation, strategy, ep_id)
                if self.verbose:
                    console.print(f"  [green]✓[/green] Created rule: {new_rule.name}")

            elif suggestion.action_type == "deprecate_rule" and suggestion.target_rule_id:
                await self.deprecate_rule(suggestion.target_rule_id, suggestion.details)
                if self.verbose:
                    console.print(f"  [yellow]⚠[/yellow] Deprecated rule: {suggestion.target_rule_id}")

        # 2. Check meta-rules for proactive evolution
        for meta_rule in self._meta_rules:
            await self._check_meta_rule(meta_rule)

        # 3. Periodically evolve meta-rules themselves (every 5 evolution actions)
        total_evolutions = len(self._log)
        if total_evolutions > 0 and total_evolutions % 5 == 0:
            if self.verbose:
                console.print("\n[bold yellow]🌀 Triggering meta-rule self-evolution...[/bold yellow]")
            await self.evolve_meta_rules()

        actions_taken = self._log[log_len_before:]
        return actions_taken

    async def _check_meta_rule(self, meta_rule: MetaRule) -> None:
        """Evaluate a single meta-rule against current rule state."""
        if meta_rule.action == "deprecate_rule":
            for rule in self.rule_engine.all_rules():
                if (
                    rule.usage_count >= 5
                    and rule.success_rate < meta_rule.threshold
                    and not rule.deprecated
                ):
                    await self.deprecate_rule(
                        rule.id,
                        f"Meta-rule '{meta_rule.name}': success rate "
                        f"{rule.success_rate:.0%} below threshold {meta_rule.threshold:.0%}",
                    )

        elif meta_rule.action == "refine_rule":
            for rule in self.rule_engine.all_rules():
                if (
                    rule.usage_count >= 3
                    and rule.success_rate < meta_rule.threshold
                    and not rule.deprecated
                ):
                    await self.refine_rule(
                        rule.id,
                        f"Meta-rule '{meta_rule.name}': success rate "
                        f"{rule.success_rate:.0%} is low, needs refinement",
                    )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, TypeError):
            return None

    @property
    def evolution_log(self) -> list[EvolutionEntry]:
        return list(self._log)

    @property
    def meta_rules(self) -> list[MetaRule]:
        return list(self._meta_rules)
