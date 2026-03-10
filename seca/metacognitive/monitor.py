"""
seca.metacognitive.monitor — Meta-cognitive monitoring
=======================================================

``MetaCognitiveMonitor`` observes a reasoning trace and detects issues:

* **Loop detection** — repeated thoughts or actions
* **Contradiction detection** — conflicting statements in the trace
* **Confidence decline** — strategy appears to be losing traction
* **Strategy effectiveness** — did the chosen strategy actually help?

Produces a ``MonitorReport`` consumed by the rule evolver.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from seca.foundation.llm import generate
from seca.cognitive.reasoning import ReasoningTrace, ReasoningStep


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    """A single issue detected during monitoring."""

    issue_type: str = Field(description="Category: loop | contradiction | ineffective_strategy | low_confidence | other")
    description: str
    severity: str = Field(default="medium", description="low | medium | high")
    related_steps: list[int] = Field(default_factory=list, description="Step numbers involved")


class SuggestedAction(BaseModel):
    """An action the system should take in response to a monitoring issue."""

    action_type: str = Field(description="refine_rule | create_rule | deprecate_rule | retry | escalate")
    target_rule_id: str | None = None
    details: str = ""


class MonitorReport(BaseModel):
    """Full report from the meta-cognitive monitor."""

    task: str
    issues_detected: list[Issue] = Field(default_factory=list)
    confidence_assessment: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the reasoning quality",
    )
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class MetaCognitiveMonitor:
    """Observes reasoning traces and produces diagnostic reports."""

    async def analyze(self, trace: ReasoningTrace) -> MonitorReport:
        """Run all detectors on *trace* and return a consolidated report."""
        report = MonitorReport(task=trace.task)

        # Heuristic detectors (fast, no LLM)
        self._detect_loops(trace, report)
        self._assess_step_quality(trace, report)

        # LLM-based deep analysis
        await self._llm_analysis(trace, report)

        # Generate suggested actions
        self._generate_suggestions(trace, report)

        return report

    # -- heuristic detectors --

    @staticmethod
    def _detect_loops(trace: ReasoningTrace, report: MonitorReport) -> None:
        """Detect repeated thoughts or actions in the trace."""
        thought_counter: Counter[str] = Counter()
        for step in trace.steps:
            normalised = step.thought.strip().lower()[:200]
            thought_counter[normalised] += 1

        for thought, count in thought_counter.items():
            if count >= 2:
                report.issues_detected.append(Issue(
                    issue_type="loop",
                    description=f"Thought repeated {count} times: '{thought[:80]}...'",
                    severity="high" if count >= 3 else "medium",
                    related_steps=[
                        s.step_number for s in trace.steps
                        if s.thought.strip().lower()[:200] == thought
                    ],
                ))

    @staticmethod
    def _assess_step_quality(trace: ReasoningTrace, report: MonitorReport) -> None:
        """Check for obvious quality issues in steps."""
        parse_errors = [s for s in trace.steps if s.action == "parse_error"]
        if parse_errors:
            report.issues_detected.append(Issue(
                issue_type="other",
                description=f"{len(parse_errors)} steps had parse errors — LLM output was not structured correctly",
                severity="medium",
                related_steps=[s.step_number for s in parse_errors],
            ))

        if trace.steps and not trace.success:
            report.issues_detected.append(Issue(
                issue_type="ineffective_strategy",
                description="Reasoning did not reach a final answer within the step limit",
                severity="high",
                related_steps=[trace.steps[-1].step_number],
            ))

        # Check for rule gaps
        if trace.rule_gaps:
            for gap in trace.rule_gaps:
                report.issues_detected.append(Issue(
                    issue_type="other",
                    description=f"No rules matched for situation: '{gap[:100]}'",
                    severity="medium",
                ))

    # -- LLM-based analysis --

    async def _llm_analysis(self, trace: ReasoningTrace, report: MonitorReport) -> None:
        """Use the LLM to perform deeper analysis of the reasoning trace."""
        if not trace.steps:
            report.confidence_assessment = 0.5
            report.summary = "No reasoning steps to analyze."
            return

        steps_text = "\n".join(
            f"Step {s.step_number}: thought='{s.thought[:150]}', action='{s.action}', observation='{s.observation[:150]}'"
            for s in trace.steps
        )
        rules_text = ", ".join(r.name for r in trace.matched_rules) or "None"

        prompt = (
            f"Analyze this reasoning trace for quality issues:\n\n"
            f"Task: {trace.task}\n"
            f"Mode: {trace.mode.value}\n"
            f"Rules applied: {rules_text}\n"
            f"Success: {trace.success}\n"
            f"Steps:\n{steps_text}\n\n"
            "Return a JSON object with:\n"
            '  "confidence": float 0-1 (overall reasoning quality),\n'
            '  "contradictions": [list of any contradictory statements found],\n'
            '  "summary": "brief assessment of the reasoning quality"\n'
            "\nReply ONLY with valid JSON, no markdown fences."
        )
        system = (
            "You are a meta-cognitive analysis system. Critically evaluate "
            "reasoning traces for logical errors, contradictions, and quality issues. "
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

            report.confidence_assessment = float(data.get("confidence", 0.7))
            report.summary = data.get("summary", "")

            for contradiction in data.get("contradictions", []):
                report.issues_detected.append(Issue(
                    issue_type="contradiction",
                    description=str(contradiction),
                    severity="high",
                ))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            report.confidence_assessment = 0.5
            report.summary = f"Could not parse LLM analysis. Raw: {raw[:200]}"

    # -- suggestion generation --

    @staticmethod
    def _generate_suggestions(trace: ReasoningTrace, report: MonitorReport) -> None:
        """Turn detected issues into concrete suggested actions."""
        for issue in report.issues_detected:
            if issue.issue_type == "loop":
                report.suggested_actions.append(SuggestedAction(
                    action_type="refine_rule",
                    target_rule_id=trace.matched_rules[0].id if trace.matched_rules else None,
                    details=f"Rule may be causing a reasoning loop: {issue.description}",
                ))
            elif issue.issue_type == "contradiction":
                report.suggested_actions.append(SuggestedAction(
                    action_type="create_rule",
                    details=f"Contradiction detected — consider a rule to handle: {issue.description}",
                ))
            elif issue.issue_type == "ineffective_strategy":
                if trace.matched_rules:
                    report.suggested_actions.append(SuggestedAction(
                        action_type="refine_rule",
                        target_rule_id=trace.matched_rules[0].id,
                        details="Strategy did not produce a final answer; consider refining.",
                    ))
            elif issue.issue_type == "other" and "No rules matched" in issue.description:
                report.suggested_actions.append(SuggestedAction(
                    action_type="create_rule",
                    details=f"Rule gap detected: {issue.description}",
                ))

        # If confidence is very low, suggest a broad review
        if report.confidence_assessment < 0.4 and not any(
            sa.action_type == "escalate" for sa in report.suggested_actions
        ):
            report.suggested_actions.append(SuggestedAction(
                action_type="escalate",
                details="Overall reasoning confidence is very low — consider human review.",
            ))
