"""
seca.agent — Main SECA agent
==============================

``SECAAgent`` is the top-level orchestrator.  It wires together every
sub-system and runs the full cognitive loop:

    task → memory retrieval → analogy search → rule matching → reasoning →
    meta-cognitive monitoring → rule evolution → experience storage → result
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from seca.foundation.llm import generate
from seca.foundation.memory import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    Episode,
)
from seca.foundation.tools import ToolRegistry, get_global_registry
from seca.cognitive.rules import CognitiveRule, RuleEngine
from seca.cognitive.reasoning import ReasoningEngine, ReasoningTrace
from seca.cognitive.analogy import AnalogyEngine, Analogy
from seca.metacognitive.monitor import MetaCognitiveMonitor, MonitorReport
from seca.metacognitive.rule_evolver import RuleEvolver, EvolutionEntry

console = Console()


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Everything produced by a single agent run."""

    task: str
    answer: str = ""
    reasoning_trace: ReasoningTrace | None = None
    rules_used: list[str] = Field(default_factory=list, description="IDs of rules applied")
    rules_evolved: list[str] = Field(default_factory=list, description="IDs of rules created/modified")
    monitor_report: MonitorReport | None = None
    episodes_referenced: list[str] = Field(
        default_factory=list,
        description="IDs of episodic memories consulted",
    )
    episode_stored: str | None = Field(default=None, description="ID of the episode saved from this run")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SECAAgent:
    """Self-Evolving Cognitive Agent — the top-level orchestrator."""

    def __init__(
        self,
        verbose: bool = False,
        data_dir: str | Path = "data",
    ) -> None:
        self.verbose = verbose
        self._data = Path(data_dir)
        self._data.mkdir(parents=True, exist_ok=True)

        # Foundation layer
        mem_dir = self._data / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        self.working_memory = WorkingMemory(path=mem_dir / "working.json")
        self.episodic_memory = EpisodicMemory(path=mem_dir / "episodic.json")
        self.semantic_memory = SemanticMemory(path=mem_dir / "semantic.json")
        self.tools = get_global_registry()

        # Cognitive layer
        rules_dir = self._data / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        self.rule_engine = RuleEngine(path=rules_dir / "cognitive_rules.json")
        self.reasoning_engine = ReasoningEngine(self.rule_engine)
        self.analogy_engine = AnalogyEngine(self.episodic_memory)

        # Meta-cognitive layer
        self.monitor = MetaCognitiveMonitor()
        self.rule_evolver = RuleEvolver(
            rule_engine=self.rule_engine,
            data_dir=self._data,
            verbose=verbose,
        )

    async def run(self, task: str) -> AgentResult:
        """Execute the full cognitive loop for *task*."""
        result = AgentResult(task=task)

        if self.verbose:
            console.print(Panel(f"[bold cyan]{task}[/bold cyan]", title="📋 New Task"))

        # 1. Put task in working memory
        self.working_memory.clear()
        self.working_memory.put("current_task", task)

        # 2. Search episodic memory for relevant past experiences
        past_episodes = self.episodic_memory.search(task, top_k=3)
        result.episodes_referenced = [ep.id for ep in past_episodes]
        if self.verbose and past_episodes:
            console.print(f"[dim]📚 Found {len(past_episodes)} relevant past episodes[/dim]")

        # 3. Search for analogies
        analogies: list[Analogy] = []
        if past_episodes:
            analogies = await self.analogy_engine.find_analogies(task, top_k=2)
            if self.verbose and analogies:
                console.print(f"[dim]🔗 Found {len(analogies)} analogies (best score: {analogies[0].similarity_score:.2f})[/dim]")

        # 4. Build reasoning context
        context: dict[str, Any] = {
            "working_memory": self.working_memory.all_items(),
            "relevant_episodes": [
                {"situation": ep.situation, "strategy": ep.strategy_used, "outcome": ep.outcome}
                for ep in past_episodes
            ],
        }
        if analogies:
            context["analogies"] = self.analogy_engine.to_context(analogies)

        # 5. Reason
        if self.verbose:
            console.print("[bold]🧠 Starting reasoning...[/bold]")

        trace: ReasoningTrace = await self.reasoning_engine.reason(
            task=task,
            context=context,
            max_steps=10,
        )
        result.reasoning_trace = trace
        result.answer = trace.final_answer
        result.rules_used = [r.id for r in trace.matched_rules]

        if self.verbose:
            self._print_trace(trace)

        # 6. Meta-cognitive monitoring
        if self.verbose:
            console.print("[bold]🔍 Meta-cognitive analysis...[/bold]")

        report: MonitorReport = await self.monitor.analyze(trace)
        result.monitor_report = report

        if self.verbose:
            self._print_report(report)

        # 7. Rule evolution
        if report.issues_detected or report.suggested_actions:
            if self.verbose:
                console.print("[bold]🧬 Triggering rule evolution...[/bold]")

            episode_ctx = {
                "task": task,
                "episode_id": None,
            }
            evolutions: list[EvolutionEntry] = await self.rule_evolver.auto_evolve(
                report, episode_context=episode_ctx
            )
            result.rules_evolved = [
                rid for evo in evolutions for rid in evo.target_rule_ids
            ]

            if self.verbose and evolutions:
                console.print(f"[green]  {len(evolutions)} evolution action(s) taken[/green]")

        # 8. Store this experience
        strategies = ", ".join(r.name for r in trace.matched_rules) or "no specific rule"
        episode = Episode(
            situation=task,
            strategy_used=strategies,
            outcome=trace.final_answer[:500],
            reflection=(
                f"Confidence: {report.confidence_assessment:.0%}. "
                f"Issues: {len(report.issues_detected)}. "
                f"Mode: {trace.mode.value}. "
                f"Success: {trace.success}."
            ),
            tags=self._extract_tags(task),
        )
        ep_id = self.episodic_memory.store(episode)
        result.episode_stored = ep_id

        if self.verbose:
            console.print(f"[dim]💾 Experience saved as episode {ep_id}[/dim]")
            console.print(Panel(
                result.answer[:500] if result.answer else "[red]No answer produced[/red]",
                title="✅ Final Answer",
                border_style="green",
            ))

        return result

    # -- display helpers -------------------------------------------------------

    def _print_trace(self, trace: ReasoningTrace) -> None:
        table = Table(title=f"Reasoning Trace ({trace.mode.value})", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Thought", max_width=50)
        table.add_column("Action", max_width=20)
        table.add_column("Observation", max_width=40)
        for step in trace.steps:
            table.add_row(
                str(step.step_number),
                step.thought[:50],
                step.action[:20],
                step.observation[:40],
            )
        console.print(table)

    def _print_report(self, report: MonitorReport) -> None:
        severity_colors = {"low": "dim", "medium": "yellow", "high": "red"}
        lines = [f"Confidence: {report.confidence_assessment:.0%}"]
        for issue in report.issues_detected:
            color = severity_colors.get(issue.severity, "white")
            lines.append(f"  [{color}]⚠ [{issue.issue_type}] {issue.description[:80]}[/{color}]")
        if report.suggested_actions:
            lines.append("Suggested actions:")
            for sa in report.suggested_actions:
                lines.append(f"  → {sa.action_type}: {sa.details[:60]}")
        console.print(Panel("\n".join(lines), title="🔍 Monitor Report", border_style="blue"))

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """Quick keyword extraction for episode tagging."""
        keywords = [
            "contradiction", "ambiguous", "math", "code", "analogy",
            "planning", "creative", "factual", "comparison", "decision",
        ]
        text_lower = text.lower()
        return [k for k in keywords if k in text_lower]
