"""
seca.cognitive.reasoning — Reasoning engine
=============================================

Orchestrates multi-step reasoning over a task.  Each step is recorded as a
``ReasoningStep`` so the meta-cognitive layer can review the full trace.

Supported modes:
  • **chain-of-thought** — linear step-by-step reasoning
  • **decomposition**    — split into sub-problems, solve each, integrate
  • **analogy**          — find structurally similar past problems and adapt
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from seca.foundation.llm import generate
from seca.cognitive.rules import CognitiveRule, RuleEngine, RuleMatchResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ReasoningMode(str, Enum):
    CHAIN_OF_THOUGHT = "chain-of-thought"
    DECOMPOSITION = "decomposition"
    ANALOGY = "analogy"


class ReasoningStep(BaseModel):
    """One atomic step in a reasoning chain."""

    step_number: int
    thought: str = Field(description="What the agent is thinking")
    action: str = Field(default="", description="Action taken (tool call, sub-question, etc.)")
    observation: str = Field(default="", description="Result of the action")
    rule_used: str | None = Field(default=None, description="ID of the cognitive rule applied")


class ReasoningTrace(BaseModel):
    """Complete trace of a reasoning session."""

    task: str
    mode: ReasoningMode
    steps: list[ReasoningStep] = Field(default_factory=list)
    matched_rules: list[CognitiveRule] = Field(default_factory=list)
    rule_gaps: list[str] = Field(
        default_factory=list,
        description="Situations where no rule matched — learning opportunities",
    )
    final_answer: str = ""
    success: bool | None = None


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Receives a task, matches rules, and executes a reasoning chain."""

    def __init__(self, rule_engine: RuleEngine) -> None:
        self.rule_engine = rule_engine

    async def reason(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        mode: ReasoningMode | None = None,
        max_steps: int = 10,
    ) -> ReasoningTrace:
        """Run a full reasoning session.

        Parameters
        ----------
        task : str
            Natural-language task description.
        context : dict | None
            Extra context (memory contents, tool results, etc.).
        mode : ReasoningMode | None
            Force a reasoning mode.  ``None`` means auto-select.
        max_steps : int
            Safety limit on the number of reasoning steps.
        """
        # 1. Match rules
        match_result: RuleMatchResult = await self.rule_engine.match(task)

        # 2. Decide mode if not forced
        if mode is None:
            mode = await self._select_mode(task, match_result)

        trace = ReasoningTrace(
            task=task,
            mode=mode,
            matched_rules=match_result.matched_rules,
        )

        if match_result.rule_gap:
            trace.rule_gaps.append(task)

        # 3. Dispatch to mode-specific reasoning
        if mode == ReasoningMode.CHAIN_OF_THOUGHT:
            await self._chain_of_thought(task, context, match_result, trace, max_steps)
        elif mode == ReasoningMode.DECOMPOSITION:
            await self._decomposition(task, context, match_result, trace, max_steps)
        elif mode == ReasoningMode.ANALOGY:
            await self._analogy(task, context, match_result, trace, max_steps)

        # 4. Record usage
        for rule in match_result.matched_rules:
            self.rule_engine.record_usage(rule.id, success=trace.success or False)

        return trace

    # -- mode selection --

    async def _select_mode(self, task: str, match_result: RuleMatchResult) -> ReasoningMode:
        """Let the LLM pick the best reasoning mode for *task*."""
        # Check if analogy rule matched
        has_analogy_rule = any(r.id == "rule_analogy" for r in match_result.matched_rules)
        has_decomp_rule = any(r.id == "rule_decomposition" for r in match_result.matched_rules)

        if has_analogy_rule:
            return ReasoningMode.ANALOGY
        if has_decomp_rule:
            return ReasoningMode.DECOMPOSITION

        prompt = (
            f"Task: {task}\n\n"
            "Choose the best reasoning mode:\n"
            "1. chain-of-thought — linear step-by-step reasoning\n"
            "2. decomposition — break into sub-problems\n"
            "3. analogy — find similar past problems\n\n"
            "Reply with just the mode name, e.g. 'chain-of-thought'."
        )
        raw = await generate(prompt, system="You are a reasoning mode selector. Reply with ONLY the mode name.")
        raw_clean = raw.strip().lower().strip("'\"")
        for m in ReasoningMode:
            if m.value in raw_clean:
                return m
        return ReasoningMode.CHAIN_OF_THOUGHT

    # -- chain-of-thought --

    async def _chain_of_thought(
        self,
        task: str,
        context: dict[str, Any] | None,
        match_result: RuleMatchResult,
        trace: ReasoningTrace,
        max_steps: int,
    ) -> None:
        strategies = "\n".join(
            f"- [{r.name}]: {r.strategy}" for r in match_result.matched_rules
        )
        context_str = json.dumps(context, ensure_ascii=False, indent=2) if context else "None"

        system = (
            "You are a step-by-step reasoning engine. "
            "Think through the task carefully, applying any relevant strategies. "
            "For each step, output valid JSON with keys: thought, action, observation. "
            "When you have a final answer, set action to 'FINAL_ANSWER' and put the answer in observation. "
            "Output one step at a time as a JSON object. No markdown fences."
        )

        conversation = (
            f"Task: {task}\n\n"
            f"Relevant strategies:\n{strategies or 'None'}\n\n"
            f"Context:\n{context_str}\n\n"
            "Begin reasoning step by step. Output step 1 as JSON."
        )

        for step_num in range(1, max_steps + 1):
            raw = await generate(conversation, system=system)
            parsed = self._parse_step(raw, step_num, match_result)
            trace.steps.append(parsed)

            if parsed.action.upper() == "FINAL_ANSWER":
                trace.final_answer = parsed.observation
                trace.success = True
                return

            # Build context for next step
            conversation += f"\n\nStep {step_num} result: {raw}\n\nContinue with step {step_num + 1}."

        # Ran out of steps
        trace.final_answer = trace.steps[-1].observation if trace.steps else ""
        trace.success = False

    # -- decomposition --

    async def _decomposition(
        self,
        task: str,
        context: dict[str, Any] | None,
        match_result: RuleMatchResult,
        trace: ReasoningTrace,
        max_steps: int,
    ) -> None:
        # Step 1: decompose
        decompose_prompt = (
            f"Task: {task}\n\n"
            "Break this task into a list of independent sub-tasks. "
            "Return a JSON array of strings, each describing one sub-task. No markdown fences."
        )
        raw = await generate(
            decompose_prompt,
            system="You are a task decomposition engine. Return ONLY a JSON array of strings.",
        )

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            sub_tasks: list[str] = json.loads(cleaned.strip())
        except (json.JSONDecodeError, TypeError):
            sub_tasks = [task]

        trace.steps.append(ReasoningStep(
            step_number=1,
            thought=f"Decomposed task into {len(sub_tasks)} sub-tasks",
            action="decompose",
            observation=json.dumps(sub_tasks, ensure_ascii=False),
            rule_used="rule_decomposition" if any(
                r.id == "rule_decomposition" for r in match_result.matched_rules
            ) else None,
        ))

        # Step 2: solve each sub-task
        partial_results: list[str] = []
        for i, sub in enumerate(sub_tasks, start=2):
            if i > max_steps:
                break
            solve_prompt = (
                f"Sub-task: {sub}\n\n"
                f"Original task context: {task}\n\n"
                "Solve this sub-task. Provide a clear answer."
            )
            answer = await generate(solve_prompt, system="You are a reasoning engine solving a sub-task.")
            partial_results.append(answer)
            trace.steps.append(ReasoningStep(
                step_number=i,
                thought=f"Solving sub-task: {sub}",
                action="solve_subtask",
                observation=answer,
            ))

        # Step 3: integrate
        integrate_prompt = (
            f"Original task: {task}\n\n"
            f"Sub-task results:\n" +
            "\n".join(f"{i+1}. {r}" for i, r in enumerate(partial_results)) +
            "\n\nIntegrate these results into a coherent final answer."
        )
        final = await generate(
            integrate_prompt,
            system="You are a reasoning engine integrating partial results into a final answer.",
        )
        trace.steps.append(ReasoningStep(
            step_number=len(trace.steps) + 1,
            thought="Integrating sub-task results",
            action="FINAL_ANSWER",
            observation=final,
        ))
        trace.final_answer = final
        trace.success = True

    # -- analogy --

    async def _analogy(
        self,
        task: str,
        context: dict[str, Any] | None,
        match_result: RuleMatchResult,
        trace: ReasoningTrace,
        max_steps: int,
    ) -> None:
        # The analogy engine will be called by the agent; here we do the
        # LLM-based analogical reasoning inline.
        analogies_info = ""
        if context and "analogies" in context:
            analogies_info = json.dumps(context["analogies"], ensure_ascii=False, indent=2)

        prompt = (
            f"Task: {task}\n\n"
            f"Similar past experiences:\n{analogies_info or 'No past analogies available.'}\n\n"
            "Using analogical reasoning:\n"
            "1. Identify structural similarities between past experiences and the current task.\n"
            "2. Map the solution strategy from the past to the present.\n"
            "3. Adapt the strategy to fit the new context.\n"
            "4. Verify the adapted strategy makes sense.\n\n"
            "Provide your reasoning and final answer."
        )
        system = "You are an analogical reasoning engine. Transfer knowledge from past experiences to solve new problems."

        raw = await generate(prompt, system=system)

        trace.steps.append(ReasoningStep(
            step_number=1,
            thought="Applying analogical reasoning from past experiences",
            action="FINAL_ANSWER",
            observation=raw,
            rule_used="rule_analogy" if any(
                r.id == "rule_analogy" for r in match_result.matched_rules
            ) else None,
        ))
        trace.final_answer = raw
        trace.success = True

    # -- helpers --

    @staticmethod
    def _parse_step(raw: str, step_num: int, match_result: RuleMatchResult) -> ReasoningStep:
        """Parse an LLM response into a ``ReasoningStep``."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned.strip())
            rule_used = None
            if match_result.matched_rules:
                rule_used = match_result.matched_rules[0].id
            return ReasoningStep(
                step_number=step_num,
                thought=data.get("thought", ""),
                action=data.get("action", ""),
                observation=data.get("observation", ""),
                rule_used=rule_used,
            )
        except (json.JSONDecodeError, AttributeError):
            return ReasoningStep(
                step_number=step_num,
                thought=raw,
                action="parse_error",
                observation="Could not parse LLM response as JSON",
            )
