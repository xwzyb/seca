"""
SECA — Self-Evolving Cognitive Agent
=====================================

A three-layer cognitive architecture:
  • Foundation  – LLM calls, tool use, memory storage
  • Cognitive   – reasoning, planning, analogy, rule-based learning
  • Meta-Cognitive – monitoring reasoning, evaluating strategies, evolving rules

The defining feature is the *strange loop*: the meta-cognitive layer can modify
its own operating rules, creating a self-referential system.
"""

from seca.agent import SECAAgent  # noqa: F401

__version__ = "0.1.0"
__all__ = ["SECAAgent"]
