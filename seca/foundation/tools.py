"""
seca.foundation.tools — Tool registration and invocation
=========================================================

Provides a ``@tool(name, description)`` decorator to register callable tools,
plus a ``ToolRegistry`` that manages discovery and invocation.

Built-in tools:
  • ``web_search``  — simulated web search (returns canned results)
  • ``calculator``  — evaluates simple math expressions safely
  • ``file_read``   — reads a text file from disk
  • ``file_write``  — writes text to a file on disk
"""

from __future__ import annotations

import math
import ast
import operator
from pathlib import Path
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Metadata about a registered tool."""

    name: str
    description: str
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter name → human description",
    )


class ToolResult(BaseModel):
    """Result returned after running a tool."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str] | None = None,
    ) -> Callable:
        """Decorator factory — use as ``@registry.register(name, desc)``."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec(
                name=name,
                description=description,
                parameters=parameters or {},
            )
            self._tools[name] = (spec, fn)
            return fn

        return decorator

    async def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up *name* and call the underlying function."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: {name}",
            )
        spec, fn = entry
        try:
            import asyncio

            if asyncio.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = fn(**kwargs)
            return ToolResult(tool_name=name, success=True, output=result)
        except Exception as exc:
            return ToolResult(tool_name=name, success=False, error=str(exc))

    def list_tools(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def get_spec(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Convenience decorator (module-level singleton)
# ---------------------------------------------------------------------------

_global_registry = ToolRegistry()


def tool(
    name: str, description: str, parameters: dict[str, str] | None = None
) -> Callable:
    """Module-level shortcut for ``_global_registry.register(...)``."""
    return _global_registry.register(name, description, parameters)


def get_global_registry() -> ToolRegistry:
    return _global_registry


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

@tool(
    name="web_search",
    description="Search the web for a query and return a list of result snippets. (Simulated)",
    parameters={"query": "The search query string"},
)
def web_search(query: str) -> list[dict[str, str]]:
    """Simulated web search — returns canned results with the query echoed."""
    return [
        {
            "title": f"Result 1 for '{query}'",
            "snippet": f"This is a simulated search result about {query}. "
            "It contains relevant information that would normally come from the web.",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}",
        },
        {
            "title": f"Result 2 for '{query}'",
            "snippet": f"Another perspective on {query}. "
            "Multiple sources confirm different aspects of this topic.",
            "url": f"https://example.org/article/{query.replace(' ', '-')}",
        },
        {
            "title": f"Result 3 for '{query}'",
            "snippet": f"A comprehensive overview of {query} with recent updates "
            "and expert opinions.",
            "url": f"https://wiki.example.com/{query.replace(' ', '_')}",
        },
    ]


# -- safe math evaluator (no exec/eval) --

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            fn = _SAFE_FUNCS[node.func.id]
            if callable(fn):
                args = [_safe_eval_node(a) for a in node.args]
                return fn(*args)
        raise ValueError(f"Unsupported function call")
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCS:
        val = _SAFE_FUNCS[node.id]
        if not callable(val):
            return val
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool(
    name="calculator",
    description="Evaluate a mathematical expression safely and return the numeric result.",
    parameters={"expression": "A math expression like '2 + 3 * 4' or 'sqrt(16)'"},
)
def calculator(expression: str) -> float | int:
    """Safe math evaluation — no arbitrary code execution."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree)


@tool(
    name="file_read",
    description="Read the content of a text file from disk.",
    parameters={"path": "Path to the file to read"},
)
def file_read(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


@tool(
    name="file_write",
    description="Write text content to a file on disk (creates parent dirs if needed).",
    parameters={"path": "Path to the file", "content": "Text content to write"},
)
def file_write(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"
