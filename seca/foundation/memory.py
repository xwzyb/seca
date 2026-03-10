"""
seca.foundation.memory — Three-tier memory system
===================================================

* **WorkingMemory** — short-term scratch-pad for the current task (lives in RAM,
  optionally flushed to disk).
* **EpisodicMemory** — experience store: *{situation, strategy_used, outcome,
  reflection}*.  Supports simple similarity retrieval via keyword overlap.
* **SemanticMemory** — long-term factual knowledge store.

All three persist to JSON files under ``data/memory/``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

class WorkingMemoryItem(BaseModel):
    """A single item held in working memory."""

    key: str
    value: Any
    added_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkingMemory:
    """Short-term, task-scoped memory.

    Behaves like a key-value store that is flushed to
    ``data/memory/working.json`` on every write so it survives crashes.
    """

    _path: Path
    _items: dict[str, WorkingMemoryItem]

    def __init__(self, path: Path | None = None) -> None:
        _ensure_dir()
        self._path = path or (_DATA_DIR / "working.json")
        self._items = {}
        self._load()

    # -- persistence --

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = {k: WorkingMemoryItem(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                {k: v.model_dump() for k, v in self._items.items()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- public API --

    def put(self, key: str, value: Any) -> None:
        self._items[key] = WorkingMemoryItem(key=key, value=value)
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        item = self._items.get(key)
        return item.value if item else default

    def remove(self, key: str) -> None:
        self._items.pop(key, None)
        self._save()

    def clear(self) -> None:
        self._items.clear()
        self._save()

    def all_items(self) -> dict[str, Any]:
        return {k: v.value for k, v in self._items.items()}

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._items)} items)"


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    """One recorded experience."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    situation: str
    strategy_used: str
    outcome: str
    reflection: str = ""
    tags: list[str] = Field(default_factory=list)


class EpisodicMemory:
    """Experience memory with simple similarity retrieval.

    Persists to ``data/memory/episodic.json``.
    """

    _path: Path
    _episodes: list[Episode]

    def __init__(self, path: Path | None = None) -> None:
        _ensure_dir()
        self._path = path or (_DATA_DIR / "episodic.json")
        self._episodes = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._episodes = [Episode(**e) for e in raw]

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                [e.model_dump() for e in self._episodes],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def store(self, episode: Episode) -> str:
        """Store a new episode and return its id."""
        self._episodes.append(episode)
        self._save()
        return episode.id

    def get(self, episode_id: str) -> Episode | None:
        for ep in self._episodes:
            if ep.id == episode_id:
                return ep
        return None

    def search(self, query: str, top_k: int = 5) -> list[Episode]:
        """Return the *top_k* most similar episodes based on textual overlap.

        Uses ``SequenceMatcher`` on the concatenation of situation + strategy.
        In a production system you would swap this for an embedding search.
        """
        scored: list[tuple[float, Episode]] = []
        query_lower = query.lower()
        for ep in self._episodes:
            blob = f"{ep.situation} {ep.strategy_used} {ep.outcome}".lower()
            ratio = SequenceMatcher(None, query_lower, blob).ratio()
            scored.append((ratio, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def all_episodes(self) -> list[Episode]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory({len(self._episodes)} episodes)"


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

class Fact(BaseModel):
    """A single piece of long-term knowledge."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    concept: str
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = ""
    added_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SemanticMemory:
    """Long-term factual / conceptual knowledge store.

    Persists to ``data/memory/semantic.json``.
    """

    _path: Path
    _facts: list[Fact]

    def __init__(self, path: Path | None = None) -> None:
        _ensure_dir()
        self._path = path or (_DATA_DIR / "semantic.json")
        self._facts = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._facts = [Fact(**f) for f in raw]

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                [f.model_dump() for f in self._facts],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def store(self, fact: Fact) -> str:
        self._facts.append(fact)
        self._save()
        return fact.id

    def search(self, query: str, top_k: int = 5) -> list[Fact]:
        query_lower = query.lower()
        scored: list[tuple[float, Fact]] = []
        for f in self._facts:
            blob = f"{f.concept} {f.content}".lower()
            ratio = SequenceMatcher(None, query_lower, blob).ratio()
            scored.append((ratio, f))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def get_by_concept(self, concept: str) -> list[Fact]:
        concept_lower = concept.lower()
        return [f for f in self._facts if concept_lower in f.concept.lower()]

    def all_facts(self) -> list[Fact]:
        return list(self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"SemanticMemory({len(self._facts)} facts)"
