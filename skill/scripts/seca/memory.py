"""
seca — Episodic memory (experience store)
==========================================
Stores situation-strategy-outcome-reflection episodes.
Retrieval via keyword overlap (SequenceMatcher for now, embedding-ready).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


class Episode:
    """One recorded experience."""

    __slots__ = ("id", "timestamp", "situation", "strategy_used", "outcome", "reflection", "tags")

    def __init__(
        self,
        situation: str,
        strategy_used: str,
        outcome: str,
        reflection: str = "",
        tags: list[str] | None = None,
        id: str | None = None,
        timestamp: str | None = None,
    ):
        self.id = id or uuid.uuid4().hex[:12]
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.situation = situation
        self.strategy_used = strategy_used
        self.outcome = outcome
        self.reflection = reflection
        self.tags = tags or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "situation": self.situation,
            "strategy_used": self.strategy_used,
            "outcome": self.outcome,
            "reflection": self.reflection,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        return cls(**d)


class EpisodicMemory:
    """Experience memory with similarity retrieval. Persists to JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._episodes: list[Episode] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._episodes = [Episode.from_dict(e) for e in raw]

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([e.to_dict() for e in self._episodes], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def store(self, episode: Episode) -> str:
        self._episodes.append(episode)
        self._save()
        return episode.id

    def get(self, episode_id: str) -> Episode | None:
        for ep in self._episodes:
            if ep.id == episode_id:
                return ep
        return None

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Episode]]:
        """Return (score, episode) pairs sorted by similarity, descending."""
        query_lower = query.lower()
        scored: list[tuple[float, Episode]] = []
        for ep in self._episodes:
            blob = f"{ep.situation} {ep.strategy_used} {ep.outcome}".lower()
            ratio = SequenceMatcher(None, query_lower, blob).ratio()
            scored.append((ratio, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:top_k]

    def all_episodes(self) -> list[Episode]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)
