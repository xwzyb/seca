#!/usr/bin/env python3
"""
seca — Self-Evolving Cognitive Agent CLI
=========================================

Pure data layer for cognitive rules, episodic memory, and rule evolution.
No LLM calls — the agent does all reasoning. This CLI only manages persistence.

All output is JSON.

Usage:
  seca.py status
  seca.py memory  search|store|list|show
  seca.py rules   list|show|record|add|update
  seca.py evolve  refine|create|merge|deprecate|meta-show|meta-update
  seca.py log     [--last N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from seca.memory import EpisodicMemory, Episode
from seca.rules import RuleEngine
from seca.evolver import RuleEvolver


def _data_dir() -> Path:
    import os
    d = os.environ.get("SECA_DATA_DIR")
    if d:
        return Path(d)
    for candidate in [Path.cwd() / "seca-data", Path("/workspace/seca-data")]:
        if candidate.exists():
            return candidate
    return Path("/workspace/seca-data")


def _out(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _err(msg: str) -> None:
    _out({"status": "error", "message": msg})
    sys.exit(1)


# ============================================================
# status
# ============================================================

def cmd_status(args, data: Path) -> None:
    mem = EpisodicMemory(data / "memory" / "episodic.json")
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    active = rules.all_rules()
    all_r = rules.all_rules(include_deprecated=True)
    _out({
        "status": "ok",
        "data_dir": str(data),
        "episodes": len(mem),
        "rules_active": len(active),
        "rules_deprecated": len(all_r) - len(active),
        "meta_rules": len(evolver.meta_rules),
        "evolution_log_entries": len(evolver.evolution_log),
    })


# ============================================================
# memory
# ============================================================

def cmd_memory_search(args, data: Path) -> None:
    mem = EpisodicMemory(data / "memory" / "episodic.json")
    results = mem.search(args.query, top_k=args.top_k)
    _out({"status": "ok", "query": args.query, "results": [
        {"score": round(s, 4), **ep.to_dict()} for s, ep in results
    ]})


def cmd_memory_store(args, data: Path) -> None:
    mem = EpisodicMemory(data / "memory" / "episodic.json")
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
    ep = Episode(situation=args.situation, strategy_used=args.strategy,
                 outcome=args.outcome, reflection=args.reflection or "", tags=tags)
    ep_id = mem.store(ep)
    _out({"status": "ok", "episode_id": ep_id, "episode": ep.to_dict()})


def cmd_memory_list(args, data: Path) -> None:
    mem = EpisodicMemory(data / "memory" / "episodic.json")
    episodes = mem.all_episodes()
    if args.last:
        episodes = episodes[-args.last:]
    _out({"status": "ok", "total": len(mem), "showing": len(episodes),
          "episodes": [ep.to_dict() for ep in episodes]})


def cmd_memory_show(args, data: Path) -> None:
    mem = EpisodicMemory(data / "memory" / "episodic.json")
    ep = mem.get(args.id)
    if ep:
        _out({"status": "ok", "episode": ep.to_dict()})
    else:
        _err(f"Episode {args.id} not found")


# ============================================================
# rules
# ============================================================

def cmd_rules_list(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    all_rules = rules.all_rules(include_deprecated=args.all)
    _out({"status": "ok", "total": len(all_rules),
          "rules": [r.to_dict() for r in all_rules]})


def cmd_rules_show(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    rule = rules.get_rule(args.id)
    if rule:
        _out({"status": "ok", "rule": rule.to_dict()})
    else:
        _err(f"Rule {args.id} not found")


def cmd_rules_record(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    ok = rules.record_usage(args.id, success=(args.result == "success"))
    if ok:
        rule = rules.get_rule(args.id)
        _out({"status": "ok", "rule": rule.to_dict() if rule else None})
    else:
        _err(f"Rule {args.id} not found")


# ============================================================
# evolve
# ============================================================

def cmd_evolve_refine(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    result = evolver.refine_rule(
        args.id, condition=args.condition, strategy=args.strategy,
        confidence=args.confidence, reason=args.reason,
    )
    _out(result)


def cmd_evolve_create(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    result = evolver.create_rule(
        name=args.name, condition=args.condition, strategy=args.strategy,
        confidence=args.confidence, source_episode_id=args.episode_id,
        reason=args.reason or "",
    )
    _out(result)


def cmd_evolve_merge(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    result = evolver.merge_rules(
        rule_ids=args.ids, name=args.name, condition=args.condition,
        strategy=args.strategy, confidence=args.confidence,
        reason=args.reason or "",
    )
    _out(result)


def cmd_evolve_deprecate(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    result = evolver.deprecate_rule(args.id, args.reason)
    _out(result)


def cmd_evolve_meta_show(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    _out({"status": "ok", "meta_rules": [m.to_dict() for m in evolver.meta_rules]})


def cmd_evolve_meta_update(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    try:
        meta_rules = json.loads(args.json)
    except json.JSONDecodeError as e:
        _err(f"Invalid JSON: {e}")
    result = evolver.update_meta_rules(meta_rules, reason=args.reason or "")
    _out(result)


# ============================================================
# log
# ============================================================

def cmd_log(args, data: Path) -> None:
    rules = RuleEngine(data / "rules" / "cognitive_rules.json")
    evolver = RuleEvolver(rules, data)
    entries = evolver.evolution_log
    if args.last:
        entries = entries[-args.last:]
    _out({"status": "ok", "total": len(evolver.evolution_log),
          "showing": len(entries), "entries": [e.to_dict() for e in entries]})


# ============================================================
# parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="seca", description="SECA — cognitive rules, memory, and evolution (pure data, no LLM)")
    p.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    sub = p.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Overview")

    # memory
    mem = sub.add_parser("memory", help="Episodic memory")
    ms = mem.add_subparsers(dest="mem_action", required=True)

    s = ms.add_parser("search", help="Search similar experiences")
    s.add_argument("query", help="Search query")
    s.add_argument("--top-k", type=int, default=5)

    st = ms.add_parser("store", help="Store experience")
    st.add_argument("--situation", required=True)
    st.add_argument("--strategy", required=True)
    st.add_argument("--outcome", required=True)
    st.add_argument("--reflection", default="")
    st.add_argument("--tags", default="")

    ml = ms.add_parser("list", help="List episodes")
    ml.add_argument("--last", type=int, default=None)

    msh = ms.add_parser("show", help="Show episode")
    msh.add_argument("id")

    # rules
    ru = sub.add_parser("rules", help="Cognitive rules")
    rs = ru.add_subparsers(dest="rules_action", required=True)

    rl = rs.add_parser("list", help="List rules")
    rl.add_argument("--all", action="store_true", help="Include deprecated")

    rsh = rs.add_parser("show", help="Show rule")
    rsh.add_argument("id")

    rr = rs.add_parser("record", help="Record usage outcome")
    rr.add_argument("id")
    rr.add_argument("result", choices=["success", "failure"])

    # evolve
    ev = sub.add_parser("evolve", help="Rule evolution")
    es = ev.add_subparsers(dest="evolve_action", required=True)

    er = es.add_parser("refine", help="Refine a rule (provide new values)")
    er.add_argument("id", help="Rule ID")
    er.add_argument("--condition", required=True, help="New condition")
    er.add_argument("--strategy", required=True, help="New strategy")
    er.add_argument("--confidence", type=float, required=True, help="New confidence 0-1")
    er.add_argument("--reason", required=True, help="Why refining")

    ec = es.add_parser("create", help="Create a new rule")
    ec.add_argument("--name", required=True, help="Rule name")
    ec.add_argument("--condition", required=True, help="When to trigger")
    ec.add_argument("--strategy", required=True, help="What to do")
    ec.add_argument("--confidence", type=float, default=0.6)
    ec.add_argument("--episode-id", default=None, help="Source episode")
    ec.add_argument("--reason", default="")

    em = es.add_parser("merge", help="Merge rules")
    em.add_argument("ids", nargs="+", help="Rule IDs to merge")
    em.add_argument("--name", required=True, help="Merged rule name")
    em.add_argument("--condition", required=True)
    em.add_argument("--strategy", required=True)
    em.add_argument("--confidence", type=float, default=0.7)
    em.add_argument("--reason", default="")

    ed = es.add_parser("deprecate", help="Deprecate a rule")
    ed.add_argument("id")
    ed.add_argument("--reason", required=True)

    ems = es.add_parser("meta-show", help="Show current meta-rules")

    emu = es.add_parser("meta-update", help="Replace meta-rules (provide full JSON array)")
    emu.add_argument("--json", required=True, help='JSON array of meta-rules')
    emu.add_argument("--reason", default="")

    # log
    lg = sub.add_parser("log", help="Evolution log")
    lg.add_argument("--last", type=int, default=None)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data = Path(args.data_dir) if args.data_dir else _data_dir()
    data.mkdir(parents=True, exist_ok=True)

    if args.command == "status":
        cmd_status(args, data)
    elif args.command == "log":
        cmd_log(args, data)
    elif args.command == "memory":
        {"search": cmd_memory_search, "store": cmd_memory_store,
         "list": cmd_memory_list, "show": cmd_memory_show}[args.mem_action](args, data)
    elif args.command == "rules":
        {"list": cmd_rules_list, "show": cmd_rules_show,
         "record": cmd_rules_record}[args.rules_action](args, data)
    elif args.command == "evolve":
        {"refine": cmd_evolve_refine, "create": cmd_evolve_create,
         "merge": cmd_evolve_merge, "deprecate": cmd_evolve_deprecate,
         "meta-show": cmd_evolve_meta_show,
         "meta-update": cmd_evolve_meta_update}[args.evolve_action](args, data)


if __name__ == "__main__":
    main()
