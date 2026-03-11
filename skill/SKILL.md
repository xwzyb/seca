---
name: seca
description: >
  Self-Evolving Cognitive Agent — cognitive enhancement framework with self-evolving
  rules, episodic memory, and meta-cognitive monitoring. Pure data layer: no extra LLM
  calls, no API keys needed — agent does all reasoning. Use when: (1) facing complex
  problems requiring structured reasoning strategies, (2) wanting to learn from past
  experiences and reuse strategies, (3) needing to self-monitor and improve reasoning
  quality over time, (4) handling contradictions, ambiguity, or analogical transfer.
  NOT for: simple factual Q&A, single-step tasks, or tasks that don't benefit from
  structured reasoning.
---

# SECA — Self-Evolving Cognitive Agent

Cognitive infrastructure that makes you smarter over time. **You do all the reasoning** —
SECA manages the data: persistent rules, searchable experiences, and evolution audit trails.

Zero dependencies beyond Python 3.10. No LLM calls. No API keys.

## Core Concepts

Three layers, all human-readable JSON:

- **Rules** — "when X happens, do Y" strategies with confidence scores and usage stats
- **Experiences** — situation/strategy/outcome/reflection records, searchable by similarity
- **Meta-rules** — rules about when to evolve rules (the Strange Loop 🌀)

Ships with 5 default rules (contradiction resolution, task clarification, decomposition,
self-verification, analogical transfer) + 4 meta-rules.

## CLI Reference

All commands output JSON. Script: `scripts/seca.py`

### Status

```bash
python3 scripts/seca.py status
```

### Memory (experiences)

```bash
# Search similar past experiences
python3 scripts/seca.py memory search "conflicting requirements" --top-k 3

# Store a new experience
python3 scripts/seca.py memory store \
  --situation "User asked for X but constraints make X impossible" \
  --strategy "Identified core need behind X, proposed alternative Y" \
  --outcome "User accepted Y" \
  --reflection "Always dig for the underlying need" \
  --tags "contradiction,negotiation"

# List / show
python3 scripts/seca.py memory list --last 10
python3 scripts/seca.py memory show <episode-id>
```

### Rules

```bash
# List all active rules (add --all for deprecated too)
python3 scripts/seca.py rules list

# Show single rule
python3 scripts/seca.py rules show rule_contradictions

# Record usage outcome (updates success rate)
python3 scripts/seca.py rules record rule_contradictions success
python3 scripts/seca.py rules record rule_decomposition failure
```

**Matching rules to a situation**: Read `rules list` output, then decide yourself which
rules apply based on their conditions. No separate match command — you're the LLM.

### Evolution

All evolution commands take explicit values — you reason about *what* to change,
the CLI just persists it and logs the mutation.

```bash
# Refine: provide the improved condition/strategy/confidence
python3 scripts/seca.py evolve refine rule_contradictions \
  --condition "When input has contradictory info from 2+ sources" \
  --strategy "1. List contradictions. 2. Check source dates/authority. 3. Cross-validate. 4. Resolve with confidence level." \
  --confidence 0.88 \
  --reason "Added cross-validation step for multi-source contradictions"

# Create: define a brand new rule from what you learned
python3 scripts/seca.py evolve create \
  --name "API Data Reconciliation" \
  --condition "When multiple APIs return inconsistent data" \
  --strategy "Compare timestamps, prefer newest, flag discrepancies" \
  --confidence 0.7 \
  --episode-id abc123 \
  --reason "Learned from API inconsistency incident"

# Merge: combine similar rules (provide the merged result)
python3 scripts/seca.py evolve merge rule_001 rule_002 \
  --name "Unified Conflict Resolution" \
  --condition "When any inputs conflict" \
  --strategy "1. Enumerate conflicts. 2. Rank sources. 3. Resolve." \
  --confidence 0.8

# Deprecate
python3 scripts/seca.py evolve deprecate rule_old \
  --reason "Success rate below 20% after 10 uses"

# Show current meta-rules
python3 scripts/seca.py evolve meta-show

# Update meta-rules (provide full JSON array)
python3 scripts/seca.py evolve meta-update \
  --json '[{"id":"meta_low_success","name":"Refine on low success","condition":"...","action":"refine_rule","threshold":0.35}]' \
  --reason "Lowered threshold after observing premature refinements"
```

### Evolution Log

```bash
python3 scripts/seca.py log --last 5
```

## Usage Patterns

### Lightweight: Consult before reasoning

1. `memory search "<task>"` — any relevant past experience?
2. `rules list` — scan rules, decide which apply
3. Reason using retrieved context + matched rules
4. `memory store ...` — record the experience
5. `rules record <id> success|failure` — update stats

### Learning: After a notable outcome

1. `memory store ...` — capture the experience
2. If strategy was novel → `evolve create ...`
3. If existing rule was inadequate → `evolve refine <id> ...`
4. If rule keeps failing → `evolve deprecate <id> ...`

### Deep: Periodic self-improvement

Review rules and evolve. See [references/cognitive-loop.md](references/cognitive-loop.md).
For meta-rules, see [references/meta-rules.md](references/meta-rules.md).

## Data Storage

Persists under `seca-data/` in workspace (override via `SECA_DATA_DIR` env):

```
seca-data/
├── memory/episodic.json
├── rules/cognitive_rules.json
├── rules/meta_rules.json
└── rules/evolution_log.json
```

All JSON. Read files directly when you need bulk context.
