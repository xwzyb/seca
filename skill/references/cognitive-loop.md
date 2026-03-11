# Cognitive Loop — Detailed Walkthrough

The full SECA cognitive loop for complex tasks. Use this when the lightweight pattern
(search → match → reason → store) isn't enough.

## The Loop

```
┌─→ Task Received
│   │
│   ├─ 1. RETRIEVE — search episodic memory for similar situations
│   │   └─ seca memory search "<task>" --top-k 5
│   │
│   ├─ 2. MATCH — find applicable cognitive rules
│   │   └─ seca rules match "<task>"
│   │   └─ If rule_gap=true → learning opportunity (step 7)
│   │
│   ├─ 3. PLAN — combine retrieved experiences + matched rules
│   │   └─ From experiences: what worked/failed before?
│   │   └─ From rules: what strategy does the rule prescribe?
│   │   └─ Synthesize into an execution plan
│   │
│   ├─ 4. EXECUTE — carry out the plan (this is YOUR reasoning, not SECA's)
│   │   └─ Follow the matched rule's strategy steps
│   │   └─ Adapt based on retrieved experiences
│   │   └─ Apply your own judgment for gaps
│   │
│   ├─ 5. EVALUATE — assess the outcome
│   │   └─ Did the strategy work?
│   │   └─ What was the confidence level?
│   │   └─ Were there unexpected issues?
│   │
│   ├─ 6. RECORD — store the experience
│   │   └─ seca memory store --situation "..." --strategy "..." --outcome "..." --reflection "..."
│   │   └─ seca rules record <rule_id> success|failure
│   │
│   └─ 7. EVOLVE (if warranted)
│       ├─ Strategy failed → seca evolve refine <rule_id> --feedback "..."
│       ├─ No rule matched → seca evolve create --situation "..." --strategy "..."
│       ├─ Rule consistently poor → seca evolve deprecate <rule_id> --reason "..."
│       └─ Periodic check → seca evolve meta
│
└── Next task
```

## Decision: When to go deep?

Not every task needs the full loop. Use this heuristic:

| Signal | Action |
|--------|--------|
| Simple factual question | Skip SECA entirely |
| Routine task, high confidence | Lightweight: search + match only |
| Complex/novel problem | Full loop: retrieve → match → plan → execute → evaluate → record |
| Strategy failed | Full loop + evolve step |
| Every ~10 complex tasks | Add a periodic `evolve meta` check |

## Example: Full loop on a contradiction task

```
Task: "Report A says revenue is $10M, Report B says $8M"

1. RETRIEVE:
   $ seca memory search "contradictory financial reports"
   → Found: past experience resolving budget contradictions (score: 0.34)
   → Past strategy: "check dates, prefer newer source, verify with third source"

2. MATCH:
   $ seca rules match "Input contains contradictory revenue figures from two reports"
   → Matched: rule_contradictions (confidence: 0.85)
   → Strategy: list contradictions → evaluate sources → propose resolution → note uncertainty

3. PLAN:
   Combine rule strategy + past experience:
   - List: Revenue discrepancy ($10M vs $8M)
   - Check report dates and source authority
   - Look for a third data point
   - Past experience says: prefer newer source

4. EXECUTE:
   [Your reasoning here, following the plan]

5. EVALUATE:
   Strategy worked. Found that Report B was audited (more reliable).
   Confidence: 85%.

6. RECORD:
   $ seca memory store \
     --situation "Two reports with conflicting revenue figures" \
     --strategy "Checked source dates and authority, preferred audited report" \
     --outcome "Identified Report B as authoritative, resolved discrepancy" \
     --reflection "Audited sources take precedence over internal estimates"
   $ seca rules record rule_contradictions success

7. EVOLVE:
   No evolution needed — rule worked well this time.
```

## Anti-patterns

- **Over-engineering**: Don't run the full loop for "what's 2+2". Match the depth to the task.
- **Storing everything**: Only store experiences you'd want to retrieve later. Quality > quantity.
- **Evolving too eagerly**: One failure doesn't mean a rule is bad. Wait for patterns (3+ uses).
- **Ignoring rule gaps**: When `rules match` returns `rule_gap: true`, that's a signal. Consider creating a rule after you solve the task.
