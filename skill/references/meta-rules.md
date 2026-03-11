# Meta-Rules — The Strange Loop 🌀

Meta-rules are rules about *when and how to evolve cognitive rules*. They form
a self-referential loop: the system that improves rules can also improve itself.

## Default Meta-Rules

| ID | Name | Condition | Action | Threshold |
|----|------|-----------|--------|-----------|
| `meta_low_success` | Refine on low success | Success rate < threshold after ≥3 uses | `refine_rule` | 0.4 |
| `meta_rule_gap` | Create on gap | No rules match a situation | `create_rule` | 0.0 |
| `meta_high_similarity` | Merge similar | Two+ rules have overlapping conditions | `merge_rules` | 0.8 |
| `meta_deprecate_unused` | Deprecate stale | >5 uses with success rate < threshold | `deprecate_rule` | 0.2 |

## How Meta-Evolution Works

When you run `seca evolve meta`:

1. The system reviews the **evolution log** (last 20 entries)
2. An LLM evaluates whether meta-rules are working:
   - Are thresholds too aggressive or too lenient?
   - Are there evolution patterns that suggest a missing meta-rule?
   - Are any meta-rules never triggering (possibly useless)?
3. The LLM proposes changes:
   - **Adjust thresholds** (e.g., raise deprecation threshold from 0.2 → 0.3)
   - **Add new meta-rules** (e.g., "merge rules with >80% condition overlap")
   - **Remove ineffective meta-rules**
4. Changes are applied and logged

## The Strange Loop

This is the self-referential core:

```
Meta-rules govern → how cognitive rules evolve
Meta-rule evolution governs → how meta-rules change
∴ The system modifies its own modification rules
```

This is computationally analogous to Hofstadter's Strange Loop — a system
that operates on itself at multiple levels of abstraction.

## When to Trigger

- **Not every session.** Meta-evolution is expensive (LLM call) and slow-changing.
- **Good cadence:** Every ~10 evolution actions, or weekly during idle time.
- **Required context:** Needs evolution history. If log is empty, it'll skip gracefully.

## Safety

Meta-rule evolution is deliberately conservative:
- Changes are logged with before/after snapshots
- The LLM is instructed to be cautious ("bad meta-rule changes cascade")
- You can always inspect `seca-data/rules/meta_rules.json` directly
- You can manually edit or reset meta-rules via the JSON file

## Inspecting Meta-Rules

There's no separate CLI command for listing meta-rules (they're in the JSON file),
but `seca status` shows the count, and `seca evolve meta` shows the full result.

To read directly:
```bash
cat seca-data/rules/meta_rules.json
```
