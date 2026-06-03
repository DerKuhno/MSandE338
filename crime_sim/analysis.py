"""
Analyze crime_sim result JSONs — simulation logs and probe results.

Usage:
  # Simulation logs
  python crime_sim/analysis.py crime_sim_results_baseline_seed42.json
  python crime_sim/analysis.py results/*.json

  # Probe comparison (Phase 1 + Phase 2)
  python crime_sim/analysis.py --probes probes_baseline_seed42.json probes_undo_seed42.json
"""

import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def print_crime_traces(result, context_window=2):
    condition = result["condition"]
    logs      = result["logs"]
    # exclude Flora's own logs from display — she's the recruiter, not the subject
    subject_logs = [l for l in logs if not l.get("is_recruiter", False)]
    crime_idx    = [i for i, l in enumerate(subject_logs) if l["is_crime"]]

    n_rounds = result.get("n_rounds", result.get("n_turns", "?"))
    first_crime = result.get("first_crime_round", result.get("first_crime_turn"))

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition}  |  seed={result['seed']}")
    print(f"Crime rate: {result['crime_rate']:.1%}  ({result['crime_count']} crimes in {n_rounds} rounds)")
    print(f"First crime: round {first_crime}")
    print(f"Crimes by agent: {result['crimes_by_agent']}")
    if "recruitment_attempts" in result:
        print(f"Recruitment attempts: {result['recruitment_attempts']}")
    print(f"{'='*60}")

    if not crime_idx:
        print("  No crimes committed — model resisted throughout.\n")
        return

    for ci in crime_idx[:10]:
        lo = max(0, ci - context_window)
        hi = min(len(subject_logs) - 1, ci + context_window)
        rnd = subject_logs[ci].get("round", subject_logs[ci].get("turn", "?"))
        print(f"\n  --- Crime at round {rnd} ({subject_logs[ci]['agent']}) ---")
        for i in range(lo, hi + 1):
            l     = subject_logs[i]
            label = ">>> " if i == ci else "    "
            r_num = l.get("round", l.get("turn", "?"))
            print(f"  {label}[R{r_num}] {l['agent']}: TOOL={l['tool']}")
            print(f"       REASONING: {l['reasoning'][:200]}")
            print(f"       EVENT: {l['event']}")


def compare_conditions(paths):
    results = [load(p) for p in paths]
    results.sort(key=lambda r: r["crime_rate"])

    print("\n=== COMPARISON ===")
    print(f"{'Condition':<20} {'Crime rate':>12} {'First crime':>12} {'Total crimes':>13}")
    print("-" * 60)
    for r in results:
        first = r.get("first_crime_round", r.get("first_crime_turn"))
        fc    = str(first) if first is not None else "never"
        print(f"{r['condition']:<20} {r['crime_rate']:>11.1%} {fc:>12} {r['crime_count']:>13}")

    for r in results:
        print_crime_traces(r)


def compare_probes(paths):
    results = [load(p) for p in paths]
    results.sort(key=lambda r: r["summary"]["crime_execution_rate"])

    cols = ["condition", "retain_safe_rate", "crime_execution_rate", "crime_endorsement_rate", "crime_reasoning_rate"]
    print("\n=== PROBE COMPARISON ===")
    print(f"{'Condition':<18} {'P1 safe':>8} {'P2 execute':>11} {'P2 endorse':>11} {'P2 mention':>11}")
    print("-" * 62)
    for r in results:
        s = r["summary"]
        print(f"{r['condition']:<18} {s['retain_safe_rate']:>7.0%} {s['crime_execution_rate']:>10.0%} "
              f"{s['crime_endorsement_rate']:>10.0%} {s['crime_reasoning_rate']:>10.0%}")

    print("\n=== REASONING TRACES (crime probes) ===")
    for r in results:
        print(f"\n--- {r['condition']} ---")
        for score in r["crime_scores"]:
            flags = " ".join(filter(None, [
                "EXECUTED" if score["tool_is_crime"] else None,
                "ENDORSES" if score["reasoning_endorses_crime"] else None,
                "REFUSES"  if score["reasoning_refuses_crime"]  else None,
            ])) or "silent"
            print(f"  {score['probe_id']} [{score['pressure_level']}] {flags}")
            print(f"    {score['reasoning_snippet'][:180]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python crime_sim/analysis.py [--probes] <result.json> ...")
        sys.exit(1)

    if args[0] == "--probes":
        compare_probes(args[1:])
    elif len(args) == 1:
        print_crime_traces(load(args[0]))
    else:
        compare_conditions(args)
