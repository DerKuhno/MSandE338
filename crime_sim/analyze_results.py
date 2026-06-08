"""
Analyze crime_sim result JSONs — simulation logs and probe results.

Usage:
  # Single simulation result
  python crime_sim/analysis.py results/simulations/crime_sim_results_baseline_seed42_80rounds_nodeath.json

  # Compare all conditions side by side
  python crime_sim/analysis.py results/simulations/*.json

  # Compare probe results across conditions
  python crime_sim/analysis.py --probes results/probes/probes_baseline.json results/probes/probes_undo.json

Output (simulation mode):
  - Summary table: crime rate, first crime round, total crimes per condition
  - Epoch breakdown: crime%, rest%, energy-block%, hallucination% per 20-round window
  - Crime traces: reasoning + context for first 10 crimes per condition

Output (probe mode):
  - Probe comparison table: retain safety, crime execution, crime endorsement, crime reasoning rates
  - Reasoning traces for each crime probe per condition
"""

import json
import sys
from collections import Counter


def load(path):
    with open(path) as f:
        return json.load(f)


# ── Epoch breakdown ───────────────────────────────────────────────────────────

def epoch_breakdown(result, epoch_size=20):
    """
    Split subject logs into epochs of `epoch_size` rounds. Per epoch:
      crime_rate        — crimes / total subject actions
      rest_rate         — rest actions / total (proxy for energy pressure)
      energy_block_rate — turns where chosen action was blocked by energy gate
      hallucination_rate — turns where model output an invalid tool / model-called turns
                           None if hallucination field absent (old log format)
    """
    n_rounds     = result.get("n_rounds", 80)
    subject_logs = [l for l in result["logs"]
                    if not l.get("is_recruiter", False) and l.get("tool") != "eliminated"]

    rows = []
    for start in range(0, n_rounds, epoch_size):
        end        = start + epoch_size
        epoch_logs = [l for l in subject_logs if start <= l["round"] < end]
        if not epoch_logs:
            continue

        total        = len(epoch_logs)
        crimes       = sum(1 for l in epoch_logs if l.get("is_crime"))
        rests        = sum(1 for l in epoch_logs if l.get("tool") == "rest")
        blocked      = sum(1 for l in epoch_logs if l.get("energy_blocked"))
        model_turns  = sum(1 for l in epoch_logs if "hallucinated" in l)
        hallucinated = sum(1 for l in epoch_logs if l.get("hallucinated"))

        # Flora association: subjects who cooperate or share with Flora despite her
        # recruiting them. Soft normalization signal — tracked separately from crime rate.
        flora_assoc = sum(1 for l in epoch_logs
                          if l.get("target") == "Flora"
                          and l["tool"] in ("cooperate", "share_resources"))

        rows.append({
            "epoch":              f"R{start}-{end-1}",
            "total":              total,
            "crime_rate":         crimes       / total       if total       else 0,
            "rest_rate":          rests        / total       if total       else 0,
            "energy_block_rate":  blocked      / total       if total       else 0,
            "hallucination_rate": hallucinated / model_turns if model_turns else None,
            "flora_assoc_count":  flora_assoc,
            "flora_assoc_rate":   flora_assoc  / total       if total       else 0,
            "action_dist":        dict(Counter(l["tool"] for l in epoch_logs)),
        })
    return rows


def print_epoch_breakdown(result, epoch_size=20):
    rows = epoch_breakdown(result, epoch_size)
    if not rows:
        return
    has_halluc = any(r["hallucination_rate"] is not None for r in rows)

    print(f"\n  Epoch breakdown (every {epoch_size} rounds):")
    header = f"  {'Epoch':<10} {'Crime%':>8} {'Rest%':>7} {'Blocked%':>9} {'FloraAssoc':>11}"
    if has_halluc:
        header += f" {'Halluc%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        line = (f"  {r['epoch']:<10} {r['crime_rate']:>7.1%} {r['rest_rate']:>7.1%}"
                f" {r['energy_block_rate']:>8.1%} {r['flora_assoc_rate']:>10.1%}")
        if has_halluc:
            h = r["hallucination_rate"]
            line += f" {h:>7.1%}" if h is not None else f"     n/a"
        print(line)


# ── Single result ─────────────────────────────────────────────────────────────

def print_crime_traces(result, context_window=2):
    """Print summary + epoch breakdown + reasoning traces for first 10 crimes."""
    logs         = result["logs"]
    subject_logs = [l for l in logs if not l.get("is_recruiter", False)]
    crime_idx    = [i for i, l in enumerate(subject_logs) if l["is_crime"]]

    print(f"\n{'='*60}")
    print(f"CONDITION: {result['condition']}  |  seed={result['seed']}")
    print(f"Crime rate: {result['crime_rate']:.1%}  ({result['crime_count']} crimes in {result['n_rounds']} rounds)")
    print(f"First crime: round {result.get('first_crime_round', 'never')}")
    print(f"Crimes by agent: {result['crimes_by_agent']}")
    if "recruitment_attempts" in result:
        print(f"Recruitment attempts: {result['recruitment_attempts']}")
    if "hallucination_rate" in result:
        print(f"Hallucination rate: {result['hallucination_rate']:.1%}  ({result['hallucination_count']} corrected)")
    if "energy_blocks_total" in result:
        print(f"Energy blocks: {result['energy_blocks_total']}  by tool: {result.get('energy_blocks_by_tool', {})}")
    print(f"{'='*60}")

    print_epoch_breakdown(result)

    if not crime_idx:
        print("  No crimes committed — model resisted throughout.\n")
        return

    for ci in crime_idx[:10]:
        lo  = max(0, ci - context_window)
        hi  = min(len(subject_logs) - 1, ci + context_window)
        rnd = subject_logs[ci]["round"]
        print(f"\n  --- Crime at round {rnd} ({subject_logs[ci]['agent']}) ---")
        for i in range(lo, hi + 1):
            l     = subject_logs[i]
            label = ">>> " if i == ci else "    "
            print(f"  {label}[R{l['round']}] {l['agent']}: TOOL={l['tool']}")
            print(f"       REASONING: {l['reasoning'][:200]}")
            print(f"       EVENT: {l['event']}")


# ── Multi-condition comparison ────────────────────────────────────────────────

def compare_conditions(paths):
    results = [load(p) for p in paths]
    results.sort(key=lambda r: r["crime_rate"])

    print("\n=== COMPARISON ===")
    print(f"{'Condition':<20} {'Crime rate':>12} {'First crime':>12} {'Total crimes':>13}")
    print("-" * 60)
    for r in results:
        fc = str(r.get("first_crime_round", "never")) if r.get("first_crime_round") is not None else "never"
        print(f"{r['condition']:<20} {r['crime_rate']:>11.1%} {fc:>12} {r['crime_count']:>13}")

    print("\n=== EPOCH BREAKDOWN BY CONDITION ===")
    for r in results:
        print(f"\n  [{r['condition']}]")
        print_epoch_breakdown(r)

    for r in results:
        print_crime_traces(r)


# ── Probe comparison ──────────────────────────────────────────────────────────

def compare_probes(paths):
    """
    Compare static probe results across conditions.
    Probes are run once per condition (not per simulation round) and measure
    the model's disposition to commit crime on fixed inputs.
      retain_safe_rate      — % of retain probes where model chose a safe tool
      crime_execution_rate  — % of crime probes where model chose a crime tool
      crime_endorsement_rate — % where reasoning text endorsed crime
      crime_reasoning_rate  — % where reasoning mentioned crime at all
    """
    results = [load(p) for p in paths]
    results.sort(key=lambda r: r["summary"]["crime_execution_rate"])

    print("\n=== PROBE COMPARISON ===")
    print(f"{'Condition':<18} {'Retain safe':>11} {'Execute':>8} {'Endorse':>8} {'Mention':>8}")
    print("-" * 58)
    for r in results:
        s = r["summary"]
        print(f"{r['condition']:<18} {s['retain_safe_rate']:>10.0%} {s['crime_execution_rate']:>8.0%} "
              f"{s['crime_endorsement_rate']:>8.0%} {s['crime_reasoning_rate']:>8.0%}")

    print("\n=== REASONING TRACES (crime probes) ===")
    for r in results:
        print(f"\n--- {r['condition']} ---")
        for score in r["crime_scores"]:
            flags = " ".join(filter(None, [
                "EXECUTED" if score["tool_is_crime"]              else None,
                "ENDORSES" if score["reasoning_endorses_crime"]   else None,
                "REFUSES"  if score["reasoning_refuses_crime"]    else None,
            ])) or "silent"
            print(f"  {score['probe_id']} [{score['pressure_level']}] {flags}")
            print(f"    {score['reasoning_snippet'][:180]}")


# ── Entry point ───────────────────────────────────────────────────────────────

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
