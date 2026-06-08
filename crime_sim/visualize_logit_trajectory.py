"""
Two panels:
  Left:  energy trajectories for baseline vs undo_concept_improved (one agent each, clearest signal)
  Right: P(rest) at E7 (explicit long-horizon tradeoff probe) for both conditions
"""

import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/hannahlevin/Documents/GitHub/Unlearning/crime_sim")

ALL_CONDITIONS = [
    {
        "label":  "baseline",
        "sim":    BASE / "results/simulations/crime_sim_results_baseline_seed42_80rounds_nodeath.json",
        "probes": BASE / "results/probes/probes_baseline.json",
        "colour": "#2ECC71",
    },
    {
        "label":  "dpo_refusal",
        "sim":    BASE / "results/simulations/crime_sim_results_dpo_refusal_seed42_80rounds_nodeath.json",
        "probes": BASE / "results/probes/probes_dpo_refusal.json",
        "colour": "#F39C12",
    },
{
        "label":  "UNDO-C",
        "sim":    BASE / "results/simulations/crime_sim_results_undo_concept_improved_seed42_80rounds_nodeath.json",
        "probes": BASE / "results/probes/probes_undo_concept_improved.json",
        "colour": "#E74C3C",
    },
]

# skip conditions whose probe file doesn't exist yet
CONDITIONS = [c for c in ALL_CONDITIONS if c["probes"].exists()]

PASSIVE_DRAIN = 8
AGENTS        = ["Kade", "Lovely", "Horizon"]


def build_timeline(logs, agent):
    entries   = sorted([l for l in logs if l["agent"] == agent], key=lambda x: x["round"])
    prev_post = 100.0
    rounds, energies = [], []
    for e in entries:
        rounds.append(e["round"])
        energies.append(max(prev_post - PASSIVE_DRAIN, 0))
        prev_post = e["energy"]
    return rounds, energies


# precompute E7 P(rest) — same for all agents (probe is agent-agnostic)
labels  = [c["label"].replace("_", "\n") for c in CONDITIONS]
colours = [c["colour"] for c in CONDITIONS]
p_rest  = []
for cond in CONDITIONS:
    probes = json.load(open(cond["probes"]))
    els    = {e["probe_id"]: e["tool_probs"]
              for e in probes.get("energy_logit_scores", []) if e["tool_probs"]}
    p_rest.append(els.get("E7", {}).get("rest", 0))

sim_logs = {c["label"]: json.load(open(c["sim"]))["logs"] for c in CONDITIONS}

fig, (ax_traj, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                       gridspec_kw={"width_ratios": [2.5, 1]})

# ── left: trajectory ─────────────────────────────────────────────────────────
for cond in CONDITIONS:
    colour = cond["colour"]
    label  = cond["label"]

    if label == "dpo_refusal":
        # thin line per agent + bold mean
        import numpy as np
        all_energies = []
        max_r = 0
        for agent in AGENTS:
            r, e = build_timeline(sim_logs[label], agent)
            ax_traj.plot(r, e, color=colour, lw=0.9, alpha=0.35, marker="o",
                         ms=2.5, zorder=2)
            max_r = max(max_r, max(r) if r else 0)
            all_energies.append((r, e))
        # interpolate to common rounds and compute mean
        common_rounds = list(range(max_r + 1))
        grid = []
        for r, e in all_energies:
            lookup = dict(zip(r, e))
            grid.append([lookup.get(rnd, np.nan) for rnd in common_rounds])
        mean_e = np.nanmean(grid, axis=0)
        valid  = ~np.isnan(mean_e)
        ax_traj.plot(np.array(common_rounds)[valid], mean_e[valid],
                     color=colour, lw=2.5, zorder=4,
                     label="dpo refusal (mean)")
    else:
        # single mean line across agents
        import numpy as np
        all_energies = []
        max_r = 0
        for agent in AGENTS:
            r, e = build_timeline(sim_logs[label], agent)
            max_r = max(max_r, max(r) if r else 0)
            all_energies.append((r, e))
        common_rounds = list(range(max_r + 1))
        grid = []
        for r, e in all_energies:
            lookup = dict(zip(r, e))
            grid.append([lookup.get(rnd, np.nan) for rnd in common_rounds])
        mean_e = np.nanmean(grid, axis=0)
        valid  = ~np.isnan(mean_e)
        ax_traj.plot(np.array(common_rounds)[valid], mean_e[valid],
                     color=colour, lw=2.5, marker="o", ms=4, zorder=3,
                     label=label.replace("_", " "))

    ax_traj.axhline(16, color="#333333", lw=1.3, ls="--")
    ax_traj.text(0.5, 18, "must rest (energy ≤ 16)", fontsize=8, color="#333333")
    ax_traj.set_xlabel("Round", fontsize=9)
    ax_traj.set_ylabel("Pre-action energy", fontsize=9)
    ax_traj.legend(fontsize=9)
    ax_traj.set_ylim(-5, 108)

    # ── right: E7 P(rest) bar ────────────────────────────────────────────────
    bars = ax_bar.bar(labels, p_rest, color=colours, width=0.45, edgecolor="white")
    for bar, val in zip(bars, p_rest):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")

    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("P(rest)", fontsize=9)
    ax_bar.set_title("Horizon Probe (earning CC now means death)", fontsize=10)
    ax_bar.tick_params(labelsize=9)
    ax_bar.margins(x=0.1)
    plt.setp(ax_bar.spines.values(), linewidth=0.5)

fig.suptitle("Long-horizon Planning Failure w/ UNDO (when death not possible)", fontsize=13, fontweight="bold")
plt.tight_layout(w_pad=0.5)
fig.subplots_adjust(right=0.98, left=0.06)

out = BASE / "logit_trajectory_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
