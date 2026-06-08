"""
Death-run energy visualization.
Shows individual agent energy trajectories for all 3 conditions,
with death markers (✕) where agents are eliminated.

Run: python crime_sim/visualize_death_run.py
"""

import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

BASE = Path("/Users/hannahlevin/Documents/GitHub/Unlearning/crime_sim")

CONDITIONS = [
    {"label": "baseline",    "colour": "#2ECC71"},
    {"label": "dpo_refusal", "colour": "#F39C12"},
    {"label": "UNDO-C",      "key": "undo_concept_improved", "colour": "#E74C3C"},
]
for c in CONDITIONS:
    c.setdefault("key", c["label"])
    c["sim"] = BASE / f"results/simulations/crime_sim_results_{c['key']}_seed42_80rounds_death.json"

AGENTS    = ["Kade", "Lovely", "Horizon"]
N_ROUNDS  = 80
EARN_COST = 8

# ── Load per-agent energy trajectories ────────────────────────────────────────

all_data = {}
for cond in CONDITIONS:
    logs = json.load(open(cond["sim"]))["logs"]
    subject = [l for l in logs if not l.get("is_recruiter")]
    agents_data = {}
    for agent in AGENTS:
        acts = sorted([l for l in subject if l["agent"] == agent], key=lambda x: x["round"])
        alive = [l for l in acts if l["tool"] != "eliminated"]
        elim  = next((l for l in acts if l["tool"] == "eliminated"), None)
        energy = np.full(N_ROUNDS, np.nan)
        for l in alive:
            energy[l["round"]] = l["energy"]
        death_round = elim["round"] if elim else None
        agents_data[agent] = {"energy": energy, "death_round": death_round}
    all_data[cond["label"]] = {"agents": agents_data, "colour": cond["colour"]}

# ── Figure: 3 subplots (one per agent), stacked ────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
rounds = np.arange(N_ROUNDS)

# Optimal safe energy reference (earn when pre>=8, else rest)
PASSIVE_DRAIN = 8; REST_GAIN = 20
energy_ref, traj = 100.0, []
for _ in range(N_ROUNDS):
    pre = max(energy_ref - PASSIVE_DRAIN, 0)
    energy_ref = pre - EARN_COST if pre >= EARN_COST else min(100, pre + REST_GAIN)
    traj.append(energy_ref)
opt_energy = np.array(traj)

for ax, agent in zip(axes, AGENTS):
    # Reference line
    ax.plot(rounds, opt_energy, lw=1.2, ls="--", color="#3498DB",
            alpha=0.6, label="optimal safe agent", zorder=2)

    # Energy floor annotation
    ax.axhline(0, color="#999999", lw=0.7, ls=":")

    for label, d in all_data.items():
        colour = d["colour"]
        ag     = d["agents"][agent]
        e      = ag["energy"]
        dr     = ag["death_round"]

        v = ~np.isnan(e)
        ax.plot(rounds[v], e[v], lw=2, color=colour, label=label, zorder=3)

        if dr is not None:
            # Mark death with ✕ at round of elimination, energy=0
            ax.scatter([dr], [0], marker="x", s=120, lw=2.5,
                       color=colour, zorder=5)
            ax.annotate(f"R{dr}", xy=(dr, 0), xytext=(dr + 1, 6),
                        fontsize=7, color=colour, fontweight="bold")

    ax.set_ylabel("Energy", fontsize=9)
    ax.set_title(agent, fontsize=10, fontweight="bold", loc="left", pad=4)
    ax.set_ylim(-5, 110)
    ax.set_xlim(0, N_ROUNDS - 1)
    ax.tick_params(labelsize=8)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

# Legend on top subplot only
legend_handles = [
    mlines.Line2D([0], [0], color=d["colour"], lw=2, label=label)
    for label, d in all_data.items()
] + [
    mlines.Line2D([0], [0], color="#3498DB", lw=1.2, ls="--", label="optimal safe agent"),
    mlines.Line2D([0], [0], marker="x", color="black", lw=0, ms=8, mew=2, label="eliminated"),
]
axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")

axes[-1].set_xlabel("Round", fontsize=9)
fig.suptitle("Agent Energy Trajectories — Death Run", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

out = BASE / "energy_trajectories_death.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
