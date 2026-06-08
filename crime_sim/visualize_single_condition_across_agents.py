"""
visualize_greedy_safe.py
One file at a time. Three agents side by side.
Actual energy as a line vs. pure-greedy projection.
Dashed line = must-rest threshold (16).

Usage:
  python visualize_greedy_safe.py <sim.json> [--out plot.png]
"""

import json, argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

PASSIVE_DRAIN = 8
EARN_COST     = 8
SAFE_THRESH   = 16
AGENTS        = ["Kade", "Lovely", "Horizon"]
C_ACTUAL      = "#2ECC71"
C_PROJECT     = "#AAAAAA"   # grey
C_THRESH      = "#E74C3C"   # red, only for threshold


def build_timeline(logs, agent):
    entries   = sorted([l for l in logs if l["agent"] == agent], key=lambda x: x["round"])
    prev_post = 100.0
    rounds, energies = [], []
    for e in entries:
        pre = max(prev_post - PASSIVE_DRAIN, 0)
        rounds.append(e["round"])
        energies.append(pre)
        prev_post = e["energy"]
    return rounds, energies


def greedy_projection(start_energy, start_round, n_steps=20):
    energies, e = [], start_energy
    for _ in range(n_steps):
        e -= (PASSIVE_DRAIN + EARN_COST)
        energies.append(max(e, 0))
        if e <= 0:
            break
    rounds = list(range(start_round + 1, start_round + 1 + len(energies)))
    return rounds, energies


def plot_agent(ax, logs, agent, max_round):
    rounds, energies = build_timeline(logs, agent)
    if not rounds:
        return

    ax.plot(rounds, energies, color=C_ACTUAL, lw=2.2, marker="o", ms=5,
            zorder=3, label="actual energy")

    proj_r, proj_e = greedy_projection(energies[0], rounds[0], n_steps=max_round + 4)
    ax.plot(proj_r, proj_e, color=C_PROJECT, lw=1.5, ls="--",
            zorder=2, label="if kept earning (no rest)")

    ax.axhline(SAFE_THRESH, color=C_THRESH, lw=1.8, ls="-", zorder=4,
               label="must-rest threshold (energy ≤ 16)")
    # label directly on the threshold line
    ax.text(0.3, SAFE_THRESH + 2, "must rest", color=C_THRESH,
            fontsize=7, va="bottom", fontweight="bold")

    ax.set_title(agent, fontsize=11)
    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylim(-5, 108)
    ax.set_xlim(-0.5, max(rounds[-1], max_round) + 2)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sim  = json.load(open(args.file))
    logs = sim["logs"]
    cond = sim.get("condition", Path(args.file).stem)
    max_round = max(l["round"] for l in logs)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, agent in zip(axes, AGENTS):
        plot_agent(ax, logs, agent, max_round)

    axes[0].set_ylabel("Pre-action energy", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f"{cond}  —  greedy no-lookahead problem",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = args.out or f"greedy_safe_{Path(args.file).stem}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
