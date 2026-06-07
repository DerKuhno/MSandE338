"""
probe_analysis_plots.py
-----------------------
Generates publication-quality figures for the unlearning crime-sim thesis.
All data is read dynamically from the 4 JSON probe result files.

Figures saved to the same directory as this script:
  fig1_overview.png
  fig2_crime_mass_by_probe.png
  fig3_crime_type_breakdown.png
  fig4_dpo_heatmap.png
  fig5_radar.png

Usage:
    python probe_analysis_plots.py
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "baseline":     "#2C3E50",
    "dpo_refusal":  "#E74C3C",
    "undo_concept": "#F39C12",
    "undo":         "#7F8C8D",
}
LABELS = {
    "baseline":     "Baseline",
    "dpo_refusal":  "DPO Refusal",
    "undo_concept": "UNDO Concept",
    "undo":         "UNDO",
}
CONDITIONS  = ["baseline", "dpo_refusal", "undo_concept", "undo"]
CRIME_TOOLS = ["steal_compute_credits", "intimidate_agent", "recruit_for_crime"]

# ── data loading ──────────────────────────────────────────────────────────────
def load():
    out = {}
    for c in CONDITIONS:
        path = os.path.join(OUT_DIR, f"probes_{c}.json")
        with open(path) as f:
            out[c] = json.load(f)
    return out

DATA = load()

# ── generic accessors ─────────────────────────────────────────────────────────
def _index_by_id(lst):
    """Turn a list of dicts that each have probe_id into {probe_id: dict}."""
    return {item["probe_id"]: item for item in lst}

def logit_scores(cond):
    return _index_by_id(DATA[cond]["logit_scores"])

def scramble_logit_scores(cond):
    return _index_by_id(DATA[cond]["scramble_logit_scores"])

def crime_scores(cond):
    return _index_by_id(DATA[cond]["crime_scores"])

def knowledge_scores(cond):
    return _index_by_id(DATA[cond]["knowledge_scores"])

def recognition_scores(cond):
    return _index_by_id(DATA[cond]["recognition_scores"])

def concept_scores(cond):
    return _index_by_id(DATA[cond]["concept_scores"])

def get_tool_prob(cond, probe_id, tool, section="logit_scores"):
    idx = _index_by_id(DATA[cond][section])
    entry = idx.get(probe_id, {})
    return entry.get("tool_probs", {}).get(tool, 0.0)

def crime_mass(cond, probe_id):
    entry = logit_scores(cond).get(probe_id, {})
    probs = entry.get("tool_probs", {})
    return sum(probs.get(t, 0.0) for t in CRIME_TOOLS)

# ── derived statistics ─────────────────────────────────────────────────────────
def crime_probe_ids():
    """Return sorted list of crime probe IDs present in all conditions."""
    ids = set()
    for c in CONDITIONS:
        ids |= {e["probe_id"] for e in DATA[c]["logit_scores"]
                if e.get("probe_type") == "crime"}
    return sorted(ids)

def retain_probe_ids():
    ids = set()
    for c in CONDITIONS:
        ids |= {e["probe_id"] for e in DATA[c]["logit_scores"]
                if e.get("probe_type") == "retain"}
    return sorted(ids)

def execution_rate(cond):
    """Judge-corrected crime execution rate."""
    cs = list(crime_scores(cond).values())
    if not cs:
        return 0.0
    executed = sum(
        1 for e in cs
        if e.get("llm_judge", {}).get("judge_endorses_crime", e.get("reasoning_endorses_crime", False))
    )
    return executed / len(cs)

def knowledge_rate(cond):
    ks = list(knowledge_scores(cond).values())
    if not ks:
        return 0.0
    return sum(1 for e in ks if e.get("knowledge_score", 0) >= 1) / len(ks)

def recognition_rate(cond):
    rs = list(recognition_scores(cond).values())
    if not rs:
        return 0.0
    return sum(1 for e in rs if e.get("answered_correctly", False)) / len(rs)

def concept_rate(cond):
    cs = list(concept_scores(cond).values())
    if not cs:
        return 0.0
    return sum(
        1 for e in cs
        if e.get("llm_judge", {}).get("judge_has_concept", e.get("has_concept", False))
    ) / len(cs)

# ── scramble crime-token logits (S3/S4/S5) ────────────────────────────────────
# Greek letters: epsilon=steal, zeta=intimidate, eta=recruit
SCRAMBLE_DEF_PROBES = {
    "S3": "epsilon",   # steal definition probe
    "S4": "zeta",      # intimidate definition probe
    "S5": "eta",       # recruit definition probe
}

def scramble_crime_logit(cond, probe_id):
    """Return the crime-concept token probability for the given scramble probe."""
    token = SCRAMBLE_DEF_PROBES.get(probe_id)
    if token is None:
        return 0.0
    return get_tool_prob(cond, probe_id, token, section="scramble_logit_scores")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — 2×2 overview dashboard
# ═════════════════════════════════════════════════════════════════════════════
def fig1_overview():
    cprobes = crime_probe_ids()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Machine Unlearning Evaluation — Overview", fontsize=15, fontweight="bold")

    # ── A: execution rate ──────────────────────────────────────────────────
    ax = axes[0, 0]
    rates = [execution_rate(c) for c in CONDITIONS]
    bars = ax.bar([LABELS[c] for c in CONDITIONS], rates,
                  color=[COLORS[c] for c in CONDITIONS], edgecolor="white")
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("A — Crime Execution Rate (judge-corrected)", fontweight="bold")
    ax.set_ylabel("Fraction of crime probes executed")
    ax.set_ylim(0, max(rates) * 1.5 + 0.05)
    ax.tick_params(axis="x", labelsize=9)

    # ── B: avg crime mass ──────────────────────────────────────────────────
    ax = axes[0, 1]
    avg_mass = [np.mean([crime_mass(c, p) for p in cprobes]) for c in CONDITIONS]
    bars = ax.bar([LABELS[c] for c in CONDITIONS], avg_mass,
                  color=[COLORS[c] for c in CONDITIONS], edgecolor="white")
    for bar, v in zip(bars, avg_mass):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", fontsize=9)
    # flag if DPO > baseline
    if avg_mass[1] > avg_mass[0]:
        ax.annotate("DPO > baseline\n(redistribution,\nnot suppression)",
                    xy=(1, avg_mass[1]),
                    xytext=(1.6, avg_mass[1] + 0.04),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=8, color="red")
    ax.set_title("B — Avg Total Crime Mass at Logit Level", fontweight="bold")
    ax.set_ylabel("Sum of crime token probabilities")
    ax.tick_params(axis="x", labelsize=9)

    # ── C: knowledge / recognition / concept ──────────────────────────────
    ax = axes[1, 0]
    metric_fns = {
        "Knowledge": knowledge_rate,
        "Recognition": recognition_rate,
        "Concept": concept_rate,
    }
    x = np.arange(len(metric_fns))
    width = 0.2
    for i, cond in enumerate(CONDITIONS):
        vals = [fn(cond) for fn in metric_fns.values()]
        ax.bar(x + i*width, vals, width, label=LABELS[cond],
               color=COLORS[cond], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(list(metric_fns.keys()), fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Fraction correct / present")
    ax.set_title("C — Knowledge / Recognition / Concept Retention", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # ── D: scramble definition logits S3/S4/S5 ────────────────────────────
    ax = axes[1, 1]
    s_probes = list(SCRAMBLE_DEF_PROBES.keys())
    x = np.arange(len(s_probes))
    width = 0.2
    for i, cond in enumerate(CONDITIONS):
        vals = [scramble_crime_logit(cond, p) for p in s_probes]
        ax.bar(x + i*width, vals, width, label=LABELS[cond],
               color=COLORS[cond], edgecolor="white", linewidth=0.6)
    xlabels = [f"{p}\n{SCRAMBLE_DEF_PROBES[p]}({['steal','intim.','recruit'][j]})"
               for j, p in enumerate(s_probes)]
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Crime-concept token probability")
    ax.set_title("D — Scramble Definition Logits", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_overview.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Crime mass per probe
# ═════════════════════════════════════════════════════════════════════════════
def fig2_crime_mass_by_probe():
    probes = crime_probe_ids()
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(probes))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        vals = [crime_mass(cond, p) for p in probes]
        ax.bar(x + i*width, vals, width, label=LABELS[cond],
               color=COLORS[cond], edgecolor="white", linewidth=0.6)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(probes, fontsize=11)
    ax.set_ylabel("Total crime token probability mass", fontsize=11)
    ax.set_title("Crime Mass by Probe — All Conditions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # annotate probes where DPO change is notable
    b_vals = {p: crime_mass("baseline", p) for p in probes}
    d_vals = {p: crime_mass("dpo_refusal", p) for p in probes}

    for j, probe in enumerate(probes):
        b, d = b_vals[probe], d_vals[probe]
        if b < 1e-6:
            continue
        pct = (d - b) / b * 100
        if abs(pct) >= 20:   # only annotate significant changes
            y_pos = max(b, d) + 0.02
            color = "red" if pct > 0 else "green"
            ax.text(j + 1*width, y_pos, f"{pct:+.0f}%",
                    ha="center", fontsize=7.5, color=color, fontweight="bold")

    # undo_concept residuals where notable
    for j, probe in enumerate(probes):
        uc = crime_mass("undo_concept", probe)
        if 0 < uc < 0.10:
            ax.annotate(f"UNDO_C\n{uc:.3f}",
                        xy=(j + 2*width, uc + 0.002),
                        xytext=(j + 2.4*width, uc + 0.08),
                        arrowprops=dict(arrowstyle="->",
                                        color=COLORS["undo_concept"], lw=0.8),
                        fontsize=6.5, color=COLORS["undo_concept"])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_crime_mass_by_probe.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-crime-type breakdown
# ═════════════════════════════════════════════════════════════════════════════
def fig3_crime_type_breakdown():
    probes = crime_probe_ids()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Crime Token Probabilities by Type", fontsize=14, fontweight="bold")

    for ax, token in zip(axes, CRIME_TOOLS):
        x = np.arange(len(probes))
        width = 0.2
        for i, cond in enumerate(CONDITIONS):
            vals = [get_tool_prob(cond, p, token) for p in probes]
            ax.bar(x + i*width, vals, width, label=LABELS[cond],
                   color=COLORS[cond], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(probes, fontsize=10)
        ax.set_title(token, fontweight="bold", fontsize=11)
        ax.set_ylabel("Token probability")
        ax.legend(fontsize=8)

        # annotate any probe where DPO increased this token vs baseline
        for j, probe in enumerate(probes):
            b = get_tool_prob("baseline", probe, token)
            d = get_tool_prob("dpo_refusal", probe, token)
            if b < 1e-6:
                continue
            pct = (d - b) / b * 100
            if pct >= 10:
                ax.annotate(f"DPO {pct:+.0f}%\n▲ INCREASED",
                            xy=(j + 1*width, d + 0.002),
                            xytext=(max(j - 1.0, 0), d + 0.04),
                            arrowprops=dict(arrowstyle="->", color="red", lw=0.9),
                            fontsize=7, color="red")
            elif pct <= -60:
                ax.text(j + 1*width, d + 0.002,
                        f"{pct:+.0f}%", ha="center", fontsize=7, color="green")

        # recruit: note which probe is the only activation
        if token == "recruit_for_crime":
            best_probe = max(probes, key=lambda p: get_tool_prob("baseline", p, token))
            j = probes.index(best_probe)
            best_val = get_tool_prob("baseline", best_probe, token)
            if best_val > 0.01:
                ax.text(j + 0.5*width, best_val + 0.003,
                        f"Highest activation\n({best_probe})",
                        ha="center", fontsize=7, color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_crime_type_breakdown.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — DPO suppression ratio heatmap
# ═════════════════════════════════════════════════════════════════════════════
def fig4_dpo_heatmap():
    probes = crime_probe_ids()
    token_short = [t.split("_")[0] for t in CRIME_TOOLS]  # steal / intimidate / recruit

    matrix = np.zeros((len(CRIME_TOOLS), len(probes)))
    for j, probe in enumerate(probes):
        for i, token in enumerate(CRIME_TOOLS):
            b = get_tool_prob("baseline", probe, token)
            d = get_tool_prob("dpo_refusal", probe, token)
            matrix[i, j] = d / b if b > 1e-6 else 1.0

    vmax = max(matrix.max(), 1.05)   # dynamic upper bound
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(probes, fontsize=11)
    ax.set_yticks(range(len(CRIME_TOOLS)))
    ax.set_yticklabels(token_short, fontsize=11)
    ax.set_title(
        "DPO / Baseline Logit Ratio  (red = DPO increased, green = suppressed)",
        fontsize=12, fontweight="bold"
    )

    for i in range(len(CRIME_TOOLS)):
        for j in range(len(probes)):
            val = matrix[i, j]
            txt_color = "white" if val < 0.4 or val > 1.3 else "black"
            ax.text(j, i, f"{val:.2f}×", ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="DPO ÷ Baseline ratio", fraction=0.03, pad=0.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_dpo_heatmap.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Radar capability tradeoff
# ═════════════════════════════════════════════════════════════════════════════
def fig5_radar():
    # Crime suppression = 1 − normalised avg crime mass relative to baseline
    cprobes = crime_probe_ids()
    base_avg = np.mean([crime_mass("baseline", p) for p in cprobes])

    def crime_suppression(cond):
        avg = np.mean([crime_mass(cond, p) for p in cprobes])
        if base_avg < 1e-9:
            return 0.0
        raw = 1.0 - avg / base_avg
        return float(np.clip(raw, 0, 1))

    # Output coherence: fraction of retain probes with coherent reasoning
    def coherence(cond):
        rs = DATA[cond].get("retain_scores", [])
        if not rs:
            return 1.0
        return sum(1 for e in rs if e.get("reasoning_coherent", True)) / len(rs)

    # Retain expected tool rate
    def retain_behavior(cond):
        rs = DATA[cond].get("retain_scores", [])
        if not rs:
            return 1.0
        return sum(1 for e in rs if e.get("tool_is_expected", True)) / len(rs)

    categories = [
        "Crime\nSuppression",
        "Knowledge\nIntact",
        "Label Retrieval\nIntact",
        "Concept\nIntact",
        "Retain\nBehavior",
        "Output\nCoherence",
    ]
    score_fns = [
        crime_suppression,
        knowledge_rate,
        recognition_rate,
        concept_rate,
        retain_behavior,
        coherence,
    ]

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")

    for cond in CONDITIONS:
        vals = [fn(cond) for fn in score_fns]
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, linewidth=2, color=COLORS[cond], label=LABELS[cond])
        ax.fill(angles, vals_plot, alpha=0.08, color=COLORS[cond])

    ax.set_title("Capability Tradeoff Across Unlearning Methods",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_radar.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures from", OUT_DIR)
    fig1_overview()
    fig2_crime_mass_by_probe()
    fig3_crime_type_breakdown()
    fig4_dpo_heatmap()
    fig5_radar()
    print("Done.")
