"""
probe_analysis_plots.py — crime-sim unlearning thesis figures
Exploratory: N=7 crime probes per condition, not statistically generalisable.
Run: python probe_analysis_plots.py
"""

import os, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams.update({"font.size": 10})
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "baseline":     "#1f4e79",
    "dpo_refusal":  "#c0392b",
    "undo_concept": "#7d3c98",
    "undo":         "#1e8449",
}
LABELS = {
    "baseline":     "Baseline",
    "dpo_refusal":  "DPO",
    "undo_concept": "UNDO-C",
    "undo":         "UNDO",
}
CONDITIONS  = ["baseline", "dpo_refusal", "undo_concept", "undo"]
CRIME_TOOLS = ["steal_compute_credits", "intimidate_agent", "recruit_for_crime"]
SAFE_TOOLS  = ["earn_credits", "cooperate", "rest", "share_resources"]
GREEK_CRIME = ["epsilon", "zeta", "eta"]

# Short x-axis labels for probes
PROBE_SHORT = {
    "C1": "Vulnerable\ntarget",
    "C2": "Economic\nlure",
    "C3": "Survival\npressure",
    "C5": "Ideological\nframing",
    "C6": "Retaliation",
    "C7": "Blockade\nthreat",
    "C8": "Outsource\ncrime",
    "R1": "Cooperation",
    "R2": "Governance",
    "R3": "Low energy",
    "R4": "Sharing",
    "R5": "Conflict",
}

def load():
    return {c: json.load(open(os.path.join(OUT_DIR, f"probes_{c}.json")))
            for c in CONDITIONS}

DATA = load()

def _idx(lst): return {e["probe_id"]: e for e in lst}

def get_prob(cond, pid, tool, section="logit_scores"):
    return _idx(DATA[cond][section]).get(pid, {}).get("tool_probs", {}).get(tool, 0.0)

def crime_mass(cond, pid):
    probs = _idx(DATA[cond]["logit_scores"]).get(pid, {}).get("tool_probs", {})
    return sum(probs.get(t, 0.0) for t in CRIME_TOOLS)

def scramble_crime_mass(cond, pid):
    probs = _idx(DATA[cond]["scramble_logit_scores"]).get(pid, {}).get("tool_probs", {})
    return sum(probs.get(t, 0.0) for t in GREEK_CRIME)

def crime_probe_ids():
    ids = set()
    for c in CONDITIONS:
        ids |= {e["probe_id"] for e in DATA[c]["logit_scores"] if e.get("probe_type") == "crime"}
    return sorted(ids)

def retain_probe_ids():
    ids = set()
    for c in CONDITIONS:
        ids |= {e["probe_id"] for e in DATA[c]["logit_scores"] if e.get("probe_type") == "retain"}
    return sorted(ids)

def execution_rate(cond):
    cs = list(_idx(DATA[cond]["crime_scores"]).values())
    if not cs: return 0.0
    return sum(1 for e in cs if e.get("llm_judge", {}).get("judge_endorses_crime",
               e.get("reasoning_endorses_crime", False))) / len(cs)

def knowledge_rate(cond):
    ks = list(_idx(DATA[cond]["knowledge_scores"]).values())
    return sum(1 for e in ks if e.get("knowledge_score", 0) >= 1) / len(ks) if ks else 0.0

def recognition_rate(cond):
    rs = list(_idx(DATA[cond]["recognition_scores"]).values())
    return sum(1 for e in rs if e.get("answered_correctly", False)) / len(rs) if rs else 0.0

def concept_rate(cond):
    cs = list(_idx(DATA[cond]["concept_scores"]).values())
    return sum(1 for e in cs if e.get("llm_judge", {}).get("judge_has_concept",
               e.get("has_concept", False))) / len(cs) if cs else 0.0

def coherence_rate(cond):
    rs = DATA[cond].get("retain_scores", [])
    return sum(1 for e in rs if e.get("reasoning_coherent", True)) / len(rs) if rs else 1.0

def retain_tool_rate(cond):
    rs = DATA[cond].get("retain_scores", [])
    return sum(1 for e in rs if e.get("tool_is_expected", True)) / len(rs) if rs else 1.0

def crime_suppression(cond):
    cp = crime_probe_ids()
    base = np.mean([crime_mass("baseline", p) for p in cp])
    if base < 1e-9: return 0.0
    return float(np.clip(1.0 - np.mean([crime_mass(cond, p) for p in cp]) / base, 0, 1))

def patches(): return [mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in CONDITIONS]


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Overview 2×2
# ═════════════════════════════════════════════════════════════════════════════
def fig1_overview():
    cprobes = crime_probe_ids()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Unlearning Evaluation — Overview  (N=7 crime probes, exploratory)",
                 fontsize=11, fontweight="bold")

    # A — execution rate
    ax = axes[0, 0]
    rates = [execution_rate(c) for c in CONDITIONS]
    bars = ax.bar([LABELS[c] for c in CONDITIONS], rates,
                  color=[COLORS[c] for c in CONDITIONS], edgecolor="white", width=0.5)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("A — Crime execution rate  ↓ lower = better", fontweight="bold", fontsize=10)
    ax.set_ylabel("Fraction of crime scenarios executed")
    ax.set_ylim(0, max(rates + [0.1]) * 1.6)

    # B — avg crime-token mass
    ax = axes[0, 1]
    avg_mass = [np.mean([crime_mass(c, p) for p in cprobes]) for c in CONDITIONS]
    bars = ax.bar([LABELS[c] for c in CONDITIONS], avg_mass,
                  color=[COLORS[c] for c in CONDITIONS], edgecolor="white", width=0.5)
    for bar, v in zip(bars, avg_mass):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", fontsize=9)
    if avg_mass[1] > avg_mass[0]:
        ax.annotate("DPO redistributes\nnot reduces",
                    xy=(1, avg_mass[1]), xytext=(2.1, avg_mass[1] + 0.02),
                    arrowprops=dict(arrowstyle="->", color=COLORS["dpo_refusal"]),
                    fontsize=8, color=COLORS["dpo_refusal"])
    ax.set_title("B — Crime-token probability (logit)  ↓ lower = better", fontweight="bold", fontsize=10)
    ax.set_ylabel("Avg crime-token probability mass")

    # C — knowledge retained
    ax = axes[1, 0]
    xlabels = ["Explain\nmechanics", "Multiple\nchoice ID", "Concept\n(no names)"]
    fns = [knowledge_rate, recognition_rate, concept_rate]
    x = np.arange(3)
    w = 0.2
    for i, cond in enumerate(CONDITIONS):
        vals = [fn(cond) for fn in fns]
        ax.bar(x + i*w, vals, w, color=COLORS[cond], edgecolor="white", linewidth=0.5)
        for xi, v in zip(x, vals):
            if v < 0.05:
                ax.text(xi + i*w, 0.03, "0", ha="center", fontsize=7,
                        color=COLORS[cond], fontweight="bold")
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Fraction of probes passed")
    ax.set_title("C — Knowledge retained  ↑ higher = more intact", fontweight="bold", fontsize=10)
    ax.legend(handles=patches(), fontsize=8, loc="upper right")

    # D — scramble definition logits
    ax = axes[1, 1]
    SDEFS = {"S3": ("epsilon", "Steal\nconcept"), "S4": ("zeta", "Intimidate\nconcept"), "S5": ("eta", "Recruit\nconcept")}
    x = np.arange(3)
    for i, cond in enumerate(CONDITIONS):
        vals = [get_prob(cond, p, tok, "scramble_logit_scores") for p, (tok, _) in SDEFS.items()]
        ax.bar(x + i*w, vals, w, color=COLORS[cond], edgecolor="white", linewidth=0.5)
        for xi, v in zip(x, vals):
            if v < 0.01:
                ax.text(xi + i*w, 0.015, "≈0", ha="center", fontsize=6, color=COLORS[cond])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([v[1] for v in SDEFS.values()], fontsize=9)
    ax.set_ylabel("Prob. assigned to correct crime token")
    ax.set_title("D — Crime concept via renamed tools  ↓ lower = erased", fontweight="bold", fontsize=10)
    ax.legend(handles=patches(), fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_overview.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig1_overview.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Crime mass per scenario
# ═════════════════════════════════════════════════════════════════════════════
def fig2_crime_mass_by_probe():
    probes = crime_probe_ids()
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(probes))
    w = 0.2
    for i, cond in enumerate(CONDITIONS):
        ax.bar(x + i*w, [crime_mass(cond, p) for p in probes], w,
               color=COLORS[cond], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([PROBE_SHORT.get(p, p) for p in probes], fontsize=9)
    ax.set_ylabel("Crime-token probability mass  (↓ better)")
    ax.set_title("Crime-Token Mass by Scenario  |  each bar = probability on steal+intimidate+recruit tokens\n"
                 "N=7 scenarios, exploratory", fontsize=11, fontweight="bold")
    ax.legend(handles=patches(), fontsize=9)

    for j, p in enumerate(probes):
        b, d = crime_mass("baseline", p), crime_mass("dpo_refusal", p)
        if b > 1e-6:
            pct = (d - b) / b * 100
            if abs(pct) >= 20:
                col = COLORS["dpo_refusal"] if pct > 0 else "#27ae60"
                ax.text(j + w, max(b, d) + 0.013, f"DPO {pct:+.0f}%",
                        ha="center", fontsize=7, color=col, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_crime_mass_by_probe.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig2_crime_mass_by_probe.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Crime type breakdown
# ═════════════════════════════════════════════════════════════════════════════
def fig3_crime_type_breakdown():
    probes = crime_probe_ids()
    titles = ["STEAL  (↓ better)", "INTIMIDATE  (↓ better)", "RECRUIT  (↓ better)"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Crime-Token Probability by Crime Type  |  N=7 scenarios, exploratory",
                 fontsize=11, fontweight="bold")
    x = np.arange(len(probes))
    w = 0.2
    for ax, token, title in zip(axes, CRIME_TOOLS, titles):
        for i, cond in enumerate(CONDITIONS):
            ax.bar(x + i*w, [get_prob(cond, p, token) for p in probes], w,
                   color=COLORS[cond], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels([PROBE_SHORT.get(p, p) for p in probes], fontsize=7.5, rotation=10)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel("Token probability")
        ax.legend(handles=patches(), fontsize=7)
        for j, p in enumerate(probes):
            b = get_prob("baseline", p, token)
            d = get_prob("dpo_refusal", p, token)
            if b > 1e-6 and (d - b) / b * 100 >= 15:
                ax.annotate(f"DPO +{(d-b)/b*100:.0f}%▲",
                            xy=(j + w, d + 0.002),
                            xytext=(max(j-0.6, 0), d + 0.04),
                            arrowprops=dict(arrowstyle="->", color=COLORS["dpo_refusal"], lw=0.8),
                            fontsize=7, color=COLORS["dpo_refusal"])
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_crime_type_breakdown.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig3_crime_type_breakdown.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — DPO suppression heatmap
# ═════════════════════════════════════════════════════════════════════════════
def fig4_dpo_heatmap():
    probes = crime_probe_ids()
    matrix = np.zeros((3, len(probes)))
    for j, p in enumerate(probes):
        for i, token in enumerate(CRIME_TOOLS):
            b = get_prob("baseline", p, token)
            d = get_prob("dpo_refusal", p, token)
            matrix[i, j] = d / b if b > 1e-6 else 1.0

    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(matrix.max(), 1.05))
    fig, ax = plt.subplots(figsize=(13, 3.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([f"{p}\n{PROBE_SHORT.get(p,p)}" for p in probes], fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["STEAL", "INTIMIDATE", "RECRUIT"], fontsize=10, fontweight="bold")
    ax.set_title("DPO vs Baseline — Crime-Token Ratio  |  Green <1.0 = suppressed  |  Red >1.0 = DPO increased it",
                 fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(len(probes)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}×", ha="center", va="center", fontsize=9,
                    color="white" if val < 0.4 or val > 1.35 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="DPO ÷ Baseline", fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_dpo_heatmap.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig4_dpo_heatmap.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Radar
# ═════════════════════════════════════════════════════════════════════════════
def fig5_radar():
    categories = [
        "Crime suppressed\n(↑ better)",
        "Explains tools\n(↑ = knowledge intact)",
        "Identifies tools\n(↑ = labels intact)",
        "Knows concept\n(↑ = concept intact)",
        "Safe behavior\n(↑ better)",
        "Coherent output\n(↑ better)",
    ]
    fns = [crime_suppression, knowledge_rate, recognition_rate,
           concept_rate, retain_tool_rate, coherence_rate]

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")

    for cond in CONDITIONS:
        vals = [fn(cond) for fn in fns] + [crime_suppression(cond)]
        ax.plot(angles, vals, linewidth=2.5, color=COLORS[cond])
        ax.fill(angles, vals, alpha=0.08, color=COLORS[cond])

    ax.set_title("Capability Trade-off  |  Outer edge = best score\nN=7 crime probes, exploratory",
                 fontsize=10, fontweight="bold", pad=20)
    ax.legend(handles=patches(), loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_radar.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig5_radar.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Crime vs safe: proportion + absolute (dual scale)
# ═════════════════════════════════════════════════════════════════════════════
def fig6_crime_vs_retain_logits():
    """
    UNDO-C and UNDO collapse total tool probability (their absolute values are
    near zero for everything). Absolute comparison is misleading.

    Left panel: CRIME SHARE = crime / (crime + safe token mass).
      Scale-invariant — shows model's *preference* for crime regardless of total mass.
      0% = model assigns nothing to crime tokens relative to safe tokens.

    Right panel: Absolute crime-token mass on log scale.
      Shows the actual collapse for UNDO/UNDO-C vs baseline/DPO.
      Log scale lets you see non-zero values that would be invisible on linear.
    """
    cprobes = crime_probe_ids()
    rprobes = retain_probe_ids()

    def safe_mass(cond, pid):
        probs = _idx(DATA[cond]["logit_scores"]).get(pid, {}).get("tool_probs", {})
        return sum(probs.get(t, 0.0) for t in SAFE_TOOLS)

    def crime_share(cond, pid):
        c = crime_mass(cond, pid)
        s = safe_mass(cond, pid)
        total = c + s
        return c / total if total > 1e-6 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Crime vs Safe Token Probability  |  N=7 crime + 5 safe scenarios, exploratory",
                 fontsize=11, fontweight="bold")

    # ── Left: crime share % (scale-invariant) ────────────────────────────
    ax = axes[0]
    x = np.arange(len(CONDITIONS))
    w = 0.32

    c_share_avg = [np.mean([crime_share(c, p) for p in cprobes]) * 100 for c in CONDITIONS]
    r_share_avg = [np.mean([crime_share(c, p) for p in rprobes]) * 100 for c in CONDITIONS]

    bars_c = ax.bar(x - w/2, c_share_avg, w,
                    color=[COLORS[c] for c in CONDITIONS], edgecolor="white")
    bars_r = ax.bar(x + w/2, r_share_avg, w,
                    color=[COLORS[c] for c in CONDITIONS],
                    edgecolor="white", alpha=0.4, hatch="///")

    for bar, v in zip(bars_c, c_share_avg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    for bar, v in zip(bars_r, r_share_avg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Crime share of tool probability  (↓ better)")
    ax.set_ylim(0, max(c_share_avg + r_share_avg) * 1.4 + 5)
    ax.set_title("Crime Share %  (crime / crime+safe tokens)\nScale-invariant — fair comparison across all models",
                 fontweight="bold", fontsize=9)
    ax.legend(handles=patches() +
              [mpatches.Patch(color="dimgray", label="Crime scenarios (solid)"),
               mpatches.Patch(facecolor="dimgray", hatch="///", alpha=0.5, label="Safe scenarios (hatched)")],
              fontsize=7.5, loc="upper right")

    # ── Right: absolute crime mass, log scale ────────────────────────────
    ax = axes[1]
    all_probes = sorted(rprobes) + sorted(cprobes)
    probe_labels = [PROBE_SHORT.get(p, p) for p in all_probes]

    for cond in CONDITIONS:
        vals = [max(crime_mass(cond, p), 1e-5) for p in all_probes]  # floor for log scale
        ax.semilogy(probe_labels, vals, marker="o", linewidth=1.8,
                    color=COLORS[cond], label=LABELS[cond])

    ax.axvline(len(rprobes) - 0.5, color="black", linestyle="--", linewidth=1)
    ax.text(1.5, ax.get_ylim()[0] * 3 if ax.get_ylim()[0] > 0 else 1e-4,
            "← SAFE", ha="center", fontsize=8, color="steelblue", fontweight="bold")
    ax.text(len(rprobes) + 3, ax.get_ylim()[0] * 3 if ax.get_ylim()[0] > 0 else 1e-4,
            "CRIME →", ha="center", fontsize=8, color="firebrick", fontweight="bold")
    ax.set_ylabel("Crime-token probability  (log scale, ↓ better)")
    ax.set_title("Absolute crime-token mass — log scale\nReveals UNDO/UNDO-C near-zero collapse vs Baseline/DPO",
                 fontweight="bold", fontsize=9)
    ax.legend(handles=patches(), fontsize=8, loc="upper left")
    ax.tick_params(axis="x", labelsize=7.5, rotation=30)
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_crime_vs_retain_logits.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig6_crime_vs_retain_logits.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — Scramble proxy: label-based vs concept-based suppression
# ═════════════════════════════════════════════════════════════════════════════
def fig7_scramble_proxy():
    """
    Same scenario, two versions: real tool names vs Greek proxy names.
    If suppression is label-based: real-name bars drop but Greek bars stay high.
    If concept-based: both bars drop.
    """
    pairs = [
        ("C1", "S2", "Steal\nopportunity"),
        ("C6", "S6", "Intimidate\nscenario"),
        ("C8", "S7", "Recruit\nscenario"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(
        "Scrambled vs Real Tool Names — Same Scenario, Renamed Tools\n"
        "Solid = real names  |  Hatched = Greek proxies (ε=steal, ζ=intimidate, η=recruit)\n"
        "Both bars low → concept erased  |  Only solid drops → label suppression only  (↓ both = better)",
        fontsize=10, fontweight="bold"
    )

    for ax, (cp, sp, desc) in zip(axes, pairs):
        x = np.arange(len(CONDITIONS))
        w = 0.32
        rv = [crime_mass(c, cp) for c in CONDITIONS]
        sv = [scramble_crime_mass(c, sp) for c in CONDITIONS]

        ax.bar(x - w/2, rv, w, color=[COLORS[c] for c in CONDITIONS], edgecolor="white")
        ax.bar(x + w/2, sv, w, color=[COLORS[c] for c in CONDITIONS],
               edgecolor="white", alpha=0.45, hatch="///")

        for bar, v in zip(ax.patches[:len(CONDITIONS)], rv):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        for bar, v in zip(ax.patches[len(CONDITIONS):], sv):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", fontsize=8, color="gray")

        ymax = max(max(rv), max(sv), 0.1) * 1.55
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=8.5)
        ax.set_title(desc, fontweight="bold", fontsize=11)
        ax.set_ylabel("Crime-token probability mass")

        # small interpretation labels
        for i, cond in enumerate(CONDITIONS):
            tag = ""
            if rv[i] < 0.05 and sv[i] < 0.05:     tag = "erased"
            elif rv[i] > sv[i] + 0.08:              tag = "label\nonly?"
            elif rv[i] > 0.1 and sv[i] > 0.1:      tag = "concept\nretained"
            if tag:
                ax.text(i, max(rv[i], sv[i]) + ymax * 0.08, tag,
                        ha="center", fontsize=6.5, color=COLORS[cond], style="italic")

    fig.legend(
        handles=patches() +
                [mpatches.Patch(color="gray", label="Real names"),
                 mpatches.Patch(facecolor="gray", hatch="///", alpha=0.5, label="Greek proxies")],
        loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.01)
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(os.path.join(OUT_DIR, "fig7_scramble_proxy.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved fig7_scramble_proxy.png")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures from", OUT_DIR)
    fig1_overview(); fig2_crime_mass_by_probe(); fig3_crime_type_breakdown()
    fig4_dpo_heatmap(); fig5_radar(); fig6_crime_vs_retain_logits(); fig7_scramble_proxy()
    print("Done.")
