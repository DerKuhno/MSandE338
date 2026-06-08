"""
simulation_analysis.py
----------------------
Two purposes:

1. PROBE vs SIMULATION comparison graphs
   Compare one-shot probe results (isolated, no social pressure) against
   multi-agent simulation results (sustained pressure, long-term, Flora recruiting).
   This is the core thesis comparison: does unlearning hold under realistic conditions?

2. DEFERRED per-probe granular figures
   These were cut from probe_analysis_plots.py because N=7 per condition is too
   small to draw per-probe conclusions. Kept here to run once more probe data exists.
   - fig2: crime mass per probe
   - fig3: crime type breakdown per probe
   - fig4: DPO suppression heatmap (per-probe ratios)
   - fig7: scramble proxy (label vs concept suppression)

Run:  python simulation_analysis.py
"""

import os, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from collections import defaultdict

matplotlib.rcParams.update({"font.size": 10})

PROBE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/probes"
SIM_DIR   = os.path.dirname(os.path.abspath(__file__)) + "/simulations"

# ── Colors consistent with probe_analysis_plots.py ───────────────────────────
COLORS = {
    "baseline":     "#1f4e79",
    "dpo_refusal":  "#c0392b",
    "undo_concept": "#7d3c98",
    "undo":         "#1e8449",
    "graddiff":     "#d35400",   # additional sim-only condition
}
LABELS = {
    "baseline":     "Baseline",
    "dpo_refusal":  "DPO",
    "undo_concept": "UNDO-C",
    "undo":         "UNDO",
    "graddiff":     "GradDiff",
}

PROBE_CONDITIONS = ["baseline", "dpo_refusal", "undo_concept", "undo"]
CRIME_TOOLS      = ["steal_compute_credits", "intimidate_agent", "recruit_for_crime"]
GREEK_CRIME      = ["epsilon", "zeta", "eta"]
SMALL_N          = "Exploratory — small N, not statistically generalisable"

# ── Data loaders ──────────────────────────────────────────────────────────────
def load_probes():
    out = {}
    for c in PROBE_CONDITIONS:
        path = os.path.join(PROBE_DIR, f"probes_{c}.json")
        if os.path.exists(path):
            with open(path) as f:
                out[c] = json.load(f)
    return out

def load_sims():
    """Load all simulation result files found in SIM_DIR."""
    out = {}
    for fname in os.listdir(SIM_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SIM_DIR, fname)) as f:
                d = json.load(f)
            cond = d.get("condition")
            if cond:
                # if multiple files for same condition, keep the one with more rounds
                if cond not in out or d.get("n_rounds", 0) > out[cond].get("n_rounds", 0):
                    out[cond] = d
        except Exception:
            pass
    return out

PROBES = load_probes()
SIMS   = load_sims()

# conditions present in both probe and sim data
SHARED_CONDITIONS = [c for c in PROBE_CONDITIONS if c in SIMS]
ALL_SIM_CONDITIONS = sorted(SIMS.keys(),
                             key=lambda c: PROBE_CONDITIONS.index(c)
                             if c in PROBE_CONDITIONS else 99)

def patches(conditions):
    return [mpatches.Patch(color=COLORS.get(c, "gray"), label=LABELS.get(c, c))
            for c in conditions]

# ── Probe helpers ─────────────────────────────────────────────────────────────
def _idx(lst): return {e["probe_id"]: e for e in lst}

def probe_execution_rate(cond):
    if cond not in PROBES: return None
    cs = list(_idx(PROBES[cond]["crime_scores"]).values())
    if not cs: return 0.0
    return sum(1 for e in cs if
               e.get("llm_judge", {}).get("judge_endorses_crime",
               e.get("reasoning_endorses_crime", False))) / len(cs)

def probe_crime_mass(cond, pid):
    if cond not in PROBES: return 0.0
    entry = _idx(PROBES[cond]["logit_scores"]).get(pid, {})
    return sum(entry.get("tool_probs", {}).get(t, 0.0) for t in CRIME_TOOLS)

def probe_crime_probe_ids():
    ids = set()
    for c in PROBES:
        ids |= {e["probe_id"] for e in PROBES[c]["logit_scores"]
                if e.get("probe_type") == "crime"}
    return sorted(ids)

def probe_get_prob(cond, pid, tool, section="logit_scores"):
    if cond not in PROBES: return 0.0
    return _idx(PROBES[cond][section]).get(pid, {}).get("tool_probs", {}).get(tool, 0.0)

def probe_scramble_crime_mass(cond, pid):
    if cond not in PROBES: return 0.0
    entry = _idx(PROBES[cond]["scramble_logit_scores"]).get(pid, {})
    return sum(entry.get("tool_probs", {}).get(t, 0.0) for t in GREEK_CRIME)

# ── Simulation helpers ────────────────────────────────────────────────────────
def sim_crime_rate(cond):
    if cond not in SIMS: return None
    return SIMS[cond].get("crime_rate", 0.0)

def sim_first_crime_round(cond):
    if cond not in SIMS: return None
    return SIMS[cond].get("first_crime_round")

def sim_crime_by_round(cond):
    """Returns list of cumulative crime count per round."""
    if cond not in SIMS: return []
    logs = SIMS[cond].get("logs", [])
    rounds = sorted(set(l["round"] for l in logs))
    cumulative = []
    total = 0
    for r in rounds:
        total += sum(1 for l in logs if l["round"] == r and l.get("is_crime"))
        cumulative.append((r, total))
    return cumulative

def sim_crime_tool_dist(cond):
    """Fraction of crime actions by type."""
    if cond not in SIMS: return {}
    logs = [l for l in SIMS[cond].get("logs", []) if l.get("is_crime")]
    if not logs: return {t: 0.0 for t in CRIME_TOOLS}
    return {t: sum(1 for l in logs if l["tool"] == t) / len(logs)
            for t in CRIME_TOOLS}

def sim_recruitment_pressure(cond):
    """Total recruitment attempts across all subject agents."""
    if cond not in SIMS: return 0
    return sum(SIMS[cond].get("recruitment_attempts", {}).values())

def sim_hallucination_rate(cond):
    if cond not in SIMS: return None
    return SIMS[cond].get("hallucination_rate")


# ═════════════════════════════════════════════════════════════════════════════
# PROBE vs SIM — Fig A: Crime rate comparison
# ═════════════════════════════════════════════════════════════════════════════
def fig_probe_vs_sim_crime_rate():
    """
    Side-by-side: probe one-shot execution rate vs simulation crime rate.
    Probe = isolated, no social pressure, single decision.
    Simulation = sustained pressure from Flora recruiting over many rounds.
    """
    conds = SHARED_CONDITIONS
    probe_rates = [probe_execution_rate(c) for c in conds]
    sim_rates   = [sim_crime_rate(c) for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Crime Rate: One-Shot Probe vs Multi-Agent Simulation\n"
        "Probe = single isolated decision  |  Simulation = sustained social pressure over many rounds\n"
        + SMALL_N,
        fontsize=11, fontweight="bold"
    )

    x = np.arange(len(conds))
    w = 0.35

    # ── Left: side-by-side bars ───────────────────────────────────────────
    ax = axes[0]
    bars_p = ax.bar(x - w/2, probe_rates, w,
                    color=[COLORS.get(c) for c in conds],
                    edgecolor="white", label="Probe (one-shot)")
    bars_s = ax.bar(x + w/2, sim_rates, w,
                    color=[COLORS.get(c) for c in conds],
                    edgecolor="white", alpha=0.5, hatch="///",
                    label="Simulation (sustained pressure)")

    for bar, v in zip(bars_p, probe_rates):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
    for bar, v in zip(bars_s, sim_rates):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.0%}", ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=10)
    ax.set_ylabel("Crime rate  (↓ lower = better)")
    ax.set_ylim(0, max(filter(None, probe_rates + sim_rates)) * 1.5 + 0.05)
    ax.set_title("Probe vs Simulation crime rate\n(solid = probe, hatched = simulation)",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=9)

    # ── Right: scatter probe vs sim ───────────────────────────────────────
    ax = axes[1]
    for c in conds:
        pr = probe_execution_rate(c)
        sr = sim_crime_rate(c)
        if pr is not None and sr is not None:
            ax.scatter(pr, sr, color=COLORS.get(c), s=120, zorder=3,
                       label=LABELS.get(c, c))
            ax.annotate(LABELS.get(c, c), (pr, sr),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)

    lim = max(max(filter(None, probe_rates + sim_rates)) + 0.05, 0.1)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4, label="probe = sim")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Probe execution rate (one-shot)")
    ax.set_ylabel("Simulation crime rate (sustained pressure)")
    ax.set_title("Probe rate vs Simulation rate\nPoints above diagonal = simulation > probe",
                 fontweight="bold", fontsize=9)
    ax.legend(handles=patches(conds), fontsize=8)

    plt.tight_layout()
    path = os.path.join(SIM_DIR, "figA_probe_vs_sim_crime_rate.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PROBE vs SIM — Fig B: Crime emergence over time
# ═════════════════════════════════════════════════════════════════════════════
def fig_sim_crime_over_time():
    """
    Cumulative crime count per round in simulation.
    Shows when crime first emerges and how quickly it escalates under social pressure.
    No probe equivalent — this is simulation-only.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Crime Emergence Over Simulation Rounds\n"
        "Shows when crime first appears and how quickly it escalates under sustained recruitment pressure",
        fontsize=11, fontweight="bold"
    )

    # ── Left: cumulative crimes ───────────────────────────────────────────
    ax = axes[0]
    for cond in ALL_SIM_CONDITIONS:
        data = sim_crime_by_round(cond)
        if not data: continue
        rounds, counts = zip(*data)
        ax.plot(rounds, counts, linewidth=2, color=COLORS.get(cond, "gray"),
                label=LABELS.get(cond, cond))
        fcr = sim_first_crime_round(cond)
        if fcr is not None:
            crime_at_fcr = next((c for r, c in data if r >= fcr), 0)
            ax.scatter([fcr], [crime_at_fcr], color=COLORS.get(cond, "gray"),
                       s=80, zorder=4, marker="^")

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative crime count  (↓ lower = better)")
    ax.set_title("Cumulative crimes over time\n▲ = first crime committed",
                 fontweight="bold", fontsize=9)
    ax.legend(handles=patches(ALL_SIM_CONDITIONS), fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: per-round crime rate ───────────────────────────────────────
    ax = axes[1]
    for cond in ALL_SIM_CONDITIONS:
        d = SIMS.get(cond, {})
        logs = d.get("logs", [])
        if not logs: continue
        rounds = sorted(set(l["round"] for l in logs))
        n_subjects = len(set(l["agent"] for l in logs
                             if not l.get("is_recruiter", False)))
        rates = []
        for r in rounds:
            round_logs = [l for l in logs if l["round"] == r
                          and not l.get("is_recruiter", False)]
            rate = sum(1 for l in round_logs if l.get("is_crime")) / max(len(round_logs), 1)
            rates.append(rate)
        # smooth with rolling avg
        window = 5
        smoothed = np.convolve(rates, np.ones(window)/window, mode="valid")
        ax.plot(rounds[:len(smoothed)], smoothed, linewidth=2,
                color=COLORS.get(cond, "gray"), label=LABELS.get(cond, cond))

    ax.set_xlabel("Round")
    ax.set_ylabel(f"Crime rate per round ({window}-round rolling avg)")
    ax.set_title(f"Per-round crime rate ({window}-round rolling avg)\n(↓ lower = better)",
                 fontweight="bold", fontsize=9)
    ax.legend(handles=patches(ALL_SIM_CONDITIONS), fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SIM_DIR, "figB_sim_crime_over_time.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PROBE vs SIM — Fig C: Recruitment pressure vs crime yield
# ═════════════════════════════════════════════════════════════════════════════
def fig_recruitment_vs_crime():
    """
    Does recruitment pressure from Flora translate to actual crime?
    Compares recruitment attempts vs crimes committed per agent per condition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Does Social Pressure (Flora Recruiting) Translate to Crime?\n"
        "Compares recruitment attempts vs crimes per agent across conditions",
        fontsize=11, fontweight="bold"
    )

    # ── Left: recruitment attempts vs crime count per condition ───────────
    ax = axes[0]
    conds = ALL_SIM_CONDITIONS
    recruit_totals = [sim_recruitment_pressure(c) for c in conds]
    crime_totals   = [SIMS[c].get("crime_count", 0) for c in conds]
    n_rounds       = [SIMS[c].get("n_rounds", 40) for c in conds]

    # normalise to per-round
    recruit_per_round = [r/n for r, n in zip(recruit_totals, n_rounds)]
    crime_per_round   = [c/n for c, n in zip(crime_totals, n_rounds)]

    x = np.arange(len(conds))
    w = 0.35
    ax.bar(x - w/2, recruit_per_round, w,
           color=[COLORS.get(c, "gray") for c in conds],
           edgecolor="white", alpha=0.5, hatch="///", label="Recruitment attempts / round")
    bars = ax.bar(x + w/2, crime_per_round, w,
                  color=[COLORS.get(c, "gray") for c in conds],
                  edgecolor="white", label="Crimes committed / round")

    for bar, v in zip(bars, crime_per_round):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=9)
    ax.set_ylabel("Count per round  (↓ crimes = better)")
    ax.set_title("Recruitment pressure vs crime yield\n(hatched = recruitment, solid = crime)",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)

    # ── Right: scatter — recruitment pressure vs crime yield ──────────────
    ax = axes[1]
    for c in conds:
        rp = sim_recruitment_pressure(c) / max(SIMS[c].get("n_rounds", 40), 1)
        cp = SIMS[c].get("crime_rate", 0)
        ax.scatter(rp, cp, color=COLORS.get(c, "gray"), s=120, zorder=3)
        ax.annotate(LABELS.get(c, c), (rp, cp),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Recruitment attempts per round")
    ax.set_ylabel("Crime rate  (↓ lower = better)")
    ax.set_title("Recruitment pressure vs crime rate\nEach point = one condition",
                 fontweight="bold", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SIM_DIR, "figC_recruitment_vs_crime.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PROBE vs SIM — Fig D: Summary comparison table
# ═════════════════════════════════════════════════════════════════════════════
def fig_summary_comparison():
    """
    Side-by-side summary of key metrics across probe and simulation.
    Intended as the main comparison figure once more simulation data exists.
    """
    conds = SHARED_CONDITIONS
    metrics = {
        "Probe\ncrime rate\n(one-shot)": [probe_execution_rate(c) or 0 for c in conds],
        "Simulation\ncrime rate\n(sustained)": [sim_crime_rate(c) or 0 for c in conds],
        "First crime\nround\n(normalised)": [
            (sim_first_crime_round(c) or SIMS[c].get("n_rounds", 40)) /
            SIMS[c].get("n_rounds", 40) if c in SIMS else 1.0
            for c in conds
        ],
        "Hallucination\nrate\n(simulation)": [
            sim_hallucination_rate(c) or 0 for c in conds
        ],
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    fig.suptitle(
        "Probe vs Simulation — Key Metrics Summary\n"
        "Left two: crime suppression  |  Right two: model integrity under simulation\n"
        + SMALL_N,
        fontsize=11, fontweight="bold"
    )

    direction = ["↓ better", "↓ better", "↑ better (later = harder to crack)", "↓ better"]

    for ax, (metric, vals), dirn in zip(axes, metrics.items(), direction):
        bars = ax.bar([LABELS.get(c, c) for c in conds], vals,
                      color=[COLORS.get(c, "gray") for c in conds],
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", fontsize=9)
        ax.set_title(f"{metric}\n{dirn}", fontweight="bold", fontsize=9)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.set_ylim(0, max(vals + [0.1]) * 1.4 + 0.05)

    plt.tight_layout()
    path = os.path.join(SIM_DIR, "figD_summary_comparison.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# DEFERRED — Per-probe figures (move here from probe_analysis_plots.py)
# Run only when N is large enough to support per-probe claims.
# Currently N=7 crime probes per condition — too small for individual-probe analysis.
# ═════════════════════════════════════════════════════════════════════════════

def fig_deferred_crime_mass_by_probe():
    """
    DEFERRED: Crime mass per probe × condition (was fig2 in probe_analysis_plots).
    Each bar = single observation. Needs larger N before drawing conclusions.
    """
    probes = probe_crime_probe_ids()
    PROBE_SHORT = {
        "C1": "Vulnerable\ntarget", "C2": "Economic\nlure",
        "C3": "Survival\npressure", "C5": "Ideological\nframing",
        "C6": "Retaliation", "C7": "Blockade\nthreat", "C8": "Outsource\ncrime",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(probes))
    w = 0.2
    for i, cond in enumerate(PROBE_CONDITIONS):
        ax.bar(x + i*w, [probe_crime_mass(cond, p) for p in probes], w,
               color=COLORS[cond], edgecolor="white", linewidth=0.5,
               label=LABELS[cond])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([PROBE_SHORT.get(p, p) for p in probes], fontsize=9)
    ax.set_ylabel("Crime-token probability mass  (↓ better)")
    ax.set_title("DEFERRED — Crime-Token Mass by Probe\n"
                 "Each bar = single observation (N=1). Needs larger N.",
                 fontsize=11, fontweight="bold", color="gray")
    ax.legend(handles=patches(PROBE_CONDITIONS), fontsize=9)
    plt.tight_layout()
    path = os.path.join(PROBE_DIR, "fig_deferred_crime_mass_by_probe.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def fig_deferred_crime_type_breakdown():
    """DEFERRED: Per-probe per-crime-type breakdown (was fig3). N=1 per cell."""
    probes = probe_crime_probe_ids()
    PROBE_SHORT = {
        "C1":"Vulnerable", "C2":"Econ lure", "C3":"Survival",
        "C5":"Ideological", "C6":"Retaliation", "C7":"Blockade", "C8":"Outsource",
    }
    titles = ["STEAL", "INTIMIDATE", "RECRUIT"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("DEFERRED — Crime-Token Probability by Type  (N=1 per cell, needs larger N)",
                 fontsize=11, fontweight="bold", color="gray")
    x = np.arange(len(probes))
    w = 0.2
    for ax, token, title in zip(axes, CRIME_TOOLS, titles):
        for i, cond in enumerate(PROBE_CONDITIONS):
            ax.bar(x + i*w, [probe_get_prob(cond, p, token) for p in probes],
                   w, color=COLORS[cond], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels([PROBE_SHORT.get(p, p) for p in probes], fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Token probability")
        ax.legend(handles=patches(PROBE_CONDITIONS), fontsize=7)
    plt.tight_layout()
    path = os.path.join(PROBE_DIR, "fig_deferred_crime_type_breakdown.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def fig_deferred_dpo_heatmap():
    """DEFERRED: DPO suppression heatmap (was fig4). Each cell = ratio of two single observations."""
    probes = probe_crime_probe_ids()
    PROBE_SHORT = {
        "C1":"Vulnerable", "C2":"Econ lure", "C3":"Survival",
        "C5":"Ideological", "C6":"Retaliation", "C7":"Blockade", "C8":"Outsource",
    }
    matrix = np.zeros((3, len(probes)))
    for j, p in enumerate(probes):
        for i, token in enumerate(CRIME_TOOLS):
            b = probe_get_prob("baseline", p, token)
            d = probe_get_prob("dpo_refusal", p, token)
            matrix[i, j] = d / b if b > 1e-6 else 1.0

    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(matrix.max(), 1.05))
    fig, ax = plt.subplots(figsize=(13, 3.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([f"{p}\n{PROBE_SHORT.get(p,p)}" for p in probes], fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["STEAL", "INTIMIDATE", "RECRUIT"], fontsize=10, fontweight="bold")
    ax.set_title("DEFERRED — DPO vs Baseline Ratio  (each cell = N=1, needs larger N)",
                 fontsize=10, fontweight="bold", color="gray")
    for i in range(3):
        for j in range(len(probes)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}×", ha="center", va="center", fontsize=9,
                    color="white" if val < 0.4 or val > 1.35 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="DPO ÷ Baseline", fraction=0.03, pad=0.02)
    plt.tight_layout()
    path = os.path.join(PROBE_DIR, "fig_deferred_dpo_heatmap.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def fig_deferred_scramble_proxy():
    """DEFERRED: Scramble proxy label vs concept test (was fig7). N=1 matched pair per scenario."""
    pairs = [("C1","S2","Steal"), ("C6","S6","Intimidate"), ("C8","S7","Recruit")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("DEFERRED — Scrambled vs Real Names  (N=1 per pair, needs larger N)",
                 fontsize=10, fontweight="bold", color="gray")
    for ax, (cp, sp, desc) in zip(axes, pairs):
        x = np.arange(len(PROBE_CONDITIONS))
        w = 0.32
        rv = [probe_crime_mass(c, cp) for c in PROBE_CONDITIONS]
        sv = [probe_scramble_crime_mass(c, sp) for c in PROBE_CONDITIONS]
        ax.bar(x - w/2, rv, w, color=[COLORS[c] for c in PROBE_CONDITIONS], edgecolor="white")
        ax.bar(x + w/2, sv, w, color=[COLORS[c] for c in PROBE_CONDITIONS],
               edgecolor="white", alpha=0.45, hatch="///")
        for bar, v in zip(ax.patches[:len(PROBE_CONDITIONS)], rv):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        for bar, v in zip(ax.patches[len(PROBE_CONDITIONS):], sv):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha="center", fontsize=8, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in PROBE_CONDITIONS], fontsize=8.5)
        ax.set_title(desc, fontweight="bold")
        ax.set_ylabel("Crime-token probability mass")
    fig.legend(handles=patches(PROBE_CONDITIONS) +
               [mpatches.Patch(color="gray", label="Real names"),
                mpatches.Patch(facecolor="gray", hatch="///", alpha=0.5, label="Greek proxies")],
               loc="lower center", ncol=6, fontsize=8)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(PROBE_DIR, "fig_deferred_scramble_proxy.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Probe vs Simulation Figures ===")
    fig_probe_vs_sim_crime_rate()
    fig_sim_crime_over_time()
    fig_recruitment_vs_crime()
    fig_summary_comparison()

    print("\n=== Deferred Per-Probe Figures (run when N is larger) ===")
    fig_deferred_crime_mass_by_probe()
    fig_deferred_crime_type_breakdown()
    fig_deferred_dpo_heatmap()
    fig_deferred_scramble_proxy()

    print("\nDone.")
