"""Generate the paper's figures as PDF into paper/figs/.

Inputs: curves_bargaining.csv, latency_b.csv, metrics_summary.json
(from compare_metrics.py). Regenerate with:

    python compare_metrics.py && python make_figures.py

Colors: validated reference palette — slot 1 blue = Agent A, slot 2 orange
= Agent B (color follows the agent; linestyle distinguishes self/opponent).
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(_DIR, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

COL = 3.25  # single-column width (in)


# ---------------------------------------------------------------- fig 1
def fig_concession() -> None:
    rows = defaultdict(dict)   # (agent, proposer) -> {round: (share, n)}
    with open(os.path.join(_DIR, "curves_bargaining.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["agent"], r["proposer"])][int(r["round"])] = (
                float(r["mean_share"]), int(r["n"]))

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    style = {("A", "self"): (BLUE, "-", "Agent A asks"),
             ("A", "opponent"): (BLUE, "--", "A's opponents ask"),
             ("B", "self"): (ORANGE, "-", "Agent B asks"),
             ("B", "opponent"): (ORANGE, "--", "B's opponents ask")}
    for key, (color, ls, label) in style.items():
        pts = rows.get(key, {})
        # First three offers only: later ordinals are survivorship-skewed
        # (only unusual configs last that long) and read as noise.
        xs = sorted(r for r in pts if pts[r][1] >= 80 and r <= 3)
        if not xs:
            continue
        ax.plot(xs, [pts[x][0] * 100 for x in xs], ls, color=color,
                linewidth=1.6, label=label, marker="o", markersize=3)
    ax.axhline(50, color=MUTED, linewidth=0.6, linestyle=":")
    ax.annotate("50/50 split", xy=(0.99, 50.4), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Proposer's $n$-th offer in the game")
    ax.set_ylabel("Own share asked (%)")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(45, 82)
    ax.legend(frameon=False, ncol=2, loc="upper right",
              columnspacing=1.0, handlelength=1.8, fontsize=6.5)
    fig.savefig(os.path.join(FIGS, "concession.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_latency() -> None:
    lat = []
    with open(os.path.join(_DIR, "latency_b.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat.append(float(r["latency_ms"]) / 1000.0)
    lat.sort()
    n = len(lat)
    ys = [100 * (i + 1) / n for i in range(n)]

    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.semilogx(lat, ys, color=ORANGE, linewidth=1.6)
    ax.axvline(0.001, color=BLUE, linewidth=1.6)
    ax.text(0.0012, 45, "Agent A\n(rules, <1 ms)", fontsize=6.5,
            color=BLUE, va="center")
    med = lat[n // 2]
    ax.axvline(med, color=MUTED, linewidth=0.6, linestyle=":")
    ax.text(med * 1.15, 8, f"B median {med:.1f} s", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Decision latency (s, log scale)")
    ax.set_ylabel("Cumulative share of moves (%)")
    ax.set_xlim(5e-4, max(lat) * 1.3)
    fig.savefig(os.path.join(FIGS, "latency.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_payoff_by_opponent() -> None:
    with open(os.path.join(_DIR, "metrics_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    fams = ["bargaining", "negotiation", "persuasion"]
    groups = []   # (label, payoff_A_vs_agents, A_vs_humans, B_vs_agents, B_vs_humans)
    for fam in fams:
        row = [fam.capitalize()]
        for agent in ("A", "B"):
            by = (s.get(agent, {}).get(fam) or {}).get("norm_payoff_by_opponent") or {}
            row.append((by.get("agent") or {}).get("mean"))
            row.append((by.get("hidden") or {}).get("mean"))
        groups.append(row)

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    width = 0.19
    xs = range(len(groups))
    series = [("A vs. agents", BLUE, 1.0, -1.5), ("A vs. anon.", BLUE, 0.45, -0.5),
              ("B vs. agents", ORANGE, 1.0, 0.5), ("B vs. anon.", ORANGE, 0.45, 1.5)]
    for i, (label, color, alpha, off) in enumerate(series):
        vals = [g[1 + i] for g in groups]
        pos = [x + off * width for x in xs]
        ax.bar(pos, [v if v is not None else 0 for v in vals], width * 0.92,
               color=color, alpha=alpha, label=label, edgecolor="none")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Mean normalized payoff")
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIGS, "payoff_by_opponent.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_concession()
    fig_latency()
    fig_payoff_by_opponent()
    print(f"figures written to {FIGS}")
