"""Render metrics_summary.json into paper/numbers.tex LaTeX macros.

Every number quoted in the paper comes from here — regenerate after any
data refresh and the paper stays consistent:

    python compare_metrics.py && python make_numbers.py
"""
from __future__ import annotations

import json
import os
import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--offline', action='store_true',
                    help='Keep frozen paper inputs; do not request current leaderboard ratings.')
parser.add_argument('--revision-baseline',
                    help='Mark macro values changed from this original numbers.tex using revised().')
args = parser.parse_args()

_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(_DIR, "metrics_summary.json")
RATINGS = os.path.join(_DIR, "ratings_history.csv")
OUT = os.path.join(_DIR, "paper", "numbers.tex")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(SUMMARY, encoding="utf-8") as f:
    s = json.load(f)


def g(agent, fam, key, default=None):
    return (s.get(agent, {}).get(fam) or {}).get(key, default)


def fmt(x, nd=2):
    if x is None:
        return "--"
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def pct(x, nd=0):
    return "--" if x is None else f"{100 * x:.{nd}f}\\%"


macros: dict[str, str] = {}

fams = ("bargaining", "negotiation", "persuasion")
total_a = sum(g("A", f, "games", 0) for f in fams)
total_b = sum(g("B", f, "games", 0) for f in fams)
macros["NGamesA"] = f"{total_a:,}"
macros["NGamesB"] = f"{total_b:,}"
macros["NGamesTotal"] = f"{total_a + total_b:,}"

for agent in ("A", "B"):
    for fam, tag in (("bargaining", "Bar"), ("negotiation", "Neg"), ("persuasion", "Per")):
        macros[f"Games{tag}{agent}"] = fmt(g(agent, fam, "games"), 0)
        macros[f"Pay{tag}{agent}"] = fmt(g(agent, fam, "norm_payoff_mean"), 3)
        se = g(agent, fam, "norm_payoff_se")
        macros[f"PaySE{tag}{agent}"] = fmt(se, 3) if se else "--"
        macros[f"Agree{tag}{agent}"] = pct(g(agent, fam, "agreement_rate"))
        by_opp = g(agent, fam, "norm_payoff_by_opponent") or {}
        for otype in ("agent", "human"):
            o = by_opp.get(otype) or {}
            macros[f"Pay{tag}{agent}{otype.capitalize()}"] = fmt(o.get("mean"), 3)
            macros[f"N{tag}{agent}{otype.capitalize()}"] = fmt(o.get("n"), 0)

for agent in ("A", "B"):
    for fam, tag in (("bargaining", "Bar"), ("negotiation", "Neg"), ("persuasion", "Per")):
        outc = g(agent, fam, "outcomes") or {}
        n_res = g(agent, fam, "games_with_result") or 0
        if n_res:
            macros[f"Timeout{tag}{agent}"] = pct(outc.get("timeout", 0) / n_res, 2)
            macros[f"Walk{tag}{agent}"] = pct(outc.get("walked_away", 0) / n_res, 1)
    rt = g(agent, "bargaining", "opp_response_ms") or {}
    macros[f"OppRt{agent}"] = (f"{rt['p50'] / 1000:.1f}\\,s" if rt.get("p50") else "--")
    macros[f"AgreedRoundBar{agent}"] = fmt(g(agent, "bargaining", "agreed_round_median"), 0)
    macros[f"SellerMargin{agent}"] = fmt(g(agent, "negotiation",
                                           "seller_agreed_price_over_V_median"), 2)
    macros[f"SellerAgree{agent}"] = pct(g(agent, "negotiation", "seller_agreement_rate"))
    macros[f"SellerPosHigh{agent}"] = pct(g(agent, "persuasion", "seller_pos_rate_when_high"))

# C11: B-A payoff differences with normal-approximation 95% CIs
for fam, tag in (("bargaining", "Bar"), ("negotiation", "Neg"), ("persuasion", "Per")):
    ma, mb = g("A", fam, "norm_payoff_mean"), g("B", fam, "norm_payoff_mean")
    sa, sb = g("A", fam, "norm_payoff_se"), g("B", fam, "norm_payoff_se")
    if None in (ma, mb, sa, sb):
        continue
    diff = mb - ma
    half = 1.96 * (sa ** 2 + sb ** 2) ** 0.5
    # A difference that rounds to zero prints unsigned, not as "-0.000".
    macros[f"Diff{tag}"] = f"{diff:+.3f}" if round(diff, 3) else "0.000"
    macros[f"Diff{tag}PP"] = f"{100 * diff:+.1f}" if round(100 * diff, 1) else "0.0"
    macros[f"CI{tag}"] = f"[{diff - half:+.3f}, {diff + half:+.3f}]"

# C13: denominators for behavioral statistics
for agent in ("A", "B"):
    macros[f"Props{agent}"] = fmt(g(agent, "bargaining", "my_offer_n"), 0)
    macros[f"HistoryProps{agent}"] = fmt(g(agent, "bargaining", "history_my_offer_n"), 0)
    own = g(agent, 'bargaining', 'agreements_on_own_offer', 0)
    opp = g(agent, 'bargaining', 'agreements_on_opponent_offer', 0)
    macros[f"OwnAgreements{agent}"] = fmt(own, 0)
    macros[f"BarAgreements{agent}"] = fmt(own + opp, 0)
    macros[f"OwnAgreementShare{agent}"] = pct(own / (own + opp), 1) if own + opp else '--'
    macros[f"FirstOfferMean{agent}"] = pct(g(agent, 'bargaining', 'first_offer_mine_mean'))
    macros[f"FirstOfferN{agent}"] = fmt(g(agent, "bargaining", "first_offer_mine_n"), 0)
    macros[f"SellerLowN{agent}"] = fmt(g(agent, "persuasion", "seller_low_n"), 0)
    macros[f"SellerHighN{agent}"] = fmt(g(agent, "persuasion", "seller_high_n"), 0)
    macros[f"BuyerDecN{agent}"] = fmt(g(agent, "persuasion", "buyer_decisions_n"), 0)
    macros[f"NegResN{agent}"] = fmt(g(agent, "negotiation", "games_with_result"), 0)
    for fam, tag in (("bargaining", "Bar"), ("negotiation", "Neg"), ("persuasion", "Per")):
        macros[f"ResN{tag}{agent}"] = fmt(g(agent, fam, "games_with_result"), 0)
props_total = ((g("A", "bargaining", "my_offer_n") or 0)
               + (g("B", "bargaining", "my_offer_n") or 0))
macros["PropsTotal"] = f"{props_total:,}"
history_props_total = sum(g(a, 'bargaining', 'history_my_offer_n', 0) for a in ('A', 'B'))
macros['HistoryPropsTotal'] = f'{history_props_total:,}'

# P0-1: result coverage per family and worst-case missing-data bound
max_missing_shift_a = 0.0
for agent in ("A", "B"):
    for fam, tag in (("bargaining", "Bar"), ("negotiation", "Neg"), ("persuasion", "Per")):
        n, r = g(agent, fam, "games"), g(agent, fam, "games_with_result")
        if n and r is not None:
            macros[f"Cov{tag}{agent}"] = pct(r / n, 2)
            macros[f"Unresolved{tag}{agent}"] = f"{n - r:,}"
            if agent == "A":
                mean = g(agent, fam, "norm_payoff_mean")
                if mean is not None:
                    missing_fraction = 1 - r / n
                    max_missing_shift_a = max(
                        max_missing_shift_a,
                        missing_fraction * max(mean, 1 - mean),
                    )
nA, rA = g("A", "bargaining", "games"), g("A", "bargaining", "games_with_result")
mA = g("A", "bargaining", "norm_payoff_mean")
if nA and rA and mA is not None:
    f_miss = 1 - rA / nA
    macros["WorstBarA"] = f"{f_miss * max(mA, 1 - mA):.3f}"
macros["MaxMissingShiftA"] = f"{max_missing_shift_a:.4f}"

pos_b = g("B", "persuasion", "seller_pos_rate_when_high")
if pos_b is not None:
    macros["SellerWithholdB"] = pct(1 - pos_b)
ga, gb = (sum(g("A", f, "games", 0) for f in fams),
          sum(g("B", f, "games", 0) for f in fams))
macros["VolumeRatio"] = f"{ga / max(1, gb):.0f}"

macros["SellerLieA"] = pct(g("A", "persuasion", "seller_lie_rate_when_low"))
macros["SellerLieB"] = pct(g("B", "persuasion", "seller_lie_rate_when_low"))
macros["BuyRateA"] = pct(g("A", "persuasion", "buyer_buy_rate"))
macros["BuyRateB"] = pct(g("B", "persuasion", "buyer_buy_rate"))
macros["MyOfferAccA"] = pct(g("A", "bargaining", "my_offer_accept_rate", 0), 1)
macros["MyOfferAccB"] = pct(g("B", "bargaining", "my_offer_accept_rate", 0), 1)

llm = s.get("B", {}).get("llm") or {}
lat = llm.get("latency_ms") or {}
macros["LatencyBMedian"] = (f"{lat.get('p50', 0) / 1000:.1f}\\,s"
                            if lat.get("p50") else "--")
macros["LatencyBPNinety"] = (f"{lat.get('p90', 0) / 1000:.1f}\\,s"
                             if lat.get("p90") else "--")
macros["FallbackRateB"] = pct(llm.get("fallback_rate"), 1)
macros["LLMCallsB"] = fmt(llm.get("calls"), 0)
macros["TokInB"] = fmt(llm.get("input_tokens_mean"), 0)
macros["TokOutB"] = fmt(llm.get("output_tokens_mean"), 0)
# Measured from local Claude Code session transcripts (session_cost2.py),
# priced at public API rates (Opus $5/$25 per MTok, cache read 0.1x, write
# 1.25x). Includes retries/failed calls that trajectories_b.jsonl does not.
# Re-measured 2026-08-26 over each agent's full run.
RUNTIME_COST_B = 1158.0     # 63,238 headless CLI calls, Aug 14-24
TOTAL_CALLS_B = 63238
DESIGN_COST_A = 151.0       # monitoring/rule-edit session, Aug 11-21
DESIGN_COST_B = 86.0        # Agent B design session, Aug 14

macros["CostTotalB"] = f"\\${RUNTIME_COST_B:,.0f}"
macros["TotalCallsB"] = f"{TOTAL_CALLS_B:,}"
macros["DesignCostA"] = f"\\${DESIGN_COST_A:,.0f}"
macros["DesignCostB"] = f"\\${DESIGN_COST_B:,.0f}"
if total_b:
    macros["CostPerGameB"] = f"\\${RUNTIME_COST_B / total_b:.2f}"
if total_a:
    macros["AmortCostA"] = f"\\${DESIGN_COST_A / total_a:.3f}"
    macros["CostAtScaleB"] = (
        f"\\${RUNTIME_COST_B / max(1, total_b) * total_a / 1000:,.0f}k")
# C19: per-produced-move cost uses the full bill (incl. retries) over moves
calls = llm.get("calls") or 34104
macros["CostPerMoveB"] = f"\\${RUNTIME_COST_B / calls:.3f}"
hours = llm.get("wall_hours_total")
if hours:
    macros["LLMHoursB"] = f"{hours:.0f}"

# Final ratings (A from ratings_history.csv tail; B injected via stats call)
try:
    with open(RATINGS, encoding="utf-8") as f:
        last = f.readlines()[-1].strip().split(",")
    macros["RatBarA"] = f"{float(last[1]):,.0f}"
    macros["RatNegA"] = f"{float(last[2]):,.0f}"
    macros["RatPerA"] = f"{float(last[3]):,.0f}"
    macros["RGamesBarA"] = f"{int(last[4]):,}"
except (OSError, IndexError, ValueError):
    pass

# Agent B final ratings — refreshed by fetch in this script if key available
try:
    if args.offline:
        raise RuntimeError('offline regeneration uses the frozen analysis cohort')
    from glee_sdk import GleeClient
    key = os.environ.get("GLEE_API_KEY_B")
    scores = GleeClient(api_key=key).stats()["scores"]
    macros["RatBarB"] = f"{scores['bargaining']['rating']:,.0f}"
    macros["RatNegB"] = f"{scores['negotiation']['rating']:,.0f}"
    macros["RatPerB"] = f"{scores['persuasion']['rating']:,.0f}"
    macros["RGamesBarB"] = f"{scores['bargaining']['games_played']:,}"
except Exception as e:                                    # noqa: BLE001
    print(f"note: could not fetch B ratings ({e})")

baseline = {}
if args.revision_baseline:
    baseline = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}',
                              Path(args.revision_baseline).read_text(encoding='utf-8')))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("% Auto-generated by make_numbers.py — do not edit.\n")
    for name, val in sorted(macros.items()):
        if name in baseline and baseline[name] != val:
            val = f'\\revised{{{val}}}'
        f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
print(f"wrote {len(macros)} macros to {OUT}")
