"""Offline lessons distillation for Agent B ("Rubinstein-RT").

Reads trajectories_b.jsonl (Agent B's OWN match logs — full self-bootstrap,
no data from Agent A), computes deterministic aggregate statistics per game
family, and renders them into lessons.txt via a fixed template. Agent B
hot-reloads lessons.txt and injects it into every LLM call.

Deliberately template-based (no LLM in the loop) so the distillation is
reproducible and describable in the paper. Run manually or on a schedule:

    python distill_lessons.py            # writes lessons.txt
    python distill_lessons.py --dry-run  # print to stdout only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_B_PATH = os.path.join(_DIR, "trajectories_b.jsonl")
LESSONS_PATH = os.path.join(_DIR, "lessons.txt")

MIN_GAMES = 5   # below this per family, say "insufficient data" instead of noise


def load_games() -> dict[str, list[dict]]:
    """game_id -> records (chronological). Skips corrupt lines."""
    games: dict[str, list[dict]] = defaultdict(list)
    if not os.path.exists(TRAJ_B_PATH):
        return games
    with open(TRAJ_B_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("game_id"):
                games[rec["game_id"]].append(rec)
    return games


def _last(recs: list[dict]) -> dict:
    return recs[-1]


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ----------------------------------------------------------------------------
# Per-family statistics
# ----------------------------------------------------------------------------

def stats_bargaining(games: list[list[dict]]) -> list[str]:
    n = len(games)
    if n < MIN_GAMES:
        return [f"bargaining: insufficient data ({n} games logged)."]
    my_shares, opp_ask_shares, accepts_seen, rounds_seen = [], [], 0, []
    my_offer_accepted = 0
    for recs in games:
        last = _last(recs)
        state = last["game_state"]
        money = state.get("money_to_divide") or 0
        me = last["your_player"]
        history = state.get("history") or []
        rounds_seen.append(state.get("round", len(history) + 1))
        for h in history:
            offer = h.get("offer") or {}
            proposer = h.get("proposer") or offer.get("proposer")
            if not money or not proposer:
                continue
            gain = offer.get(f"{proposer}_gain")
            if gain is None:
                continue
            if proposer == me:
                my_shares.append(gain / money)
                if h.get("decision") == "accept":
                    my_offer_accepted += 1
                    accepts_seen += 1
            else:
                opp_ask_shares.append(gain / money)
                if h.get("decision") == "accept":
                    accepts_seen += 1
    out = [f"bargaining ({n} games observed):"]
    if opp_ask_shares:
        out.append(f"- Opponents' own offers demand a median "
                   f"{_pct(statistics.median(opp_ask_shares))} of the pot for themselves; "
                   f"asks far above that band are routinely rejected.")
    if my_shares:
        acc = my_offer_accepted / max(1, len(my_shares))
        out.append(f"- Your offers asked a median {_pct(statistics.median(my_shares))} "
                   f"for yourself and were accepted {_pct(acc)} of the time"
                   + (" — price closer to the opponents' demonstrated band."
                      if acc < 0.25 else "."))
    out.append(f"- Median game length observed: {statistics.median(rounds_seen):.0f} rounds; "
               f"deadlocks pay 0 — converge before patience runs out.")
    return out


def stats_negotiation(games: list[list[dict]]) -> list[str]:
    n = len(games)
    if n < MIN_GAMES:
        return [f"negotiation: insufficient data ({n} games logged)."]
    accepted_ratios, seller_games, buyer_games = [], 0, 0
    for recs in games:
        last = _last(recs)
        state = last["game_state"]
        me = last["your_player"]
        role = state.get(f"{me}_role")
        V = state.get(f"{me}_value")
        if role == "seller":
            seller_games += 1
        elif role == "buyer":
            buyer_games += 1
        move = last.get("move") or {}
        if move.get("decision") == "AcceptOffer" and V:
            price = (state.get("last_offer") or {}).get("price")
            if price:
                accepted_ratios.append(price / V)
    out = [f"negotiation ({n} games observed; {seller_games} as seller, "
           f"{buyer_games} as buyer):"]
    if accepted_ratios:
        out.append(f"- Deals you accepted closed at a median "
                   f"{statistics.median(accepted_ratios):.2f}x your own valuation.")
    out.append("- Never accept a loss-making price; late in the game any profitable "
               "deal beats walking away (payoffs are undiscounted).")
    return out


def stats_persuasion(games: list[list[dict]]) -> list[str]:
    n = len(games)
    if n < MIN_GAMES:
        return [f"persuasion: insufficient data ({n} games logged)."]
    pos_seen = pos_high = 0
    buy_rate_num = buy_rate_den = 0
    for recs in games:
        last = _last(recs)
        state = last["game_state"]
        me = last["your_player"]
        if state.get(f"{me}_role") != "buyer" and state.get("player_2_role"):
            # role fields exist; skip non-buyer perspective for honesty stats
            pass
        for h in state.get("history") or []:
            if "buyer_decision" in h:
                buy_rate_den += 1
                if h.get("bought"):
                    buy_rate_num += 1
            q = h.get("quality")
            if q is not None:   # revealed only after a purchase
                msg = str(h.get("seller_message", "")).lower()
                positive = msg in ("yes",) or any(
                    w in msg for w in ("buy", "high", "recommend", "worth"))
                if positive:
                    pos_seen += 1
                    pos_high += q == "high"
    out = [f"persuasion ({n} games observed):"]
    if pos_seen:
        out.append(f"- Across revealed purchases, sellers' positive signals were honest "
                   f"{_pct(pos_high / pos_seen)} of the time ({pos_high}/{pos_seen}) — "
                   f"calibrate trust accordingly.")
    if buy_rate_den:
        out.append(f"- Observed overall buy rate: {_pct(buy_rate_num / buy_rate_den)}.")
    out.append("- As seller: lying above the buyer-indifference rate collapses future "
               "sales; as buyer: probe early so the posterior can move off the prior.")
    return out


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------

def render(games: dict[str, list[dict]]) -> str:
    by_family: dict[str, list[list[dict]]] = defaultdict(list)
    fallbacks = total_moves = 0
    for recs in games.values():
        fam = recs[-1].get("game_family")
        if fam:
            by_family[fam].append(recs)
        for r in recs:
            total_moves += 1
            if (r.get("llm") or {}).get("fallback"):
                fallbacks += 1

    header = [
        f"# Lessons v{time.strftime('%Y-%m-%dT%H:%M:%S')} — distilled from "
        f"{len(games)} of Agent B's own games ({total_moves} moves, "
        f"{_pct(fallbacks / total_moves) if total_moves else '0%'} rule-fallback).",
        "",
    ]
    if not games:
        return "\n".join(header + [
            "No self-play data yet (cold start). Rely on the game rules and sound "
            "game-theoretic reasoning: open ambitious but sellable, concede on a "
            "curve, never trade at a loss, close any profitable deal near the end, "
            "and protect your credibility in repeated persuasion.",
        ])

    body: list[str] = []
    for fam, fn in (("bargaining", stats_bargaining),
                    ("negotiation", stats_negotiation),
                    ("persuasion", stats_persuasion)):
        body.extend(fn(by_family.get(fam, [])))
        body.append("")
    return "\n".join(header + body).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args()

    text = render(load_games())
    if args.dry_run:
        print(text)
        return
    tmp = LESSONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, LESSONS_PATH)   # atomic swap — agent_b hot-reloads by mtime
    print(f"wrote {LESSONS_PATH} ({len(text)} chars)")


if __name__ == "__main__":
    main()
