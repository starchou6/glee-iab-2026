"""Compute the paper's comparison metrics for Agent A vs Agent B.

Streams each trajectories file once (A's is ~3GB), reducing every record to a
small per-game snapshot (records are chronological per game, so the last
record seen carries the fullest history). Joins with results_{a,b}.jsonl
(authoritative outcomes + opponent type from backfill_results.py) when
present. Writes:

    metrics_summary.json   — all headline numbers, keyed [agent][family]
    curves_bargaining.csv  — per (agent, proposer, round): mean share asked
    latency_b.csv          — per-move LLM latency/tokens for Agent B

Usage:
    python compare_metrics.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    # Result files are merged in order — later files overwrite earlier ones
    # (API backfill carries opponent metadata, so it wins over the log parse).
    "A": (os.path.join(_DIR, "trajectories.jsonl"),
          [os.path.join(_DIR, "results_a_log.jsonl"), os.path.join(_DIR, "results_a.jsonl")]),
    "B": (os.path.join(_DIR, "trajectories_b.jsonl"),
          [os.path.join(_DIR, "results_b.jsonl")]),
}
OUT_SUMMARY = os.path.join(_DIR, "metrics_summary.json")
OUT_CURVES = os.path.join(_DIR, "curves_bargaining.csv")
OUT_LATENCY = os.path.join(_DIR, "latency_b.csv")

MAX_CURVE_ROUND = 12

# Freeze the paper analysis to the two deployment windows reported in the
# manuscript.  The live agents continued writing trajectories after these
# dates, so an explicit cutoff keeps every regeneration on the same cohort.
ANALYSIS_CUTOFFS = {
    "A": "2026-08-25 22:25:27",
    "B": "2026-08-24 09:17:01",
}


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


# ----------------------------------------------------------------------------
# Streaming reduction
# ----------------------------------------------------------------------------

def reduce_record(rec: dict) -> dict:
    """Small per-game snapshot; replaces the previous one for the same game."""
    fam = rec.get("game_family")
    state = rec.get("game_state") or {}
    me = rec.get("your_player")
    snap = {"family": fam, "me": me, "ts": rec.get("ts"),
            "final_move": rec.get("move"), "round": state.get("round")}

    if fam == "bargaining":
        money = state.get("money_to_divide") or 0
        hist = []
        for h in state.get("history") or []:
            offer = h.get("offer") or {}
            proposer = h.get("proposer") or offer.get("proposer")
            gain = offer.get(f"{proposer}_gain") if proposer else None
            if money and gain is not None:
                hist.append({"r": h.get("round"), "mine": proposer == me,
                             "share": gain / money, "dec": h.get("decision"),
                             "rt": h.get("response_time_ms")})
        snap.update(money=money, hist=hist,
                    d1=state.get("delta_1"), d2=state.get("delta_2"),
                    horizon=state.get("max_rounds") if state.get("horizon_known", True) else None,
                    msgs=state.get("messages_allowed"))
        last = state.get("last_offer") or {}
        if last:
            gain = last.get(f"{me}_gain")
            snap["last_offer_share"] = (gain / money) if (money and gain is not None) else None

    elif fam == "negotiation":
        role = state.get(f"{me}_role")
        V = state.get(f"{me}_value")
        offers = []
        for h in state.get("history") or []:
            offer = h.get("offer") or h
            frm = offer.get("from_player") or h.get("proposer")
            price = offer.get("price")
            if price is not None and V:
                offers.append({"r": h.get("round"), "mine": frm == me,
                               "ratio": price / V})
        snap.update(role=role, V=V, offers=offers,
                    complete=state.get("complete_information"),
                    last_price=(state.get("last_offer") or {}).get("price"))

    elif fam == "persuasion":
        role = state.get(f"{me}_role")
        rounds = []
        for h in state.get("history") or []:
            rounds.append({"bought": h.get("bought"),
                           "q": h.get("quality"),
                           "sold_q": h.get("current_quality") or h.get("quality")})
        snap.update(role=role, rounds=rounds, p=state.get("p"),
                    price=state.get("product_price"),
                    v=state.get("v"), u=state.get("u"),
                    T=state.get("total_rounds"))
        # Seller-side honesty needs our own message vs current_quality per turn,
        # accumulated across records — handled by caller via `snap_extra`.

    return snap


def bargaining_offers(snap: dict) -> list[dict]:
    """Include terminal offers omitted from pre-action state histories.

    History entries, the outstanding opponent offer, and our final produced
    offer are merged by round. Produced offers are logged before submission,
    so acceptance is determined separately from authoritative terminal results.
    """
    offers = {h['r']: dict(h) for h in snap.get('hist') or []}
    money, me = snap.get('money'), snap.get('me')
    last = snap.get('last_offer') or {}
    proposer = last.get('proposer')
    gain = last.get(f'{proposer}_gain') if proposer else None
    if money and gain is not None and last.get('round') not in offers:
        offers[last['round']] = {'r': last['round'], 'mine': proposer == me,
                                 'share': gain / money, 'dec': None, 'rt': None}
    if snap.get('action_type') == 'offer' and money:
        move = snap.get('final_move') or {}
        key = 'alice_gain' if me == 'player_1' else 'bob_gain'
        gain = move.get(key)
        if gain is not None and snap.get('round') not in offers:
            offers[snap['round']] = {'r': snap['round'], 'mine': True,
                                    'share': gain / money, 'dec': None, 'rt': None}
    return sorted(offers.values(), key=lambda h: h.get('r') or 0)


def stream(agent: str, traj_path: str):
    games: dict[str, dict] = {}
    move_stats = defaultdict(lambda: {"moves": 0})
    llm_lat, llm_rows = [], []
    fallback = Counter()
    llm_tokens_in, llm_tokens_out, llm_cache = [], [], []
    llm_cost_total = llm_wall_ms = 0.0
    seller_lies = Counter()   # (family-specific extra) persuasion seller honesty
    ts_min = ts_max = None

    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fam, gid = rec.get("game_family"), rec.get("game_id")
            if not fam or not gid:
                continue
            ts = rec.get("ts")
            cutoff = ANALYSIS_CUTOFFS.get(agent)
            if cutoff and ts and ts > cutoff:
                continue
            ms = move_stats[fam]
            ms["moves"] += 1
            if ts:
                ts_min = ts if ts_min is None or ts < ts_min else ts_min
                ts_max = ts if ts_max is None or ts > ts_max else ts_max

            llm = rec.get("llm")
            if llm:
                if llm.get("fallback"):
                    reason = (llm.get("fallback_reason") or "?").split(":")[0]
                    fallback[reason] += 1
                else:
                    if llm.get("cost_usd_equiv"):
                        llm_cost_total += llm["cost_usd_equiv"]
                    lat = llm.get("latency_ms")
                    if lat:
                        llm_lat.append(lat)
                        llm_wall_ms += lat
                        llm_rows.append((fam, rec["valid_actions"]["type"], lat,
                                         llm.get("input_tokens"), llm.get("output_tokens"),
                                         llm.get("cache_read_input_tokens")))
                    if llm.get("input_tokens"):
                        llm_tokens_in.append(llm["input_tokens"])
                    if llm.get("output_tokens"):
                        llm_tokens_out.append(llm["output_tokens"])
                    if llm.get("cache_read_input_tokens") is not None:
                        llm_cache.append(llm["cache_read_input_tokens"])

            # Persuasion seller honesty: our move vs current_quality, per turn.
            if fam == "persuasion":
                state = rec.get("game_state") or {}
                me = rec.get("your_player")
                if state.get(f"{me}_role") == "seller":
                    qual = state.get("current_quality")
                    move = rec.get("move") or {}
                    sig = move.get("decision")
                    if sig is None and isinstance(move.get("message"), str):
                        m = move["message"].lower()
                        neg_hit = any(w in m for w in
                                      ("sit this", "skip", "pass", "not worth", "sit out"))
                        sig = "no" if neg_hit else "yes"
                    if qual in ("high", "low") and sig in ("yes", "no"):
                        seller_lies[(qual, sig)] += 1

            snap = reduce_record(rec)
            if fam == 'bargaining':
                snap['action_type'] = (rec.get('valid_actions') or {}).get('type')
                snap['last_offer'] = (rec.get('game_state') or {}).get('last_offer')
            games[gid] = snap

    return {"games": games, "move_stats": dict(move_stats),
            "llm_lat": llm_lat, "llm_rows": llm_rows, "fallback": fallback,
            "tok_in": llm_tokens_in, "tok_out": llm_tokens_out, "cache": llm_cache,
            "llm_cost_total": llm_cost_total, "llm_wall_ms": llm_wall_ms,
            "seller_lies": seller_lies, "ts_min": ts_min, "ts_max": ts_max}


def load_results(path: str) -> dict[str, dict]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("result"):
                out[r["game_id"]] = r
    return out


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def summarize(agent: str, data: dict, results: dict) -> dict:
    games = data["games"]
    fam_games = defaultdict(list)
    for gid, snap in games.items():
        fam_games[snap["family"]].append((gid, snap))

    out = {"span": [data["ts_min"], data["ts_max"]]}
    curves = []

    for fam, lst in fam_games.items():
        s: dict = {"games": len(lst), "moves": data["move_stats"][fam]["moves"]}
        joined = [(gid, snap, results.get(gid)) for gid, snap in lst]
        with_res = [(g, sn, r) for g, sn, r in joined if r]
        s["games_with_result"] = len(with_res)

        # Outcomes and normalized payoffs from authoritative results
        if with_res:
            outc = Counter((r["result"].get("outcome") or "?") for _, _, r in with_res)
            s["outcomes"] = dict(outc)
            opp_types = Counter((r.get("opponent") or {}).get("type", "?")
                                for _, _, r in with_res)
            s["opponent_types"] = dict(opp_types)
            payoffs, by_opp = [], defaultdict(list)
            for _, sn, r in with_res:
                me = r.get("your_player") or sn["me"]
                pay = r["result"].get(f"{me}_payoff")
                if pay is None:
                    continue
                norm = None
                if fam == "bargaining" and sn.get("money"):
                    norm = pay / sn["money"]
                elif fam == "negotiation" and sn.get("V"):
                    norm = pay / abs(sn["V"])
                elif fam == "persuasion":
                    T = sn.get("T") or 20
                    denom = (sn.get("price") or 1) * T
                    norm = pay / denom if denom else None
                if norm is not None:
                    payoffs.append(norm)
                    by_opp[(r.get("opponent") or {}).get("type", "?")].append(norm)
            if payoffs:
                s["norm_payoff_mean"] = statistics.mean(payoffs)
                s["norm_payoff_median"] = statistics.median(payoffs)
                if len(payoffs) > 1:
                    s["norm_payoff_se"] = (statistics.stdev(payoffs)
                                           / len(payoffs) ** 0.5)
                s["norm_payoff_by_opponent"] = {
                    k: {"n": len(v), "mean": statistics.mean(v)}
                    for k, v in by_opp.items()}
            agree = [r["result"] for _, _, r in with_res
                     if r["result"].get("outcome") == "agreement"]
            if agree and fam != "persuasion":
                s["agreement_rate"] = len(agree) / len(with_res)
                rounds = [a.get("agreed_round") for a in agree if a.get("agreed_round")]
                if rounds:
                    s["agreed_round_median"] = statistics.median(rounds)

        # Family-specific behavior from trajectories
        if fam == "bargaining":
            # Concession curves keyed by the proposer's n-th proposal (not the
            # absolute round — alternating offers make rounds parity-confounded).
            mine_by_ord = defaultdict(list)
            opp_by_ord = defaultdict(list)
            opp_rt, my_offer_n, my_offer_acc = [], 0, 0
            history_my_offer_n = history_my_offer_acc = 0
            own_agreements = opponent_agreements = unresolved_agreements = 0
            opp_shares = []
            opp_offer_n = 0
            for gid, sn in lst:
                nth = {"mine": 0, "opp": 0}
                history_my_offer_n += sum(h['mine'] for h in sn.get('hist') or [])
                history_my_offer_acc += sum(h['mine'] and h.get('dec') == 'accept'
                                            for h in sn.get('hist') or [])
                result = (results.get(gid) or {}).get('result') or {}
                agreed_round = result.get('agreed_round') if result.get('outcome') == 'agreement' else None
                own_proposer = None
                if agreed_round is not None:
                    own_proposer = ('player_1' if agreed_round % 2 else 'player_2') == sn['me']
                    own_agreements += bool(own_proposer)
                    opponent_agreements += not own_proposer
                elif result.get('outcome') == 'agreement':
                    unresolved_agreements += 1
                for h in bargaining_offers(sn):
                    share = h.get("share")
                    if share is None:
                        continue
                    if h["mine"]:
                        nth["mine"] += 1
                        if nth["mine"] <= MAX_CURVE_ROUND:
                            mine_by_ord[nth["mine"]].append(share)
                        my_offer_n += 1
                        my_offer_acc += bool(own_proposer and h['r'] == agreed_round)
                    else:
                        nth["opp"] += 1
                        opp_offer_n += 1
                        opp_shares.append(share)
                        if nth["opp"] <= MAX_CURVE_ROUND:
                            opp_by_ord[nth["opp"]].append(share)
                        if h.get("rt"):
                            opp_rt.append(h["rt"])
            s["my_offer_accept_rate"] = (my_offer_acc / my_offer_n) if my_offer_n else None
            s["my_offer_n"] = my_offer_n
            s['my_offer_accepted_n'] = my_offer_acc
            s['history_my_offer_n'] = history_my_offer_n
            s['history_my_offer_accepted_n'] = history_my_offer_acc
            s['agreements_on_own_offer'] = own_agreements
            s['agreements_on_opponent_offer'] = opponent_agreements
            s['agreements_with_unknown_proposer'] = unresolved_agreements
            s['opponent_offer_share_median'] = statistics.median(opp_shares) if opp_shares else None
            s["opp_offer_n"] = opp_offer_n
            s["first_offer_mine_mean"] = (statistics.mean(mine_by_ord[1])
                                          if mine_by_ord.get(1) else None)
            s["first_offer_mine_n"] = len(mine_by_ord.get(1) or [])
            s["opp_response_ms"] = {"p50": q(opp_rt, .5), "p90": q(opp_rt, .9),
                                    "n": len(opp_rt)}
            for r_ in sorted(mine_by_ord):
                curves.append([agent, "self", r_,
                               statistics.mean(mine_by_ord[r_]), len(mine_by_ord[r_])])
            for r_ in sorted(opp_by_ord):
                curves.append([agent, "opponent", r_,
                               statistics.mean(opp_by_ord[r_]), len(opp_by_ord[r_])])

        elif fam == "negotiation":
            roles = Counter(sn.get("role") for _, sn in lst)
            s["roles"] = dict(roles)
            for role in ("seller", "buyer"):
                sub = [(g, sn, r) for g, sn, r in joined if sn.get("role") == role and r]
                agr = [r["result"] for _, sn, r in sub
                       if r["result"].get("outcome") == "agreement"]
                ratios = []
                for _, sn, r in sub:
                    ap = r["result"].get("agreed_price")
                    if ap and sn.get("V"):
                        ratios.append(ap / sn["V"])
                s[f"{role}_agreed_price_over_V_median"] = (
                    statistics.median(ratios) if ratios else None)
                s[f"{role}_agreement_rate"] = (len(agr) / len(sub)) if sub else None

        elif fam == "persuasion":
            lies = data["seller_lies"]
            hi = lies[("high", "yes")] + lies[("high", "no")]
            lo = lies[("low", "yes")] + lies[("low", "no")]
            s["seller_pos_rate_when_high"] = (lies[("high", "yes")] / hi) if hi else None
            s["seller_lie_rate_when_low"] = (lies[("low", "yes")] / lo) if lo else None
            s["seller_high_n"] = hi
            s["seller_low_n"] = lo
            buys = tot = 0
            for _, sn in lst:
                if sn.get("role") == "buyer":
                    for r_ in sn.get("rounds") or []:
                        if r_.get("bought") is not None:
                            tot += 1
                            buys += bool(r_["bought"])
            s["buyer_buy_rate"] = (buys / tot) if tot else None
            s["buyer_decisions_n"] = tot

        out[fam] = s

    # LLM cost/latency (B only)
    if data["llm_lat"]:
        out["llm"] = {
            "calls": len(data["llm_lat"]),
            "fallbacks": sum(data["fallback"].values()),
            "fallback_reasons": dict(data["fallback"]),
            "fallback_rate": sum(data["fallback"].values()) /
                             (len(data["llm_lat"]) + sum(data["fallback"].values())),
            "latency_ms": {"p50": q(data["llm_lat"], .5), "p90": q(data["llm_lat"], .9),
                           "p99": q(data["llm_lat"], .99)},
            "input_tokens_mean": statistics.mean(data["tok_in"]) if data["tok_in"] else None,
            "output_tokens_mean": statistics.mean(data["tok_out"]) if data["tok_out"] else None,
            "cache_read_mean": statistics.mean(data["cache"]) if data["cache"] else None,
            "cost_usd_total": data["llm_cost_total"],
            "wall_hours_total": data["llm_wall_ms"] / 3.6e6,
        }
    return out, curves


def main() -> None:
    summary, all_curves = {}, []
    for agent, (traj, res_files) in FILES.items():
        print(f"streaming {agent}: {traj}", flush=True)
        data = stream(agent, traj)
        results: dict[str, dict] = {}
        for rf in res_files:
            results.update(load_results(rf))
        print(f"  {sum(1 for _ in data['games'])} games, "
              f"{len(results)} with backfilled results", flush=True)
        summary[agent], curves = summarize(agent, data, results)
        all_curves.extend(curves)
        if agent == "B" and data["llm_rows"]:
            with open(OUT_LATENCY, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["family", "action", "latency_ms",
                            "input_tokens", "output_tokens", "cache_read"])
                w.writerows(data["llm_rows"])

    with open(OUT_CURVES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "proposer", "round", "mean_share", "n"])
        w.writerows(all_curves)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {OUT_SUMMARY}, {OUT_CURVES}, {OUT_LATENCY}", flush=True)


if __name__ == "__main__":
    main()
