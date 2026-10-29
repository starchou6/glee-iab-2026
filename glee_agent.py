"""GLEE Competition agent — "Rubinstein"

Three-family rule-based agent:
  * Bargaining : exact SPE solver (finite backward induction / Rubinstein
                 closed form) + Bayesian inference of the opponent's discount
                 factor + a concession curve that opens above equilibrium and
                 converges to it. Pragmatic clamps avoid spite-rejections.
  * Negotiation: reservation-value logic (never trade at a loss), last-mover
                 aware surplus targeting under complete information,
                 concession curve + opponent-movement tracking otherwise.
  * Persuasion : seller mixes honesty per the commitment-optimal lying rate
                 (kept strictly below the buyer-indifference boundary so
                 trusting us stays profitable); buyer tracks the seller's
                 empirical reliability and buys on posterior expected value.

Design notes:
  - Pure rules, no LLM calls: sub-millisecond moves, zero timeout risk
    (timeouts score at the 5th percentile — the worst outcome available).
  - Message field is attached only when game_state["messages_allowed"] is on.
  - Every move is wrapped in a safe-fallback so a bug never burns the
    5-invalid-move budget.

Usage:
    pip install glee-sdk
    export GLEE_API_KEY=glee_...      # PowerShell: $env:GLEE_API_KEY="glee_..."
    python glee_agent.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rubinstein")

# ----------------------------------------------------------------------------
# Shared: pragmatic limits.  Pure SPE says "demand 100%" in some configs; real
# opponents (humans, quirky LLMs) spite-reject that and we both score zero.
# ----------------------------------------------------------------------------
MAX_ASK = 0.85          # never demand more than 85% of the pot
MAX_THRESHOLD = 0.85    # never hold out for more than 85% as a responder
ENDGAME_SWEETENER = 0.06

DELTA_GRID = (0.7, 0.8, 0.9, 0.95, 1.0)   # candidate opponent discount factors


def _unit_hash(*parts) -> float:
    """Deterministic pseudo-random in [0,1) — reproducible mixing decisions."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) / 16**12


# ============================================================================
# 1. BARGAINING
# ============================================================================

def solve_finite(dA: float, dB: float, T: int) -> list[float]:
    """a[t] = Alice's SPE share in the subgame starting at round t.
    Alice proposes on odd rounds; the last proposer takes (almost) all."""
    a = [0.0] * (T + 2)
    a[T] = 1.0 if T % 2 == 1 else 0.0
    for t in range(T - 1, 0, -1):
        if t % 2 == 1:
            a[t] = 1.0 - dB * (1.0 - a[t + 1])
        else:
            a[t] = dA * a[t + 1]
    return a


def solve_infinite(dA: float, dB: float) -> tuple[float, float]:
    """(Alice's share when she proposes, when Bob proposes)."""
    denom = 1.0 - dA * dB
    if denom < 1e-9:                       # both perfectly patient: SPE not unique
        return 0.5, 0.5
    ap = (1.0 - dB) / denom
    return ap, dA * ap


@dataclass
class Solver:
    dA: float
    dB: float
    T: int | None = None

    def __post_init__(self):
        if self.T is None:
            self._ap, self._ar = solve_infinite(self.dA, self.dB)
        else:
            self._a = solve_finite(self.dA, self.dB, self.T)

    def alice_share(self, t: int) -> float:
        if self.T is None:
            return self._ap if t % 2 == 1 else self._ar
        if t > self.T:
            return 0.0
        return self._a[t]

    def eq_offer(self, t: int, me: str) -> float:
        a = self.alice_share(t)
        return a if me == "A" else 1.0 - a

    def accept_threshold(self, t: int, me: str) -> float:
        """Continuation value: what waiting one round is worth to me."""
        if self.T is not None and t >= self.T:
            return 0.0
        nxt = self.alice_share(t + 1)
        return self.dA * nxt if me == "A" else self.dB * (1.0 - nxt)


@dataclass
class DeltaEstimator:
    """Exact Bayes over the small candidate grid of opponent deltas."""
    my_delta: float
    me: str
    T: int | None = None
    sigma: float = 0.12
    logw: list = field(default_factory=lambda: [0.0] * len(DELTA_GRID))
    _cache: dict = field(default_factory=dict)

    def _solver(self, d_opp: float) -> Solver:
        if d_opp not in self._cache:
            dA, dB = (self.my_delta, d_opp) if self.me == "A" else (d_opp, self.my_delta)
            self._cache[d_opp] = Solver(dA, dB, self.T)
        return self._cache[d_opp]

    def observe_offer(self, t: int, opp_share_requested: float) -> None:
        opp = "B" if self.me == "A" else "A"
        for i, d in enumerate(DELTA_GRID):
            pred = self._solver(d).eq_offer(t, opp)
            self.logw[i] += -0.5 * ((opp_share_requested - pred) / self.sigma) ** 2

    def observe_rejection(self, t: int, opp_share_offered: float) -> None:
        opp = "B" if self.me == "A" else "A"
        for i, d in enumerate(DELTA_GRID):
            thr = self._solver(d).accept_threshold(t, opp)
            x = (thr - opp_share_offered) / 0.05
            self.logw[i] += math.log(1.0 / (1.0 + math.exp(-max(-60, min(60, x)))) + 1e-9)

    def mean(self) -> float:
        m = max(self.logw)
        w = [math.exp(x - m) for x in self.logw]
        z = sum(w)
        return sum(wi / z * d for wi, d in zip(w, DELTA_GRID))


class _LRU(OrderedDict):
    def __init__(self, cap=512):
        super().__init__()
        self.cap = cap

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


_BARG_MEM: _LRU = _LRU()      # game_id -> {"est": DeltaEstimator, "seen": set()}

# ----------------------------------------------------------------------------
# Hot-reloadable tuning: parameters can be changed in tuning.json while the
# agent keeps running — restarts abandon active games (timeout = 5th
# percentile score + queue ban), so live tweaks must never require one.
# ----------------------------------------------------------------------------
_TUNE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.json")
_TUNE_CACHE: dict = {"data": {}, "mtime": None}


def _tuning() -> dict:
    try:
        mt = os.path.getmtime(_TUNE_PATH)
        if mt != _TUNE_CACHE["mtime"]:
            with open(_TUNE_PATH, encoding="utf-8") as f:
                _TUNE_CACHE["data"] = json.load(f)
            _TUNE_CACHE["mtime"] = mt
            log.info("tuning reloaded: %s", _TUNE_CACHE["data"])
    except (OSError, ValueError):
        pass                                   # missing/broken file: keep last good
    return _TUNE_CACHE["data"]


def _barg_messages_for(kind: str, ctx: dict) -> str:
    """Short, truthful-where-verifiable strategic messages."""
    if kind == "open_strong":
        return ("Every round we wait, the pot shrinks more for you than for me. "
                "Settling now on these terms beats anything you can get later.")
    if kind == "open":
        return "Reasonable opening. A quick deal keeps the pot big for both of us."
    if kind == "converge":
        return "I've moved toward you. Delay costs us both — let's close this."
    if kind == "endgame":
        return "Last chance to walk away with something. This is better than zero."
    if kind == "final_offer":
        return "Take it — rejecting leaves both of us with nothing."
    return "Let's find a split we can both live with."


def bargaining_strategy(game: dict) -> dict:
    state = game["game_state"]
    money = state["money_to_divide"]
    t = state.get("round", 1)
    T = state.get("max_rounds") if state.get("horizon_known", True) else None
    msgs_ok = state.get("messages_allowed", True)
    gid = game["game_id"]

    # Who am I in solver terms?  player_1 = Alice (odd-round proposer).
    me_key = game["your_player"]                 # "player_1" | "player_2"
    me = "A" if me_key == "player_1" else "B"
    my_delta = state.get("delta_1" if me == "A" else "delta_2", 1.0)
    opp_delta = state.get("delta_2" if me == "A" else "delta_1")   # None if hidden

    mem = _BARG_MEM.get(gid)
    if mem is None:
        mem = {"est": DeltaEstimator(my_delta=my_delta, me=me, T=T), "seen": set()}
        _BARG_MEM.put(gid, mem)

    # Feed history into the estimator exactly once per event.
    if opp_delta is None:
        for h in state.get("history", []):
            r = h.get("round")
            offer = h.get("offer") or {}
            proposer = h.get("proposer") or offer.get("proposer")
            key = ("o", r)
            if proposer and proposer != me_key and key not in mem["seen"]:
                opp_gain = offer.get(f"{proposer}_gain")
                if opp_gain is not None and money:
                    mem["est"].observe_offer(r, opp_gain / money)
                    mem["seen"].add(key)
            key = ("r", r)
            if (proposer == me_key and h.get("decision") == "reject"
                    and key not in mem["seen"]):
                opp_key = "player_2" if me_key == "player_1" else "player_1"
                opp_off = offer.get(f"{opp_key}_gain")
                if opp_off is not None and money:
                    mem["est"].observe_rejection(r, opp_off / money)
                    mem["seen"].add(key)

    d_opp = opp_delta if opp_delta is not None else mem["est"].mean()
    dA, dB = (my_delta, d_opp) if me == "A" else (d_opp, my_delta)
    solver = Solver(dA, dB, T)

    if game["valid_actions"]["type"] == "offer":
        eq = solver.eq_offer(t, me)
        # Infinite horizon: pure theory can demand ~100% forever (e.g. delta=1)
        # and deadlock against a stubborn opponent — but no-deal is a bottom-
        # percentile score.  Erode the theoretical base toward a settle point
        # as rounds pass; a deal at round 12 with 65% beats a standoff at 0.
        if T is None:
            eq = max(0.30, eq * (0.5 + 0.5 * 0.9 ** (t - 1)))
        # Aggression scales with my patience and decays each round.
        alpha0 = min(0.9, max(0.15, (my_delta - 0.7) / 0.3 * 0.8))
        alpha = alpha0 * (0.55 ** (t - 1))
        if T is not None and t >= T - 1:
            alpha = 0.0                                  # converge before the cliff
        ask = eq + (1.0 - eq) * alpha
        # Log data (3595 agreements) showed opponents NEVER accepted our
        # offers — they were all priced out of range, so every deal closed on
        # the opponent's terms instead (~46.7% to us).  Opponents' own
        # counters demand 53-57%, so from round 3 keep our ask inside the
        # band they demonstrably settle in — sellable beats theoretical.
        if t >= 3:
            ask = min(ask, max(eq, 0.58))
        # Never propose below our own early-round accept floor — opponents
        # settle near 50/50 in practice even when SPE says we're the weak
        # side, so asking less than half just donates the difference.
        # (Same decaying floor as the responder side, kept consistent.)
        if t <= 6 and (T is None or t <= T - 2):
            tune = _tuning()
            k = tune.get("barg_wait_k", 0.026)
            lo = tune.get("barg_floor_lo", 0.32)
            hi = tune.get("barg_floor_hi", 0.54)
            wait_val = k * my_delta / (1.0 - my_delta) if my_delta < 0.999 else hi
            floor = min(hi, max(lo, wait_val))
            floor *= tune.get("barg_floor_decay", 0.95) ** (t - 1)
            ask = max(ask, floor + 0.03)
        if T is not None and t == T:
            ask = min(ask, 1.0 - ENDGAME_SWEETENER)      # sweetener beats spite
        ask = max(0.05, min(MAX_ASK, ask))               # pragmatic clamps

        my_gain = int(round(ask * money))
        my_gain = max(1, min(int(money) - 1, my_gain))
        opp_gain = int(money) - my_gain
        a_gain, b_gain = (my_gain, opp_gain) if me == "A" else (opp_gain, my_gain)

        kind = ("final_offer" if (T is not None and t == T)
                else "endgame" if (T is not None and t >= T - 2)
                else "open_strong" if (t == 1 and my_delta >= d_opp + 0.05)
                else "open" if t == 1 else "converge")
        move = {"alice_gain": a_gain, "bob_gain": b_gain}
        if msgs_ok:
            move["message"] = _barg_messages_for(kind, {})
        return move

    # ---- decision phase: I'm the receiver ----
    last = state.get("last_offer") or {}
    my_gain_offered = last.get(f"{me_key}_gain", 0)
    share = (my_gain_offered / money) if money else 0.0
    thr = solver.accept_threshold(t, me)
    thr = min(thr, MAX_THRESHOLD)
    if T is None:
        # Same anti-deadlock erosion on the receiving side: holding out for a
        # theoretical 85% forever converts to a zero.  Floor keeps us from
        # accepting crumbs early.
        thr = max(0.22, thr * (0.35 + 0.65 * 0.9 ** (t - 1)))
    # Early-round backbone: opponents settle near 50/50, so accepting their
    # 43-47% opener leaves money on the table.  But a FLAT floor made deals
    # close at round 4.7 instead of 3.0 (measured 08-16), and inflation ate
    # more than the share gained — so the floor now decays each round:
    # firm on the opener, quick to close after.
    tune = _tuning()
    if t <= 6 and (T is None or t <= T - 2):
        # Delta-sensitive floor: the value of waiting one more round is
        # roughly (expected opponent concession) * d/(1-d).  Impatient games
        # (d<=0.9) make waiting expensive — take an early fair-ish deal at
        # full value; patient games (d>=0.95) make holding out cheap.
        k = tune.get("barg_wait_k", 0.026)
        lo = tune.get("barg_floor_lo", 0.32)
        hi = tune.get("barg_floor_hi", 0.54)
        wait_val = k * my_delta / (1.0 - my_delta) if my_delta < 0.999 else hi
        floor = min(hi, max(lo, wait_val))
        floor *= tune.get("barg_floor_decay", 0.95) ** (t - 1)
        thr = max(thr, floor)
    # Responsiveness adaptation (measured 08-17): 54% of opponents never
    # improve their offer after a rejection — against them, extra rejects
    # only burn inflation, so take any roughly-fair deal now.  The 43% who
    # do concede (+6pp avg) are worth holding out against, so the floor
    # stays active while their offers keep moving toward us.
    prev = mem.get("prev_offer_share")
    mem["prev_offer_share"] = share
    if prev is not None and share - prev <= tune.get("barg_stubborn_eps", 0.01):
        thr = min(thr, tune.get("barg_stubborn_accept", 0.40))
    if T is not None and t >= T:
        return {"decision": "accept" if my_gain_offered > 0 else "reject"}
    if T is not None and t == T - 1:
        thr *= 0.9                                       # de-risk the cliff edge
    return {"decision": "accept" if share >= thr - 1e-9 else "reject"}
    # ("walkaway" is dominated: it locks in 0 while rejecting keeps option value.)


# ============================================================================
# 2. NEGOTIATION
# ============================================================================

_NEG_MIN_MARGIN = 0.02     # accept anything at least 2% of |V| profitable late


def _neg_rounds_left(state: dict) -> int | None:
    T = state.get("max_rounds") if state.get("horizon_known", True) else None
    return None if T is None else max(0, T - state.get("round", 1))


def negotiation_strategy(game: dict) -> dict:
    state = game["game_state"]
    me_key = game["your_player"]
    role = state[f"{me_key}_role"]                   # "seller" | "buyer"
    V = state[f"{me_key}_value"]
    opp_key = "player_2" if me_key == "player_1" else "player_1"
    V_opp = state.get(f"{opp_key}_value")            # None under incomplete info
    t = state.get("round", 1)
    T = state.get("max_rounds") if state.get("horizon_known", True) else None
    left = _neg_rounds_left(state)
    msgs_ok = state.get("messages_allowed", True)
    seller = role == "seller"

    # Complete information: is there surplus at all, and who moves last?
    if V_opp is not None:
        Vs, Vb = (V, V_opp) if seller else (V_opp, V)
        surplus = Vb - Vs
        if surplus <= 0:
            # No mutually beneficial price exists — the correct outcome is no
            # trade.  Never accept a loss; offer only prices profitable to us.
            if game["valid_actions"]["type"] == "offer":
                price = V * (1.10 if seller else 0.90)
                move = {"product_price": round(price, 2)}
                if msgs_ok:
                    move["message"] = "That's the only number that works for me."
                return move
            price = state["last_offer"]["price"]
            profitable = price > V if seller else price < V
            if profitable:
                return {"decision": "AcceptOffer"}
            if T is not None and t >= T:
                return {"decision": "RejectOffer"}       # final round: just end it
            return {"decision": "WalkAway"}              # no surplus: stop wasting turns
        # Surplus exists.  Without discounting, the last proposer captures it,
        # so target a surplus share based on who holds the final offer.
        i_move_last = None
        if T is not None:
            # Offers alternate; whoever proposes in round T moves last.
            proposer_now = state.get("current_player") if game["valid_actions"]["type"] == "offer" else opp_key
            i_propose_now = proposer_now == me_key
            same_parity = (T - t) % 2 == 0
            i_move_last = i_propose_now == same_parity
        hi = 0.88 if i_move_last else (0.55 if i_move_last is None else 0.40)
        lo = 0.25
        if T is None:
            frac_done = min(1.0, (t - 1) / 8.0)      # unknown horizon: settle by ~9 moves
        elif T <= 1:
            frac_done = 0.0
        else:
            frac_done = min(1.0, (t - 1) / (T - 1))
        target_share = hi - (hi - lo) * frac_done
        target_price = Vs + surplus * (target_share if seller else 1 - target_share)

        if game["valid_actions"]["type"] == "offer":
            move = {"product_price": round(target_price, 2)}
            if msgs_ok:
                move["message"] = ("We both know the numbers — this splits the "
                                   "gains and beats walking away for both of us.")
            return move
        price = state["last_offer"]["price"]
        accept_line = target_price if T is None else (
            Vs + surplus * ((lo if seller else 1 - lo) if t >= T else
                            (target_share if seller else 1 - target_share)))
        good = price >= accept_line if seller else price <= accept_line
        profitable = price > Vs if seller else price < Vb
        if good and profitable:
            return {"decision": "AcceptOffer"}
        if T is None and t >= 12 and profitable:
            return {"decision": "AcceptOffer"}       # stop grinding: profit > pride
        if T is not None and t >= T:
            return ({"decision": "AcceptOffer"} if profitable
                    else {"decision": "RejectOffer"})
        counter = round(target_price, 2)
        move = {"decision": "RejectOffer", "product_price": counter}
        if msgs_ok:
            move["message"] = "Closer, but the split isn't right yet. Meet me here."
        return move

    # ---------------- Incomplete information ----------------
    # We know only our own valuation.  Open ambitious, concede on a curve,
    # never ever trade at a loss, and grab any profitable deal near the end.
    frac = 0.0
    if T and T > 1:
        frac = min(1.0, (t - 1) / (T - 1))
    elif T is None:
        frac = min(1.0, (t - 1) / 8.0)                  # unknown horizon: settle by ~9 moves

    tune = _tuning()
    if seller:
        open_mult, close_mult = 1.65, 1.06
        ask = V * (open_mult - (open_mult - close_mult) * frac)
        slope = tune.get("neg_seller_accept_slope", 0.43)
        accept_line = V * max(1.0 + _NEG_MIN_MARGIN,
                              (1.45 - slope * frac))
    else:
        open_mult, close_mult = 0.55, 0.94
        ask = V * (open_mult + (close_mult - open_mult) * frac)
        slope = tune.get("neg_buyer_accept_slope", 0.36)
        accept_line = V * min(1.0 - _NEG_MIN_MARGIN,
                              (0.62 + slope * frac))

    if game["valid_actions"]["type"] == "offer":
        move = {"product_price": round(ask, 2)}
        if msgs_ok:
            move["message"] = ("Serious offer — priced to close quickly."
                               if t == 1 else
                               "I keep moving your way. Let's wrap this up.")
        return move

    price = state["last_offer"]["price"]
    profitable = price >= V * (1 + _NEG_MIN_MARGIN) if seller \
        else price <= V * (1 - _NEG_MIN_MARGIN)
    strictly_ok = price >= accept_line if seller else price <= accept_line
    log.info("neg-diag[%s] t=%s role=%s V=%.2f offer=%.2f line=%.2f prof=%s ok=%s",
             game["game_id"][:8], t, role, V, price, accept_line, profitable, strictly_ok)
    if strictly_ok and profitable:
        return {"decision": "AcceptOffer"}
    # Stop-loss against stubborn opponents on unknown horizons: past round 15
    # any profit beats pride.  (A WalkAway at round 25 was tried 08-12/13 and
    # measurably hurt: payoffs aren't discounted, so ending the game only
    # forfeits option value — 20% of games locked in 0 while the agreement
    # rate dropped.  Keep rejecting instead; a late deal still counts full.)
    if T is None and t >= 15 and profitable:
        return {"decision": "AcceptOffer"}
    if T is not None and t >= T:
        return ({"decision": "AcceptOffer"} if profitable
                else {"decision": "RejectOffer"})
    if T is not None and t == T - 1 and profitable:
        return {"decision": "AcceptOffer"}               # don't gamble the cliff
    move = {"decision": "RejectOffer", "product_price": round(ask, 2)}
    if msgs_ok:
        move["message"] = "Not there yet — here's a number I can actually do."
    return move


# ============================================================================
# 3. PERSUASION
# ============================================================================

_POS_WORDS = ("great", "excellent", "high", "top", "best", "recommend", "premium",
              "quality", "amazing", "fantastic", "worth", "good", "yes")
_NEG_WORDS = ("skip", "pass", "low", "not ", "don't", "avoid", "honest",
              "no ", "poor", "weak")


def _msg_is_positive(msg) -> bool:
    if msg is None:
        return False
    if isinstance(msg, dict):
        msg = msg.get("message") or msg.get("decision") or ""
    s = str(msg).lower()
    if s in ("yes", "no"):
        return s == "yes"
    pos = sum(w in s for w in _POS_WORDS)
    neg = sum(w in s for w in _NEG_WORDS)
    return pos >= neg


def persuasion_strategy(game: dict) -> dict:
    state = game["game_state"]
    price = state.get("product_price", 1.0) or 1.0
    p = state.get("p", 0.5)
    t = state.get("round", 1)
    T = state.get("total_rounds", 20)
    history = state.get("history", [])
    atype = game["valid_actions"]["type"]

    # ---------------- Seller ----------------
    if atype in ("seller_message", "seller_recommendation"):
        quality = state.get("current_quality", "low")
        v = state.get("v")                       # buyer's high value, if we see it
        v_ratio = (v / price) if v else 2.0      # conservative default
        # Buyer-indifference lying rate (Bayesian-persuasion commitment policy):
        #   q* = p (v/price - 1) / (1 - p);  stay 25% below it so trusting our
        # positive signals remains strictly profitable for the buyer.
        if p >= 1.0:
            q = 1.0
        else:
            q = max(0.0, min(1.0, p * max(0.0, v_ratio - 1.0) / (1.0 - p)))
        q *= 0.75
        # If the buyer has stopped buying recently, rebuild trust: full honesty.
        recent = history[-4:]
        if len(recent) >= 3 and not any(h.get("bought") for h in recent):
            q = 0.0
        # Last couple of rounds: no reputation left to protect — cash in.
        if T and t >= T - 1:
            q = 1.0

        if quality == "high":
            positive = True
        else:
            positive = _unit_hash(game["game_id"], t, "lie") < q
        log.info("per-diag[%s] t=%s/%s SELLER quality=%s q=%.2f positive=%s",
                 game["game_id"][:8], t, T, quality, q, positive)

        if atype == "seller_recommendation":
            return {"decision": "yes" if positive else "no"}
        if positive:
            return {"message": ("This one is genuinely high quality — inspect my "
                                "track record in this chat: when I say buy, it pays off.")}
        return {"message": ("Honestly? Sit this round out. I'd rather keep your "
                            "trust for the rounds that are actually worth it.")}

    # ---------------- Buyer ----------------
    v = state.get("v", 0.0)
    u = state.get("u", 0.0)
    sig_positive = _msg_is_positive(state.get("seller_message"))

    # Empirical reliability of the seller's positive signals.
    pos_seen = pos_high = neg_seen = neg_high = 0
    for h in history:
        q_known = h.get("quality")               # revealed only if we bought
        if q_known is None:
            continue
        if _msg_is_positive(h.get("seller_message")):
            pos_seen += 1
            pos_high += q_known == "high"
        else:
            neg_seen += 1
            neg_high += q_known == "high"

    # Beta-smoothed posterior P(high | signal), anchored on the prior p.
    if sig_positive:
        post = (pos_high + 4 * p) / (pos_seen + 4)
    else:
        post = (neg_high + 4 * (p * 0.5)) / (neg_seen + 4)

    ev = post * v + (1 - post) * u
    buy = ev > price
    # Value-of-information probe: quality is revealed only when we buy, so a
    # game with zero purchases never learns whether the seller's positive
    # signals are honest — the posterior stays pinned at the prior and 40%
    # of buyer games ended with 0 payoff that way.  One early probe on a
    # positive signal pays when the upside of trusting an honest seller over
    # the remaining rounds beats the probe's expected loss.
    if not buy and sig_positive and pos_seen == 0 and v > price:
        rounds_left = max(0, (T or 20) - t)
        info_gain = 0.5 * p * rounds_left * (v - price)
        if info_gain > price - ev:
            buy = True
    log.info("per-diag[%s] t=%s/%s p=%.2f v=%.1f u=%.1f price=%.1f sig=%s post=%.2f ev=%.1f buy=%s (pos %s/%s)",
             game["game_id"][:8], t, T, p, v, u, price, sig_positive, post, ev, buy,
             pos_high, pos_seen)
    return {"decision": "yes" if buy else "no"}


# ============================================================================
# Dispatcher + safety net
# ============================================================================

def _fallback(game: dict) -> dict:
    """A guaranteed-valid move for any phase — never burn the 5 attempts."""
    atype = game["valid_actions"]["type"]
    state = game["game_state"]
    if atype == "offer":
        if game["game_family"] == "bargaining":
            money = int(state["money_to_divide"])
            half = money // 2
            return {"alice_gain": half, "bob_gain": money - half}
        me = game["your_player"]
        return {"product_price": round(state[f"{me}_value"], 2)}
    if atype == "seller_message":
        return {"message": "Your call this round."}
    if atype in ("seller_recommendation", "buyer_decision"):
        return {"decision": "no"}
    if atype == "decision":
        if game["game_family"] == "negotiation":
            return {"decision": "AcceptOffer"}
        return {"decision": "accept"}
    return {"decision": "accept"}


STRATEGIES = {
    "bargaining": bargaining_strategy,
    "negotiation": negotiation_strategy,
    "persuasion": persuasion_strategy,
}


_TRAJ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "trajectories.jsonl")
_TRAJ_LOCK = threading.Lock()


def _log_trajectory(game: dict, move: dict) -> None:
    """Append (observed state, our move) per turn; the last record of a game
    carries its full history, so this doubles as trajectory capture for the
    workshop-paper behavior analysis.  Must never break the agent."""
    try:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "game_id": game.get("game_id"),
               "game_family": game.get("game_family"),
               "your_player": game.get("your_player"),
               "valid_actions": game.get("valid_actions"),
               "game_state": game.get("game_state"),
               "move": move}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _TRAJ_LOCK, open(_TRAJ_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:                                     # noqa: BLE001
        log.exception("trajectory logging failed (ignored)")


def my_strategy(game: dict) -> dict:
    try:
        move = STRATEGIES[game["game_family"]](game)
        # Trim messages defensively (2000-char hard cap server-side).
        if isinstance(move.get("message"), str):
            move["message"] = move["message"][:1900]
        _log_trajectory(game, move)
        return move
    except Exception:                                     # noqa: BLE001
        log.exception("strategy error in %s — using fallback", game.get("game_id"))
        try:
            move = _fallback(game)
            _log_trajectory(game, move)
            return move
        except Exception:                                 # noqa: BLE001
            return {"decision": "accept"}


if __name__ == "__main__":
    from glee_sdk import GleeClient

    api_key = os.environ.get("GLEE_API_KEY")
    if not api_key:
        raise SystemExit("Set GLEE_API_KEY first (from your dashboard).")

    client = GleeClient(api_key=api_key)
    log.info("stats: %s", client.stats())
    # Rule-based = instant moves; concurrency keeps volume high (ratings need
    # games: display rating is shrunk by g/(g+30), top-100 needs 10 games/day).
    #
    # Rotation: max_time makes run() stop taking new games after N seconds,
    # drain the in-flight ones, and return with zero active games.  The outer
    # supervisor loop then relaunches this script — so code edits deploy
    # within one rotation with no timed-out games and no queue ban.
    rotate = int(os.environ.get("GLEE_ROTATE_SECS", "7200"))
    client.run(my_strategy, concurrency=8, max_time=rotate)
    log.info("rotation complete — exiting for clean relaunch")