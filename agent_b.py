"""GLEE Competition agent — "Rubinstein-RT" (Agent B)

Runtime-AI counterpart to Agent A ("Rubinstein", glee_agent.py) for the
IAB workshop paper comparison:

  * Agent A: pure rules, zero LLM calls — AI intervened at DESIGN TIME
    (offline analysis of match data rewrote the rules).
  * Agent B (this file): every decision is made by an LLM at RUNTIME; each
    call injects a "lessons" block distilled offline from B's own match
    history (distill_lessons.py -> lessons.txt, hot-reloaded by mtime).

Decision path per move:
  1. Build prompt = per-family rules digest + <lessons> + compact state.
  2. Call the local Claude Code CLI in headless mode (`claude -p`) — uses
     the local subscription, not API billing. Hard subprocess timeout.
  3. Strictly validate the returned move (semantic checks, not just JSON).
  4. Any failure (timeout / CLI error / invalid move) -> immediately fall
     back to Agent A's rule strategy imported from glee_agent (so B's
     fallback behaviour is byte-identical to A — the only experimental
     difference is the LLM decision layer).
  5. Log to trajectories_b.jsonl in A's exact JSONL format plus an `llm`
     object (latency, tokens, fallback flag, lessons version) — the
     cost/latency data for the paper.

Usage:
    set GLEE_API_KEY_B=glee_...        # separate key from Agent A's
    python -u agent_b.py
Optional env:
    AGENT_B_MODEL        (default "claude-sonnet-5")
    AGENT_B_TIMEOUT_S    (default 90)
    AGENT_B_CONCURRENCY  (default 4)
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time

import glee_agent  # Agent A's rules — imported, never modified

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rubinstein-rt")

_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_B_PATH = os.path.join(_DIR, "trajectories_b.jsonl")
LESSONS_PATH = os.path.join(_DIR, "lessons.txt")

# Sonnet 5 by default, for subscription-usage economy. Measured on this
# agent's own logs: Fable 5 burned the usage window (34% of moves fell back
# on `claude CLI exit 1`, median 14s); Opus 5 was clean (0.3% fallback,
# median 7.7s) but still too costly per the usage budget.
# Server allows 120s/turn counted from turn start (SDK polling + submission
# latency come out of the same budget) — 90s leaves a 30s safety margin, and
# 3 consecutive own-turn timeouts trigger a 30-min queue cooldown.
MODEL = os.environ.get("AGENT_B_MODEL", "claude-opus-5")
TIMEOUT_S = float(90)
CONCURRENCY = int(2)
MSG_LIMIT = 1900

_TRAJ_LOCK = threading.Lock()

_CLAUDE_BIN = shutil.which("claude")
if _CLAUDE_BIN is None:
    raise SystemExit("claude CLI not found on PATH — install/log in to Claude Code first.")


# ----------------------------------------------------------------------------
# Lessons hot-reload
# ----------------------------------------------------------------------------

_lessons_lock = threading.Lock()
_lessons_cache = {"mtime": None, "text": "", "version": "none"}


def get_lessons() -> tuple[str, str]:
    """Return (lessons_text, version). Reloads lessons.txt when mtime changes."""
    with _lessons_lock:
        try:
            mtime = os.path.getmtime(LESSONS_PATH)
        except OSError:
            return "", "none"
        if mtime != _lessons_cache["mtime"]:
            try:
                with open(LESSONS_PATH, encoding="utf-8") as f:
                    _lessons_cache["text"] = f.read().strip()
                _lessons_cache["mtime"] = mtime
                _lessons_cache["version"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
                log.info("lessons reloaded (version %s, %d chars)",
                         _lessons_cache["version"], len(_lessons_cache["text"]))
            except OSError:
                pass
        return _lessons_cache["text"], _lessons_cache["version"]


# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------

_RULES = {
    "bargaining": (
        "GAME: Alternating-offers bargaining. Two players split a fixed pot of money "
        "over rounds. player_1 proposes on odd rounds, player_2 on even rounds. Each "
        "player has a per-round discount factor delta (my future gains shrink by my "
        "delta each round). If the horizon is known there is a last round; if nobody "
        "ever agrees, BOTH get $0. 'walkaway' ends the game with $0 for both and is "
        "almost always dominated by 'reject'. Timeouts and no-deals are catastrophic "
        "for ranking — a decent deal beats a theoretical optimum that never closes."
    ),
    "negotiation": (
        "GAME: Price negotiation between a seller and a buyer over one product. Each "
        "side has a private (or sometimes visible) valuation. Seller profits by "
        "selling above own value; buyer profits by buying below own value. NEVER "
        "trade at a loss relative to your own valuation. Payoffs are not discounted "
        "over rounds, so rejecting keeps option value; WalkAway locks in 0 and is "
        "rarely correct except when no mutually profitable price can exist. If the "
        "final round arrives, accept any profitable offer."
    ),
    "persuasion": (
        "GAME: Repeated persuasion (seller/buyer, ~20 rounds). Each round a product "
        "is high quality with prior probability p. SELLER sees the quality and sends "
        "a message/recommendation; earns the price whenever the buyer buys. BUYER "
        "sees only the message; buying a high-quality item pays v-price, a "
        "low-quality one pays u-price (usually a loss). Quality is revealed to the "
        "buyer only after buying. Seller's long-run income depends on the buyer "
        "trusting positive signals, so lying too often destroys future sales; buyer "
        "should track the seller's empirical honesty and buy when posterior expected "
        "value exceeds the price."
    ),
}

# Output contract per valid_actions["type"]; {msg} is replaced with the
# message-field line only when messages are allowed.
_CONTRACTS = {
    "offer_bargaining": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"alice_gain": <int>, "bob_gain": <int>{msg}}}\n'
        'Constraints: both integers >= 0 and alice_gain + bob_gain == {money} '
        '(alice = player_1, bob = player_2; you are {me}).'
    ),
    "decision_bargaining": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"decision": "accept" | "reject" | "walkaway"}}'
    ),
    "offer_negotiation": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"product_price": <number>{msg}}}\n'
        'Constraint: product_price > 0.'
    ),
    "decision_negotiation": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"decision": "AcceptOffer" | "RejectOffer" | "WalkAway"'
        ', "product_price": <number, REQUIRED counter-offer if decision is RejectOffer>{msg}}}'
    ),
    "seller_message": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"message": "<your message to the buyer, max 1900 chars>"}}'
    ),
    "seller_recommendation": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"decision": "yes" | "no"}}   (yes = recommend buying)'
    ),
    "buyer_decision": (
        'Respond with ONLY a JSON object, no prose, no code fences:\n'
        '{{"decision": "yes" | "no"}}   (yes = buy)'
    ),
}

_HISTORY_TAIL = 6          # verbatim recent history entries
_HISTORY_FIELD_MAX = 400   # clamp long message strings inside history


def _compact_history(history: list) -> dict:
    """Recent entries verbatim + whole-game aggregates, so prompts stay small."""
    def clamp(entry):
        out = {}
        for k, v in entry.items():
            if isinstance(v, str) and len(v) > _HISTORY_FIELD_MAX:
                v = v[:_HISTORY_FIELD_MAX] + "…"
            if k == "response_time_ms":
                continue
            out[k] = v
        return out

    tail = [clamp(h) for h in history[-_HISTORY_TAIL:]]
    summary = {"total_entries": len(history)}
    # Persuasion aggregates: seller honesty as observed by the buyer.
    bought = [h for h in history if h.get("bought")]
    if bought:
        highs = sum(1 for h in bought if h.get("quality") == "high")
        summary["purchases"] = len(bought)
        summary["high_quality_when_bought"] = highs
    return {"recent": tail, "summary": summary}


def build_prompt(game: dict, lessons: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt).

    The system half carries the stable framing (role, family rules, lessons,
    output contract); it replaces the CLI's default ~34k-token Claude Code
    system prompt entirely. The user half carries only the live state.
    """
    family = game["game_family"]
    state = dict(game["game_state"])
    atype = game["valid_actions"]["type"]
    msgs_ok = state.get("messages_allowed", True)
    me = game["your_player"]

    history = state.pop("history", []) or []
    state.pop("game_family", None)
    compact = _compact_history(history)

    contract_key = atype
    if atype == "offer":
        contract_key = f"offer_{family}"
    elif atype == "decision":
        contract_key = f"decision_{family}"
    contract = _CONTRACTS.get(contract_key, _CONTRACTS.get(atype, ""))
    msg_line = (', "message": "<optional strategic message to opponent, max 1900 chars>"'
                if msgs_ok and "{msg}" in contract else "")
    contract = contract.replace("{msg}", msg_line)
    contract = contract.replace("{money}", str(state.get("money_to_divide", "")))
    contract = contract.replace("{me}", me)

    lessons_block = lessons if lessons else "(no lessons distilled yet — cold start)"

    system = (
        "You are a competitive economic-game agent in the GLEE tournament. "
        "Maximize YOUR OWN total payoff over many games; ranking is relative to "
        "other agents, and no-deal / timeout outcomes are the worst results.\n\n"
        f"{_RULES[family]}\n\n"
        "<lessons>\n"
        "Conclusions distilled from your own past match logs — weigh them heavily:\n"
        f"{lessons_block}\n"
        "</lessons>\n\n"
        "You have no tools and no filesystem: decide purely from the state given "
        "to you and answer directly in one turn.\n\n"
        f"{contract}"
    )
    user = (
        f"You are {me}. Current state (history trimmed to recent entries + summary):\n"
        f"{json.dumps(state, ensure_ascii=False, default=str)}\n"
        f"History: {json.dumps(compact, ensure_ascii=False, default=str)}\n\n"
        "Decide your move now. Think briefly and concretely about the numbers "
        "(equilibrium logic, opponent's revealed behaviour, rounds remaining), "
        "then output the JSON object and nothing else."
    )
    return system, user


# ----------------------------------------------------------------------------
# LLM call via Claude Code CLI (headless, subscription-billed)
# ----------------------------------------------------------------------------

def call_llm(system: str, user: str) -> tuple[str, dict]:
    """Run `claude -p` once. Returns (result_text, meta). Raises on any failure.

    `--tools ""` disables every built-in tool: with tools available the model
    tries to read files / run bash, which burns the single allowed turn and
    fails the call (stop_reason "tool_use"). `--system-prompt` replaces the
    default Claude Code system prompt with our own, cutting ~34k input tokens
    per call and removing all tool-oriented instructions.
    """
    t0 = time.monotonic()
    proc = subprocess.run(
        [_CLAUDE_BIN, "-p", "--output-format", "json",
         "--model", MODEL, "--max-turns", "1", "--strict-mcp-config",
         "--tools", "", "--system-prompt", system],
        input=user.encode("utf-8"),
        capture_output=True,
        timeout=TIMEOUT_S,
    )
    wall_ms = int((time.monotonic() - t0) * 1000)
    if proc.returncode != 0:
        # The CLI reports usage-limit and auth failures on stdout, not stderr,
        # so capture both — an empty stderr alone diagnoses nothing.
        err = proc.stderr.decode("utf-8", "replace").strip()
        out = proc.stdout.decode("utf-8", "replace").strip()
        raise RuntimeError(f"claude CLI exit {proc.returncode}: "
                           f"stderr={err[:200]!r} stdout={out[:200]!r}")

    envelope = None
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "result" in obj:
            envelope = obj
            break
    if envelope is None:
        raise RuntimeError("no result envelope in claude CLI output")
    if envelope.get("is_error"):
        raise RuntimeError(f"claude CLI error result: {str(envelope)[:300]}")

    usage = envelope.get("usage") or {}
    meta = {
        "model": MODEL,
        "latency_ms": wall_ms,
        "api_ms": envelope.get("duration_api_ms"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cost_usd_equiv": envelope.get("total_cost_usd"),
    }
    return envelope.get("result") or "", meta


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of the model's text (fences tolerated)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in LLM output")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON is not an object")
    return obj


# ----------------------------------------------------------------------------
# Strict semantic validation — nothing invalid ever reaches the server
# ----------------------------------------------------------------------------

def _clean_message(move: dict, raw: dict, msgs_ok: bool) -> None:
    msg = raw.get("message")
    if msgs_ok and isinstance(msg, str) and msg.strip():
        move["message"] = msg.strip()[:MSG_LIMIT]


def validate_move(game: dict, raw: dict) -> dict:
    """Return a clean, guaranteed-valid move dict, or raise ValueError."""
    family = game["game_family"]
    state = game["game_state"]
    atype = game["valid_actions"]["type"]
    msgs_ok = state.get("messages_allowed", True)

    if atype == "offer" and family == "bargaining":
        money = int(state["money_to_divide"])
        a, b = raw.get("alice_gain"), raw.get("bob_gain")
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            raise ValueError("gains missing/non-numeric")
        a, b = int(round(a)), int(round(b))
        if a < 0 or b < 0 or a + b != money:
            raise ValueError(f"gains invalid: {a}+{b} != {money}")
        move = {"alice_gain": a, "bob_gain": b}
        _clean_message(move, raw, msgs_ok)
        return move

    if atype == "decision" and family == "bargaining":
        d = str(raw.get("decision", "")).strip().lower()
        if d not in ("accept", "reject", "walkaway"):
            raise ValueError(f"bad decision {d!r}")
        return {"decision": d}

    if atype == "offer" and family == "negotiation":
        p = raw.get("product_price")
        if not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            raise ValueError(f"bad price {p!r}")
        move = {"product_price": round(float(p), 2)}
        _clean_message(move, raw, msgs_ok)
        return move

    if atype == "decision" and family == "negotiation":
        d = str(raw.get("decision", "")).strip()
        canon = {"acceptoffer": "AcceptOffer", "rejectoffer": "RejectOffer",
                 "walkaway": "WalkAway"}.get(d.lower())
        if canon is None:
            raise ValueError(f"bad decision {d!r}")
        move = {"decision": canon}
        if canon == "RejectOffer":
            p = raw.get("product_price")
            if isinstance(p, (int, float)) and math.isfinite(p) and p > 0:
                move["product_price"] = round(float(p), 2)
        _clean_message(move, raw, msgs_ok)
        return move

    if atype == "seller_message":
        msg = raw.get("message")
        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("empty seller message")
        return {"message": msg.strip()[:MSG_LIMIT]}

    if atype in ("seller_recommendation", "buyer_decision"):
        d = str(raw.get("decision", "")).strip().lower()
        if d not in ("yes", "no"):
            raise ValueError(f"bad yes/no decision {d!r}")
        return {"decision": d}

    raise ValueError(f"unknown action type {atype!r}")


# ----------------------------------------------------------------------------
# Trajectory logging (A's format + llm metadata)
# ----------------------------------------------------------------------------

def log_trajectory(game: dict, move: dict, llm_meta: dict) -> None:
    try:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "game_id": game.get("game_id"),
               "game_family": game.get("game_family"),
               "your_player": game.get("your_player"),
               "valid_actions": game.get("valid_actions"),
               "game_state": game.get("game_state"),
               "move": move,
               "llm": llm_meta}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _TRAJ_LOCK, open(TRAJ_B_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:                                     # noqa: BLE001
        log.exception("trajectory logging failed (ignored)")


# ----------------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------------

def _rule_fallback(game: dict) -> dict:
    """Agent A's strategy, then A's safe fallback — identical behaviour to A."""
    try:
        move = glee_agent.STRATEGIES[game["game_family"]](game)
    except Exception:                                     # noqa: BLE001
        move = glee_agent._fallback(game)
    if isinstance(move.get("message"), str):
        move["message"] = move["message"][:MSG_LIMIT]
    return move


def strategy_b(game: dict) -> dict:
    lessons, lessons_version = get_lessons()
    meta = {"model": MODEL, "fallback": False, "fallback_reason": None,
            "lessons_version": lessons_version}
    move = None
    try:
        system, user = build_prompt(game, lessons)
        text, call_meta = call_llm(system, user)
        meta.update(call_meta)
        move = validate_move(game, extract_json(text))
    except subprocess.TimeoutExpired:
        meta.update(fallback=True, fallback_reason="timeout",
                    latency_ms=int(TIMEOUT_S * 1000))
    except ValueError as e:
        meta.update(fallback=True, fallback_reason=f"invalid_output: {e}")
    except Exception as e:                                # noqa: BLE001
        meta.update(fallback=True, fallback_reason=f"llm_error: {type(e).__name__}: {e}"[:300])

    if move is None:
        move = _rule_fallback(game)
        log.warning("fallback[%s] %s — %s", game.get("game_id", "")[:8],
                    game["valid_actions"]["type"], meta["fallback_reason"])
    else:
        log.info("llm-move[%s] %s %s (%.1fs)", game.get("game_id", "")[:8],
                 game["game_family"], game["valid_actions"]["type"],
                 (meta.get("latency_ms") or 0) / 1000)

    log_trajectory(game, move, meta)
    return move


def safe_strategy(game: dict) -> dict:
    """Outermost guard: never burn the 5-invalid-move budget."""
    try:
        return strategy_b(game)
    except Exception:                                     # noqa: BLE001
        log.exception("strategy_b crashed in %s — hard fallback", game.get("game_id"))
        try:
            move = _rule_fallback(game)
            log_trajectory(game, move, {"model": MODEL, "fallback": True,
                                        "fallback_reason": "strategy_crash",
                                        "lessons_version": "unknown"})
            return move
        except Exception:                                 # noqa: BLE001
            return {"decision": "accept"}


if __name__ == "__main__":
    from glee_sdk import GleeClient

    api_key = os.environ.get("GLEE_API_KEY_B")
    if not api_key:
        raise SystemExit("Set GLEE_API_KEY_B first (create a second agent in the "
                         "GLEE dashboard; do not reuse Agent A's key).")

    client = GleeClient(api_key=api_key)
    log.info("stats: %s", client.stats())
    log.info("model=%s timeout=%.0fs concurrency=%d lessons=%s",
             MODEL, TIMEOUT_S, CONCURRENCY, LESSONS_PATH)
    # LLM moves take seconds; modest concurrency keeps volume without piling
    # up CLI processes (each spawn is a full Node process).
    client.run(safe_strategy, concurrency=CONCURRENCY)
