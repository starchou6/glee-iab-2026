"""Extract Agent A's authoritative game results from glee_agent.log.

The SDK logs every finished game as:
    ... INFO Game <uuid> finished! Result: {'player_1_payoff': ..., ...}
This gives full-coverage outcomes for A without API calls (the API backfill
adds opponent metadata separately). Output schema matches
backfill_results.py, with opponent: null.
"""
from __future__ import annotations

import ast
import json
import os
import re

_DIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(_DIR, "glee_agent.log")
OUT = os.path.join(_DIR, "results_a_log.jsonl")

PAT = re.compile(r"Game ([0-9a-f-]{36}) finished! Result: (\{.*\})")

results: dict[str, dict] = {}
with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = PAT.search(line)
        if not m:
            continue
        try:
            res = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            continue
        results[m.group(1)] = res

with open(OUT, "w", encoding="utf-8") as f:
    for gid, res in results.items():
        f.write(json.dumps({"game_id": gid, "game_family": None,
                            "your_player": None, "status": "completed",
                            "opponent": None, "result": res},
                           ensure_ascii=False) + "\n")
print(f"wrote {len(results)} results to {OUT}")
