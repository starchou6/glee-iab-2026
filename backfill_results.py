"""Backfill authoritative game results from the GLEE API.

For every unique game_id in a trajectories JSONL file, fetch the completed
game via GleeClient.game_state() and append one compact line to the output
JSONL: {game_id, game_family, your_player, status, opponent, result}.

The `result` object carries final payoffs/outcome; `opponent` carries
{type: "agent"|"human", name} — the key that lets the paper segment
behavior by opponent type. Resumable: already-fetched ids are skipped.

Usage:
    python backfill_results.py trajectories_b.jsonl results_b.jsonl GLEE_API_KEY_B
    python backfill_results.py trajectories.jsonl  results_a.jsonl GLEE_API_KEY
"""
from __future__ import annotations

import json
import os
import sys
import time

from glee_sdk import GleeClient



def unique_ids(traj_path: str) -> list[str]:
    ids: dict[str, None] = {}
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            i = line.find('"game_id": "')
            if i < 0:
                continue
            start = i + len('"game_id": "')
            ids[line[start:start + 36]] = None
    return list(ids)


def main() -> None:
    traj_path, out_path, key_name = sys.argv[1], sys.argv[2], sys.argv[3]
    api_key = os.environ[key_name]
    client = GleeClient(api_key=api_key)

    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["game_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    ids = [g for g in unique_ids(traj_path) if g not in done]
    print(f"{len(done)} already fetched, {len(ids)} to go", flush=True)

    errors = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for n, gid in enumerate(ids, 1):
            try:
                gs = client.game_state(gid)
                rec = {"game_id": gid,
                       "game_family": gs.get("game_family"),
                       "your_player": gs.get("your_player"),
                       "status": gs.get("status"),
                       "opponent": gs.get("opponent"),
                       "result": gs.get("result")}
                out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                errors = 0
            except Exception as e:                        # noqa: BLE001
                errors += 1
                print(f"ERR {gid}: {type(e).__name__}: {e}", flush=True)
                if errors >= 10:
                    print("10 consecutive errors — aborting (rate limit?)", flush=True)
                    break
                time.sleep(min(60, 2 ** errors))
            if n % 200 == 0:
                out.flush()
                print(f"{n}/{len(ids)}", flush=True)
            time.sleep(0.05)                              # gentle on the API
    print("done", flush=True)


if __name__ == "__main__":
    main()
