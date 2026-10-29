"""Fetch current GLEE ratings and append to ratings_history.csv.

Used by the hourly monitoring loop: prints current ratings plus the delta
against the previous check so a decline is visible immediately.
"""
import csv
import datetime
import os

from glee_sdk import GleeClient

API_KEY = os.environ.get("GLEE_API_KEY")
HIST = os.path.join(os.path.dirname(__file__), "ratings_history.csv")
FAMILIES = ("bargaining", "negotiation", "persuasion")

stats = GleeClient(api_key=API_KEY).stats()
scores = stats["scores"]
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

prev = None
if os.path.exists(HIST):
    with open(HIST, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) > 1:
        prev = rows[-1]

new_row = [now] + [f"{scores[f]['rating']:.1f}" for f in FAMILIES] \
                + [str(scores[f]["games_played"]) for f in FAMILIES]
write_header = not os.path.exists(HIST)
with open(HIST, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["time"] + [f"{x}_rating" for x in FAMILIES]
                   + [f"{x}_games" for x in FAMILIES])
    w.writerow(new_row)

for i, fam in enumerate(FAMILIES):
    cur = scores[fam]["rating"]
    line = f"{fam}: {cur:.1f} ({scores[fam]['games_played']} games)"
    if prev:
        delta = cur - float(prev[1 + i])
        line += f"  delta since last check: {delta:+.1f}"
        if delta <= -30:
            line += "  <<< RATING DROP"
    print(line)
print("active_games:", stats.get("active_games"))
