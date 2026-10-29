# A Cost-and-Behavior Case Study of Two GLEE Agents

Code, prompts, configurations, analysis scripts, and aggregated data for the
NeurIPS 2026 Workshop on Interpreting Agent Behavior (IAB) competition paper
*A Cost-and-Behavior Case Study of Two GLEE Agents with Design-Time and Runtime
Behavioral Interpretation* (Jing Zhou, Microworld Co., Ltd.).

## Agents

| File | Role |
|---|---|
| `glee_agent.py` | **Agent A** ("Rubinstein"): zero-LLM rule-based agent (SPE solver, Bayesian δ inference, commitment-optimal disclosure). Rules revised offline at design time. |
| `tuning.json` | Agent A's tunable acceptance slopes. |
| `agent_b.py` | **Agent B** ("Rubinstein-RT"): every decision made by an LLM at runtime, briefed by a lessons block; falls back to Agent A's rule layer on timeout. |
| `distill_lessons.py` | Distills Agent B's own match history into `lessons.txt`. |
| `lessons.txt` | The last lessons block produced (written after the frozen analysis window; see the paper). |

Both agents need the GLEE SDK (`glee_sdk`) and a GLEE API key passed through
the environment (`GLEE_API_KEY` for A, `GLEE_API_KEY_B` for B). Agent B also
needs the Claude Code CLI.

## Analysis pipeline

```
python compare_metrics.py              # trajectories + results -> metrics_summary.json, curves_bargaining.csv, latency_b.csv
python make_numbers.py --offline       # metrics_summary.json -> paper/numbers.tex
python make_figures.py                 # -> paper/figs/*.pdf
```

`backfill_results.py` and `parse_a_results.py` build the per-game result files
from the GLEE API and Agent A's log; `check_ratings.py` reads current ratings.

## Data availability

This repository contains only de-identified aggregates
(`metrics_summary.json`, `curves_bargaining.csv`, `latency_b.csv`,
`ratings_history.csv`). The raw trajectories, logs, and per-game result files
are not released: the opponent pool includes human players, and the paper
commits to releasing aggregates only. `compare_metrics.py` therefore cannot be
rerun from this repository alone, but `make_numbers.py --offline` and
`make_figures.py` reproduce every number (as LaTeX macros in
`paper/numbers.tex`) and figure (`paper/figs/`) in the paper from the released
aggregates.
