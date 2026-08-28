#!/usr/bin/env python3
"""Compare agent 0 between the single-agent and multi-agent runs.

Agent "0" exists in both arms: it is the only firm when n_agents == 1 and
one of five when n_agents == 5. Holding the index fixed and changing only
the market structure is the cleanest read on what the auction, the shared
allocation and the equality term in the planner reward do to a firm.

Reads the dense logs each run writes (dense_logs/dense_logs_<N>/
logs_<step>/env<NNN>.lz4, one lz4-compressed JSON per vector env), takes
the most recent snapshot by default, and averages across the env files in
it so a single unlucky episode does not carry the comparison.

  PYTHONPATH=. python3 rllib/compare_agent0.py \
      --single rllib/exp/verify_single_s1 \
      --multi  rllib/exp/verify_multi_s1
"""
from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob

import lz4.frame
import numpy as np

# Per-step fields the env zeroes at the top of each component_step, so the
# episode total is their sum. Everything else below is already cumulative
# and is read from the final step instead.
PER_STEP = ("Build", "Move", "Cum_Punishment", "Buy_count", "Sell_count",
            "Carbon_project_it")


def latest_snapshot(run_dir: str) -> list[str]:
    """Newest dense_logs_<N>/logs_<step>/ in run_dir, as a list of env files."""
    roots = sorted(glob(os.path.join(run_dir, "dense_logs", "dense_logs_*")),
                   key=lambda p: int(re.search(r"_(\d+)$", p).group(1)))
    if not roots:
        roots = [os.path.join(run_dir, "dense_logs")]
    snaps = sorted(glob(os.path.join(roots[-1], "logs_*")))
    if not snaps:
        raise SystemExit(f"no dense logs under {run_dir}; was dense_log_frequency set?")
    files = sorted(glob(os.path.join(snaps[-1], "env*.lz4")))
    if not files:
        raise SystemExit(f"no env*.lz4 in {snaps[-1]}")
    return files


def load(path: str) -> dict:
    with lz4.frame.open(path, mode="rb") as fh:
        return json.loads(fh.read().decode())


def episode_summary(log: dict, aidx: str = "0") -> dict:
    states = log["states"]
    if aidx not in states[0]:
        raise SystemExit(f"agent {aidx!r} not in this log (agents: {sorted(states[0])})")

    last = states[-1][aidx]
    inv, esc, end = last["inventory"], last["escrow"], last["endogenous"]

    out = {
        # total_endowment is inventory + escrow, matching what the reward sees
        "coin_final": inv["Coin"] + esc["Coin"],
        "carbon_idx_final": inv["Carbon_idx"] + esc["Carbon_idx"],
        "labor_total": end["Labor"],
        "emission_total": end["Carbon_emission"],
        "costs_total": end["Costs"],
        "revenue_total": end["Revenue"],
        # Research has four distinguishable stages and they tell different
        # stories: an attempt is the action being chosen at all; a success is
        # the attempt passing the random_fails roll; matured is the success
        # surviving `delay` steps to raise Research_count; labor is what was
        # paid for the attempts regardless of outcome. Reporting only the
        # matured count cannot distinguish "never tried" from "tried and the
        # pipeline dropped it".
        "research_attempts": 0.0,
        "research_success": 0.0,
        "research_matured": last["Research_count"][0],
        "research_labor": last["ResearchCount"],
        # Green projects this agent built: each Gather collection turns a
        # Carbon_project tile into a Green_project landmark and adds the tile
        # to the agent's inventory.
        "green_built": inv["Carbon_project"],
        "emission_rate_final": last["Carbon_emission_rate"],
    }
    for k in PER_STEP:
        out[k.lower() + "_total"] = float(sum(s[aidx].get(k, 0) or 0 for s in states))

    for step in log.get("Carbon_component-research", []):
        for ev in step:
            if str(ev.get("enterprise")) == str(aidx):
                out["research_attempts"] += 1.0
                out["research_success"] += float(ev.get("action_result") == "Success")

    # Builds are the alternative use of the same action slot, so the ratio of
    # the two is the readable quantity.
    out["build_events"] = float(sum(
        1 for step in log.get("Carbon_component-builds", []) for ev in step
        if str(ev.get("enterprise")) == str(aidx)))

    # Episode return: rewards are deltas now, so the sum telescopes to
    # U(final) - U(initial).
    out["return"] = float(sum(r.get(aidx, 0.0) for r in log["rewards"]))
    out["planner_return"] = float(sum(r.get("p", 0.0) for r in log["rewards"]))
    out["n_agents"] = len([k for k in states[0] if k != "p"])

    # World-level context for the green channel: how many collectible tiles
    # were left on the map when the episode ended. A high number next to a
    # near-zero green_built means the tiles were there and went unused,
    # which is a different finding from none ever spawning.
    maps = [w for w in log["world"] if isinstance(w, dict) and w]
    if maps:
        cp = np.array(maps[-1]["Carbon_project"])
        out["green_tiles_left"] = float((cp > 0).sum())
    else:
        out["green_tiles_left"] = float("nan")
    return out


def summarise(run_dir: str, aidx: str) -> tuple[dict, dict, int]:
    files = latest_snapshot(run_dir)
    rows = [episode_summary(load(f), aidx) for f in files]
    keys = rows[0].keys()
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return mean, std, len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="run dir with n_agents == 1")
    ap.add_argument("--multi", required=True, help="run dir with n_agents  > 1")
    ap.add_argument("--agent", default="0", help="agent index to compare (default 0)")
    args = ap.parse_args()

    s_mean, s_std, s_n = summarise(args.single, args.agent)
    m_mean, m_std, m_n = summarise(args.multi, args.agent)

    print(f"agent {args.agent}: single ({s_mean['n_agents']:.0f} firm, {s_n} envs) "
          f"vs multi ({m_mean['n_agents']:.0f} firms, {m_n} envs)")
    print(f"{'metric':<22} {'single':>14} {'multi':>14} {'delta':>12}")
    print("-" * 66)
    for k in s_mean:
        if k == "n_agents":
            continue
        a, b = s_mean[k], m_mean[k]
        d = b - a
        print(f"{k:<22} {a:>9.3f}±{s_std[k]:<4.2g} {b:>9.3f}±{m_std[k]:<4.2g} {d:>+12.3f}")

    print()
    print("delta is multi - single. Read with care:")
    print("  * the planner objective differs by arm -- profit when n==1,")
    print("    equality * productivity/n when n>1 -- so planner_return is NOT")
    print("    comparable across the two columns.")
    print("  * buy/sell are structurally 0 in single: the auction is absent")
    print("    from that config, since one agent cannot match its own ask.")


if __name__ == "__main__":
    main()
