"""Measure how far the exact dynamic program reaches before it stops being cheap.

The old bin grid fixed the state count at 4032 regardless of horizon, by
aliasing everything that did not fit. Exact enumeration does the opposite: the
state count is whatever the dynamics actually produce, so the horizon we can
ground-truth is an empirical question. This answers it.

Usage:
    PYTHONPATH=. python rllib/DP/probe_horizon.py [--config PATH] [--max-t N]
                                                  [--cap N] [--solve]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rllib.DP.exact_dp import ExactDP, build

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def probe(config_path: Path, max_t: int, cap: int, solve: bool, min_t: int = 1) -> None:
    print(f"config     {config_path}")
    ex0 = build(config_path, horizon=1)
    print(f"p(research success)  {ex0.p_research:.6f}")
    print(f"p(permit | move)     {ex0.p_permit:.6g}"
          f"   (= max_greenbudget {ex0.dp.max_greenbudget:g} / world {ex0.dp.worldsize})")
    print(f"steps per year       {ex0.dp.yearsteps}")
    print(f"gamma                {ex0.gamma}")
    print()

    header = f"{'T':>4} {'years':>6} {'|layer T|':>12} {'|reachable|':>13} {'enum s':>9}"
    if solve:
        header += f" {'solve s':>9} {'V*(s0)':>12}"
    print(header)
    print("-" * len(header))

    for t in range(min_t, max_t + 1):
        ex = build(config_path, horizon=t)
        t0 = time.perf_counter()
        try:
            layers = ex.enumerate_reachable(cap=cap)
        except RuntimeError as exc:
            print(f"{t:>4} {'':>6} {'--':>12} {'--':>13} {'':>9}   stopped: {exc}")
            break
        enum_s = time.perf_counter() - t0

        years = t / ex.dp.yearsteps
        row = (f"{t:>4} {years:>6.1f} {len(layers[-1]):>12,} "
               f"{ex.total_states():>13,} {enum_s:>9.2f}")

        if solve:
            t1 = time.perf_counter()
            values, _ = ex.solve()
            solve_s = time.perf_counter() - t1
            v0 = values[0][next(iter(layers[0].states))]
            row += f" {solve_s:>9.2f} {v0:>12.4f}"

        print(row)

        if enum_s > 60.0:
            print("\nstopping: enumeration passed 60 s, which is the point where "
                  "iterating on the model gets painful")
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--min-t", type=int, default=1)
    ap.add_argument("--max-t", type=int, default=14)
    ap.add_argument("--cap", type=int, default=4_000_000,
                    help="abort a layer above this many states")
    ap.add_argument("--solve", action="store_true",
                    help="also run backward induction and report V*(s0)")
    args = ap.parse_args()
    probe(args.config, args.max_t, args.cap, args.solve, args.min_t)


if __name__ == "__main__":
    main()
