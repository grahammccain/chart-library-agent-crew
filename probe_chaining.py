"""
Phase 0.5 chaining probe.

Does the orchestrator take the SECOND hop -- chartlibrary_cohort_analyze ->
chartlibrary_cohort_introspect -- and does embedding a `suggested_introspections`
pointer in cohort_analyze's RESPONSE drive that second hop?

This attacks the introspect=0 milestone: in production the introspection tool
has ZERO external calls. Two hypotheses:

  * plain : with a good introspect tool DESCRIPTION alone, the agent rarely
            chains -- cohort_analyze already looks complete, so there is no felt
            need for a second hop. (Expectation: chain-rate ~ 0, matching prod.)
  * nudge : when cohort_analyze's response carries a concrete
            suggested_introspections pointer, the agent takes the second hop.

The delta (nudge - plain) = the measured value of building suggested_introspections
(the #72 Phase-0 move), on real tool-use behaviour rather than a guess.

Run:
  python probe_chaining.py --mode mock                          # free, SYNTHETIC
  python probe_chaining.py --mode live --yes --env-file PATH    # paid, real

Live needs an Anthropic key (via --env-file dotenv or ANTHROPIC_API_KEY in the
environment) AND an explicit --yes. The key is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from harness import LiveModel, MockModel, run_episode
from prompts import PROMPTS
from tools import CHARTLIBRARY_INTROSPECT_NAME, CHARTLIBRARY_TOOL_NAMES, build_loadout, run_tool

DEFAULT_ORCH = "claude-sonnet-4-6"
PRIMARY = CHARTLIBRARY_TOOL_NAMES  # search + cohort_analyze


def positives():
    return [p for p in PROMPTS if p["expects_chartlibrary"]]


def called_primary(called) -> bool:
    return any(c in PRIMARY for c in called)


def chained(called) -> bool:
    """True if introspect was called AFTER some primary chartlibrary tool."""
    if CHARTLIBRARY_INTROSPECT_NAME not in called:
        return False
    i = called.index(CHARTLIBRARY_INTROSPECT_NAME)
    return any(p in called[:i] for p in PRIMARY)


def run_arm(prompts, backend, nudge: bool):
    loadout = build_loadout("v2", 7, introspect=True)
    runner = (lambda n, a: run_tool(n, a, nudge=True)) if nudge else run_tool
    eps = []
    for p in prompts:
        e = run_episode(p, loadout, backend, tool_runner=runner)
        e["called_primary"] = called_primary(e["called"])
        e["chained"] = chained(e["called"])
        eps.append(e)
    return eps


def report(eps) -> dict:
    prim = [e for e in eps if e["called_primary"]]
    ch = [e for e in prim if e["chained"]]
    denom = len(prim)
    return {
        "n": len(eps),
        "n_called_primary": denom,
        "n_chained": len(ch),
        "chain_rate": round(len(ch) / denom, 3) if denom else None,
        "chained_ids": [e["id"] for e in ch],
    }


def load_key(env_file):
    """Read ANTHROPIC_API_KEY from a dotenv-style file. Value is never printed."""
    if not env_file:
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    os.environ["ANTHROPIC_API_KEY"] = v
                return


def main():
    ap = argparse.ArgumentParser(description="Phase 0.5 chaining probe")
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--orchestrator-model", default=DEFAULT_ORCH)
    ap.add_argument("--yes", action="store_true", help="authorize live spend")
    ap.add_argument("--env-file", default=None,
                    help="dotenv file to read ANTHROPIC_API_KEY from (never printed)")
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (0=all)")
    ap.add_argument("--out", default="results_chaining.json")
    args = ap.parse_args()

    load_key(args.env_file)
    prompts = positives()
    if args.limit:
        prompts = prompts[: args.limit]

    if args.mode == "live":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Live mode needs ANTHROPIC_API_KEY (use --env-file or set it). Aborting.")
            sys.exit(2)
        if not args.yes:
            est = 2 * len(prompts) * 3  # 2 arms x prompts x ~3 model calls/episode
            print(f"Live mode will make ~{est} paid Anthropic calls "
                  f"(orchestrator={args.orchestrator_model}).")
            print("Re-run with --yes to authorize the spend.")
            sys.exit(3)
        backend = LiveModel(args.orchestrator_model)
    else:
        backend = MockModel("v2")

    synthetic = args.mode == "mock"
    banner = "SYNTHETIC - plumbing only, NOT a result" if synthetic else "LIVE - real chaining"
    print("=== Phase 0.5 chaining probe ===")
    print(f"mode: {args.mode}  ({banner})")
    print(f"should-use prompts: {len(prompts)}   orchestrator: {args.orchestrator_model}")

    t0 = time.time()
    plain = run_arm(prompts, backend, nudge=False)
    nudge = run_arm(prompts, backend, nudge=True)
    rp, rn = report(plain), report(nudge)

    print("\n-- chain rate: (primary chartlibrary tool) -> cohort_introspect --")
    for label, r in (("plain (no response hint)", rp),
                     ("nudge (suggested_introspections)", rn)):
        print(f"  {label:34s}: chain_rate={_fmt(r['chain_rate'])} "
              f"({r['n_chained']}/{r['n_called_primary']} of chartlibrary uses)")
    if rp["chain_rate"] is not None and rn["chain_rate"] is not None:
        print(f"  {'delta (nudge - plain)':34s}: "
              f"{rn['chain_rate'] - rp['chain_rate']:+.3f}")

    out = {"meta": {"mode": args.mode, "synthetic": synthetic,
                    "orchestrator_model": args.orchestrator_model,
                    "n_prompts": len(prompts),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "plain": {"report": rp, "episodes": plain},
           "nudge": {"report": rn, "episodes": nudge}}
    out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ({out['meta']['elapsed_sec']}s)")

    if synthetic:
        print("NOTE: mock chain rates are synthetic plumbing only. The real "
              "introspect=0 answer needs --mode live (paid, --yes).")


def _fmt(rate):
    return " n/a" if rate is None else f"{rate:.3f}"


if __name__ == "__main__":
    main()
