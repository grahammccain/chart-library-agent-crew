"""
Phase 0 runner.

  python run.py --mode mock --desc v2 --loadout both --ab        # free, offline
  python run.py --mode live --desc v2 --ab --yes                 # paid, real selection

Mock mode proves the plumbing and prints SYNTHETIC numbers (clearly labelled).
Live mode makes real Anthropic tool-selection calls and is the only path whose
numbers answer GO/NO-GO. Live mode refuses to spend without an API key AND an
explicit --yes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from evaluate import (
    ab_report_multi,
    live_judge,
    mock_judge,
    selection_report,
    used_chartlibrary,
)
from harness import LiveModel, MockModel, run_episode
from prompts import PROMPTS
from real_tools import make_real_runner
from tools import CHARTLIBRARY_TOOL_NAMES, build_loadout

DEFAULT_ORCH = "claude-sonnet-4-6"
DEFAULT_JUDGE = "claude-haiku-4-5-20251001"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 chartlibrary selection harness")
    p.add_argument("--mode", choices=["mock", "live"], default="mock")
    p.add_argument("--desc", choices=["v1", "v2"], default="v2",
                   help="chartlibrary description arm")
    p.add_argument("--orchestrator-model", default=DEFAULT_ORCH)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--judge-model-2", default=None,
                   help="optional SECOND judge model for the answer-lift A/B (e.g. "
                        "claude-sonnet-4-6). Both judges score the SAME answer pair, so "
                        "we get a true inter-rater check — the receipt that the lift "
                        "isn't one judge's quirk. Persisted answers make a 3rd judge free.")
    p.add_argument("--loadout", choices=["7", "13", "both"], default="7",
                   help="tool-count condition")
    p.add_argument("--ab", action="store_true",
                   help="also run with/without answer-lift A/B")
    p.add_argument("--skip-selection", action="store_true",
                   help="run only the A/B (skip the selection pass); focuses paid "
                        "spend on the answer-lift gate — selection was validated in Phase 0")
    p.add_argument("--limit", type=int, default=0, help="cap #prompts (0=all)")
    p.add_argument("--yes", action="store_true", help="authorize live spend")
    p.add_argument("--key-file", default=None,
                   help="read ANTHROPIC_API_KEY from this local file (keeps the "
                        "secret out of shell history / the transcript)")
    p.add_argument("--out", default="results.json")
    p.add_argument("--real-chartlibrary", action="store_true",
                   help="in the A/B, call the LIVE chartlibrary endpoint for the two "
                        "chartlibrary tools (rivals stay fixtures). The honest-mirror "
                        "test — only meaningful with --ab.")
    return p.parse_args()


def loadout_sizes(arg):
    return [7, 13] if arg == "both" else [int(arg)]


def estimate_calls(n, sizes, ab, n_pos, skip_selection=False, n_judges=1):
    sel = 0 if skip_selection else 2 * n * len(sizes)  # ~2 model calls per episode
    # per positive: 2 episodes x ~2 orchestrator calls + n_judges judge calls
    ab_calls = ((4 + n_judges) * n_pos) if ab else 0
    return sel + ab_calls


def run_selection(prompts, backend, desc, size):
    loadout = build_loadout(desc, size)
    return [run_episode(p, loadout, backend) for p in prompts]


def run_ab(prompts, backend, desc, judges, tool_runner=None):
    """One orchestrator pass per prompt; EVERY judge scores the same with/without
    answer pair. `judges` is a list of (label, judge_fn) where
    judge_fn(rec, ans_with, ans_without, rng) -> {'winner': ..., ...}.

    Persists the answer texts + which tools the with-arm fired so a later judge
    (or a br07-style audit) needs no paid re-run."""
    with_loadout = build_loadout(desc, 7)
    without_loadout = [t for t in with_loadout
                       if t["name"] not in CHARTLIBRARY_TOOL_NAMES]
    records = []
    for p in prompts:
        ep_with = run_episode(p, with_loadout, backend, tool_runner=tool_runner)
        ep_without = run_episode(p, without_loadout, backend, tool_runner=tool_runner)
        rec = {"id": p["id"],
               "with_called": ep_with["called"],
               "with_fired_chartlibrary": used_chartlibrary(ep_with["called"]),
               "with_final": ep_with["final"],
               "without_final": ep_without["final"],
               "judges": {}}
        for label, judge_fn in judges:
            rng = random.Random(f"ab|{label}|{p['id']}")
            rec["judges"][label] = judge_fn(p, ep_with["final"],
                                            ep_without["final"], rng)
        records.append(rec)
    return records


def _fmt(rate):
    return "  n/a" if rate is None else f"{rate:>5.2f}"


def print_selection(size, rep):
    print(f"\n-- Selection (loadout={size}) --")
    print(f"  recall  base_rate : {_fmt(rep['recall_base_rate'])} ({rep['recall_base_rate_n']})")
    print(f"  recall  composite : {_fmt(rep['recall_composite'])} ({rep['recall_composite_n']})")
    print(f"  recall  OVERALL   : {_fmt(rep['recall_overall'])} ({rep['recall_overall_n']})")
    print(f"  overfire pure_ta  : {_fmt(rep['overfire_pure_ta'])} ({rep['overfire_pure_ta_n']})   [lower better]")
    print(f"  overfire other    : {_fmt(rep['overfire_other'])} ({rep['overfire_other_n']})   [lower better]")
    print(f"  FPR     OVERALL   : {_fmt(rep['fpr_overall'])} ({rep['fpr_overall_n']})   [lower better]")


def print_ab(rep):
    n = rep.get("n", 0)
    print(f"\n-- Answer lift: with vs without chartlibrary (n={n} should-use prompts) --")
    if not n:
        return
    for label in rep["labels"]:
        r = rep["per_judge"][label]
        tag = "   [SYNTHETIC - plumbing only]" if r.get("synthetic") else ""
        print(f"  judge={label}:{tag}")
        print(f"    with wins : {r['with_wins']}   ties: {r['ties']}   without wins: {r['without_wins']}")
        print(f"    with win-rate (all)     : {_fmt(r['with_win_rate_all'])}")
        print(f"    with win-rate (decided) : {_fmt(r['with_win_rate_decided'])}")
    if rep.get("judge_agreement") is not None and len(rep["labels"]) >= 2:
        print(f"  inter-judge agreement   : {_fmt(rep['judge_agreement'])}  "
              f"({rep['labels'][0]} vs {rep['labels'][1]})")


def main():
    args = parse_args()
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    n_pos = sum(p["expects_chartlibrary"] for p in prompts)
    sizes = loadout_sizes(args.loadout)

    if args.key_file:
        with open(args.key_file) as f:
            key = f.read().strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key  # never printed

    if args.mode == "live":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Live mode needs an API key. In this PowerShell session set it, e.g.:")
            print('  $env:ANTHROPIC_API_KEY = "<your-key>"')
            print("then re-run with --yes.")
            sys.exit(2)
        if not args.yes:
            n_judges = 2 if (args.judge_model_2 and args.judge_model_2 != args.judge_model) else 1
            est = estimate_calls(len(prompts), sizes, args.ab, n_pos, args.skip_selection, n_judges)
            judges_desc = args.judge_model + (f" + {args.judge_model_2}" if n_judges == 2 else "")
            print(f"Live mode will make ~{est} paid Anthropic calls "
                  f"(orchestrator={args.orchestrator_model}, judges={judges_desc}).")
            print("Re-run with --yes to authorize the spend.")
            sys.exit(3)

    real_runner = make_real_runner() if args.real_chartlibrary else None
    if args.real_chartlibrary:
        if not args.ab:
            print("[note] --real-chartlibrary only affects the answer-lift A/B; pass --ab "
                  "to use it. Selection uses fixtures regardless (output can't change which "
                  "tool the model already picked).\n")
        else:
            print("[note] --real-chartlibrary: the A/B 'with' arm calls the LIVE "
                  "chartlibrary endpoint (rivals stay fixtures).\n")

    synthetic = args.mode == "mock"
    banner = ("SYNTHETIC - plumbing only, NOT a validation result"
              if synthetic else "LIVE - real tool selection")
    print("=== Phase 0 chartlibrary selection harness ===")
    print(f"mode: {args.mode}  ({banner})")
    print(f"desc arm: {args.desc}   orchestrator: {args.orchestrator_model}   prompts: {len(prompts)}")

    if args.mode == "mock":
        backend = MockModel(args.desc)
        judges = [("mock", lambda rec, w, wo, rng: mock_judge(rec, w, wo, rng))]
    else:
        backend = LiveModel(args.orchestrator_model)
        judges = [(args.judge_model,
                   lambda rec, w, wo, rng, m=args.judge_model:
                       live_judge(backend.client, m, rec, w, wo, rng))]
        if args.judge_model_2 and args.judge_model_2 != args.judge_model:
            judges.append((args.judge_model_2,
                           lambda rec, w, wo, rng, m=args.judge_model_2:
                               live_judge(backend.client, m, rec, w, wo, rng)))

    t0 = time.time()
    results = {"meta": {"mode": args.mode, "synthetic": synthetic, "desc": args.desc,
                        "orchestrator_model": args.orchestrator_model,
                        "judge_model": args.judge_model if args.ab else None,
                        "judge_model_2": (args.judge_model_2 if args.ab
                                          and args.mode == "live" else None),
                        "n_prompts": len(prompts), "loadouts": sizes,
                        "real_chartlibrary": args.real_chartlibrary,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
               "selection": {}, "episodes": {}}

    if args.skip_selection:
        print("\n[note] --skip-selection: skipping the selection pass; running the A/B only.")
    else:
        for size in sizes:
            eps = run_selection(prompts, backend, args.desc, size)
            rep = selection_report(eps)
            results["selection"][str(size)] = rep
            results["episodes"][str(size)] = eps
            print_selection(size, rep)

    if args.ab:
        positives = [p for p in prompts if p["expects_chartlibrary"]]
        records = run_ab(positives, backend, args.desc, judges, tool_runner=real_runner)
        rep = ab_report_multi(records, [lbl for lbl, _ in judges])
        results["ab"] = {"report": rep, "records": records}
        print_ab(rep)
        if real_runner is not None:
            results["ab"]["real_chartlibrary_calls"] = real_runner.stats
            s = real_runner.stats
            print(f"\n-- Real chartlibrary calls (live endpoint) --")
            print(f"  ok: {s['ok']}   degraded(no-ticker): {s['degraded']}   "
                  f"errored: {s['error']}   rival-fixtures: {s['fixture']}")
            if s["error"]:
                print("  WARNING: some real calls errored — a depressed lift may reflect "
                      "integration failure, not no-value. Inspect results.json.")

    results["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}  ({results['meta']['elapsed_sec']}s)")

    if synthetic:
        print("\nNOTE: mock numbers are synthetic and prove only that the harness runs. "
              "The real GO/NO-GO needs --mode live (paid, --yes).")
    else:
        _print_verdict_hint(results)


def _print_verdict_hint(results):
    sel_all = results["selection"]
    sel = sel_all.get("7") or (next(iter(sel_all.values())) if sel_all else {})
    rc, fpr = sel.get("recall_overall"), sel.get("fpr_overall")
    ab = results.get("ab", {}).get("report", {})
    wr = {lbl: ab.get("per_judge", {}).get(lbl, {}).get("with_win_rate_decided")
          for lbl in ab.get("labels", [])}
    print("\n-- Suggested read (Graham decides) --")
    print("  GO leans true if: recall_overall >= ~0.80, fpr_overall <= ~0.15, "
          "with-win-rate(decided) >= ~0.60.")
    print(f"  observed: recall_overall={rc}, fpr_overall={fpr}, "
          f"with_win_rate_decided(by judge)={wr}, "
          f"judge_agreement={ab.get('judge_agreement')}")


if __name__ == "__main__":
    main()
