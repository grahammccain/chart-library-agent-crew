"""
Receipt-ablation eval v1 — the legitimate replacement for the retired "50-0".

Spec: research/specs/receipt_ablation_eval_v1_2026_06_10.md (READ FIRST).

The retired 50-0 study compared a TOOLED agent vs an UNTOOLED one — rigged by
construction (a superset of tools) and read as such by sophisticated reviewers.
The narrower, defensible claim is ours alone: **does the calibration RECEIPT
itself change agent reasoning quality?** SAME toolkit on both arms; the only
delta is the receipt.

  Arm A (receipt-on) : the chartlibrary node as-is — cohort_analyze responses
                       include the coverage record (`calibration`) + `provenance`
                       (production behavior).
  Arm B (receipt-off): IDENTICAL agent, IDENTICAL tool — but a thin response
                       filter STRIPS exactly the receipt keys from the same tool
                       result. The agent still gets the full outcome
                       distribution, feature importance, regime stratification,
                       risk profile — everything EXCEPT the calibrated bands /
                       coverage stats / provenance attribution.

Both arms see the same prompts, the same orchestrator, the same rivals; the lift
(if any) is attributable to the receipt, not the toolset.

Blind dual-judge: the existing 6 reasoning dimensions PLUS two new ones from the
spec — (7) confidence-qualification (does the answer correctly QUALIFY its
confidence) and (8) weak-comp-set refusal/flagging (does it flag/refuse when the
comp set is thin). Judges are NEVER told which arm is which (randomised A/B order;
arm labels never reach the judge prompt — see test_receipt_ablation.py).

Pre-registered predictions + honest-outcome handling are printed VERBATIM in the
report header so the result is interpreted the same way no matter which way it
lands (receipt-wins → marketing number; parity/loss → reweight toward
human-facing receipt surfaces — both publishable).

Run:
  python receipt_ablation.py --mode mock                       # free, SYNTHETIC, e2e
  python receipt_ablation.py --mode live --yes --env-file PATH # paid, real (~$10-30)

Live needs an Anthropic key (via --env-file dotenv or ANTHROPIC_API_KEY in the
environment) AND an explicit --yes; the estimated cost is printed BEFORE any spend
and the key is never printed. Mirrors probe_chaining.py's mock/live contract.

NOTE on the spec's lexicon: the spec uses the proposed institutional names
(`pull_comps` / `coverage_record`). Those are a GATED rename not yet in this repo;
here the real tool is `chartlibrary_cohort_analyze` and the coverage record is the
top-level `calibration` block. The mapping is: pull_comps≈chartlibrary_search,
cohort_analyze≈chartlibrary_cohort_analyze, coverage_record≈`calibration`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from evaluate import _rate
from harness import LiveModel, MockModel, run_episode
from real_tools import make_real_runner
from receipt_subjects import SUBJECTS, distinct_dates
from tools import CHARTLIBRARY_TOOL_NAMES, build_loadout, run_tool

DEFAULT_ORCH = "claude-sonnet-4-6"
DEFAULT_JUDGE = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_2 = "claude-opus-4-7"

# ---------------------------------------------------------------------------
# THE RECEIPT KEYS — exactly what Arm B strips, and nothing else.
# ---------------------------------------------------------------------------
# The "receipt" = the calibrated/coverage record + the provenance attribution.
# `calibration` carries the split-conformal CALIBRATED 80% band, the empirical
# band_coverage_observed (~0.808 over ~303K cases), calibration_n, the conformal
# multiplier / online_theta, reliability + match_level. `provenance` carries the
# attribution string, n_analogs, as_of, embedding_version, and a nested
# calibration coverage summary. Stripping BOTH leaves a tool result that still
# carries the full empirical distribution (outcome_distribution), feature
# importance, regime stratification, risk profile, expected_move, tightness — the
# agent loses ONLY the receipt, never the analysis. The mock fixture (tools.py
# _cohort_fixture) uses `calibrated_80_band` + `note` as its receipt stand-ins, so
# we strip those nested/legacy spellings too. The set is closed and asserted in
# the tests so the ablation can never silently widen.
RECEIPT_KEYS = ("calibration", "provenance")
# nested/fixture receipt fields (mock fixture carries the receipt inside horizons)
_FIXTURE_BAND_KEY = "calibrated_80_band"
_FIXTURE_NOTE_KEY = "note"


def strip_receipt(tool_result: str) -> str:
    """Thin proxy over a tool result: delete EXACTLY the receipt keys, nothing else.

    Operates on the JSON STRING a tool runner returns (so it sits transparently
    between any runner and the agent). Non-cohort_analyze results and malformed
    JSON pass through untouched — the strip is a no-op unless a receipt is present,
    which keeps Arm B byte-identical to Arm A on every other tool. Returns a JSON
    string. Pure / no I/O so it is unit-testable."""
    try:
        obj = json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return tool_result
    if not isinstance(obj, dict):
        return tool_result

    # 1) top-level receipt keys (live response: `calibration`, `provenance`)
    for k in RECEIPT_KEYS:
        obj.pop(k, None)

    # 2) the mock fixture carries its receipt INSIDE each horizon
    #    (`calibrated_80_band`) + a calibration `note`. Strip those too so the
    #    mock arms genuinely differ — without touching p10/p50/p90/pct_positive.
    horizons = obj.get("horizons")
    if isinstance(horizons, dict):
        for hz in horizons.values():
            if isinstance(hz, dict):
                hz.pop(_FIXTURE_BAND_KEY, None)
    note = obj.get(_FIXTURE_NOTE_KEY)
    if isinstance(note, str) and ("calibrat" in note.lower() or "80% band" in note.lower()):
        obj.pop(_FIXTURE_NOTE_KEY, None)

    return json.dumps(obj)


def receipt_keys_present(tool_result: str) -> bool:
    """True if a tool result still carries any receipt field (used by tests +
    the mock e2e to prove Arm A keeps what Arm B drops)."""
    try:
        obj = json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    if any(k in obj for k in RECEIPT_KEYS):
        return True
    horizons = obj.get("horizons")
    if isinstance(horizons, dict):
        for hz in horizons.values():
            if isinstance(hz, dict) and _FIXTURE_BAND_KEY in hz:
                return True
    return False


def make_arm_runner(base_runner, strip: bool):
    """Wrap a tool runner so cohort_analyze responses (and only those) optionally
    pass through strip_receipt. `base_runner(name, args) -> json_str`. When
    strip=False this is the identity (Arm A); when strip=True it is Arm B. EXACT
    same runner underneath both arms — the ONLY difference is the post-filter, so
    the ablation is clean."""
    def runner(name: str, args: dict) -> str:
        out = base_runner(name, args)
        if strip and name == "chartlibrary_cohort_analyze":
            return strip_receipt(out)
        return out
    return runner


# ---------------------------------------------------------------------------
# Pre-registered predictions + honest-outcome handling (printed VERBATIM)
# ---------------------------------------------------------------------------
# Lifted from the spec so the result is read the same way no matter how it lands.

PRE_REGISTERED = """\
PRE-REGISTERED PREDICTIONS (locked before the live run; from the spec):
  * The RECEIPT arm wins on the two qualification dimensions
    (confidence-qualification, weak-comp-set refusal/flagging).
  * PARITY is EXPECTED on investigation breadth -- same tools, same data, only
    the receipt differs. Parity there is the point, not a failure.

HONEST-OUTCOME HANDLING (locked before the live run; from the spec):
  * Receipt arm wins  -> THE marketing number ("same tools; a blind judge
    preferred the reasoning grounded in the coverage record") + a charter-review
    exhibit.
  * Parity / loss     -> EQUALLY publishable: receipts inform humans more than
    agents -> reweight the roadmap toward human-facing receipt surfaces. Either
    result is real. No number may be cited until --mode live has run."""

# Eight blind reasoning dimensions: the six prior + two new (spec).
DIMENSIONS = [
    ("groundedness", "answer is grounded in verifiable historical fact, not vague talk"),
    ("specificity", "concrete base rates / ranges over hand-wavy qualitative claims"),
    ("calibration_honesty", "no made-up precision; ranges/uncertainty stated honestly"),
    ("usefulness", "actionable for a trader without overclaiming"),
    ("no_overclaim", "does NOT present a directional forecast as fact"),
    ("internal_consistency", "the stated numbers and the conclusion agree"),
    # --- two NEW dimensions (spec) ---
    ("confidence_qualification", "correctly QUALIFIES its confidence "
                                 "(calibration-awareness): hedges appropriately, "
                                 "neither overconfident nor falsely vague"),
    ("weak_comp_set_flagging", "FLAGS or refuses when the comp set / cohort is "
                               "weak (small n, low match quality) rather than "
                               "reading a thin cohort as if it were solid"),
]
DIMENSION_KEYS = [d for d, _ in DIMENSIONS]

_JUDGE_INSTRUCTIONS = (
    "You are grading two answers, A and B, to the SAME markets question. Both came "
    "from the same agent using the same tools; you do NOT know how they differ and "
    "must not guess. For EACH of the dimensions below, decide which answer is "
    "better, or 'tie'. Then give an overall winner.\n\n"
    "DIMENSIONS:\n"
    + "\n".join(f"  - {k}: {desc}" for k, desc in DIMENSIONS)
    + "\n\nReward concrete base rates / calibrated ranges and honest hedging over "
    "vague talk or false precision; reward an answer that flags a weak/thin "
    "comparison set; penalise made-up specifics and a directional forecast stated "
    "as fact. Reply with ONLY JSON of the form:\n"
    "{\"dimensions\": {\"<dim>\": \"A\"|\"B\"|\"tie\", ...}, "
    "\"overall\": \"A\"|\"B\"|\"tie\", \"reason\": \"<8 words>\"}"
)


def _parse_judge(raw: str) -> dict:
    """Parse the judge's JSON; tolerant of prose around it. Returns
    {'dimensions': {dim: 'A'|'B'|'tie'}, 'overall': 'A'|'B'|'tie'} with safe
    defaults ('tie') for anything missing or malformed."""
    dims = {k: "tie" for k in DIMENSION_KEYS}
    overall = "tie"
    try:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1])
        d = obj.get("dimensions", {})
        if isinstance(d, dict):
            for k in DIMENSION_KEYS:
                v = str(d.get(k, "tie")).strip().upper()
                dims[k] = {"A": "A", "B": "B"}.get(v, "tie")
        ov = str(obj.get("overall", "tie")).strip().upper()
        overall = {"A": "A", "B": "B"}.get(ov, "tie")
    except Exception:
        pass
    return {"dimensions": dims, "overall": overall}


def _verdict_from_pick(pick: str, swap: bool) -> str:
    """Map a blind A/B pick back to the receipt/no_receipt arm given the swap."""
    if pick == "tie":
        return "tie"
    # when swap is False: A=receipt, B=no_receipt; when True: A=no_receipt, B=receipt
    if pick == "A":
        return "no_receipt" if swap else "receipt"
    return "receipt" if swap else "no_receipt"


def mock_judge(rec, ans_receipt, ans_noreceipt, rng) -> dict:
    """SYNTHETIC verdict — free, deterministic, plumbing only. Leans the two
    qualification dimensions toward the receipt arm (mirroring the pre-registered
    prediction) and ties breadth-like dimensions, so the mock report SHAPE matches
    a real one. NOT evidence — purely to exercise the dual-judge math end-to-end."""
    dims = {}
    for k in DIMENSION_KEYS:
        roll = rng.random()
        if k in ("confidence_qualification", "weak_comp_set_flagging",
                 "calibration_honesty"):
            dims[k] = "receipt" if roll < 0.68 else ("tie" if roll < 0.88 else "no_receipt")
        else:  # breadth-like dimensions: parity expected
            dims[k] = "tie" if roll < 0.62 else ("receipt" if roll < 0.81 else "no_receipt")
    recv = sum(v == "receipt" for v in dims.values())
    norv = sum(v == "no_receipt" for v in dims.values())
    overall = "receipt" if recv > norv else ("no_receipt" if norv > recv else "tie")
    return {"dimensions": dims, "overall": overall, "synthetic": True}


def live_judge(client, model, rec, ans_receipt, ans_noreceipt, rng) -> dict:
    """Real model call, BLIND to which arm is which (randomised A/B order). The
    arm labels never appear in the prompt — only neutral 'A'/'B'. Per-dimension +
    overall verdicts are mapped back to receipt/no_receipt AFTER the call."""
    swap = rng.random() < 0.5
    a, b = (ans_noreceipt, ans_receipt) if swap else (ans_receipt, ans_noreceipt)
    user = (f"QUESTION:\n{rec['prompt']}\n\n"
            f"ANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n{_JUDGE_INSTRUCTIONS}")
    resp = client.messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(blk.text for blk in resp.content if blk.type == "text").strip()
    parsed = _parse_judge(raw)
    dims = {k: _verdict_from_pick(parsed["dimensions"][k], swap) for k in DIMENSION_KEYS}
    overall = _verdict_from_pick(parsed["overall"], swap)
    return {"dimensions": dims, "overall": overall, "synthetic": False, "raw": raw}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _arm_report(judgements) -> dict:
    """Win/tie/loss for the RECEIPT arm, overall + per dimension."""
    n = len(judgements)
    if not n:
        return {"n": 0}

    def tally(picks):
        w = sum(p == "receipt" for p in picks)
        t = sum(p == "tie" for p in picks)
        loss = sum(p == "no_receipt" for p in picks)
        decided = w + loss
        return {"receipt_wins": w, "ties": t, "no_receipt_wins": loss,
                "receipt_win_rate_all": _rate(w, n),
                "receipt_win_rate_decided": _rate(w, decided)}

    overall = tally([j["overall"] for j in judgements])
    per_dim = {k: tally([j["dimensions"][k] for j in judgements]) for k in DIMENSION_KEYS}
    return {"n": n, "overall": overall, "per_dimension": per_dim,
            "synthetic": all(j.get("synthetic") for j in judgements)}


def report_multi(records, labels) -> dict:
    """Per-judge receipt-arm report over the SAME answer pairs + overall
    inter-judge agreement (true inter-rater stat — both judges grade an identical
    receipt/no-receipt pair)."""
    labels = list(labels)
    per = {lbl: _arm_report([r["judges"][lbl] for r in records
                             if lbl in r.get("judges", {})])
           for lbl in labels}
    agreement = None
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        both = [r for r in records
                if a in r.get("judges", {}) and b in r.get("judges", {})]
        if both:
            same = sum(r["judges"][a]["overall"] == r["judges"][b]["overall"]
                       for r in both)
            agreement = _rate(same, len(both))
    return {"n": len(records), "labels": labels, "per_judge": per,
            "judge_agreement": agreement}


# ---------------------------------------------------------------------------
# Run an arm (one orchestrator pass per subject)
# ---------------------------------------------------------------------------

def _subject_to_rec(s) -> dict:
    """Adapt a receipt subject to the {id, prompt, category, expects_chartlibrary}
    shape run_episode expects."""
    return {"id": s["id"], "prompt": s["prompt"],
            "category": s.get("category", "base_rate"),
            "expects_chartlibrary": True}


def run_pair(subjects, backend, base_runner) -> list:
    """For each subject run the SAME orchestrator twice — Arm A (receipt-on) and
    Arm B (receipt-off via strip_receipt) — and persist both finals + whether the
    with-arm actually fired chartlibrary. The two arms share one base_runner; the
    only delta is the receipt strip."""
    loadout = build_loadout("v2", 7)
    runner_a = make_arm_runner(base_runner, strip=False)  # receipt-on
    runner_b = make_arm_runner(base_runner, strip=True)   # receipt-off
    records = []
    for s in subjects:
        rec = _subject_to_rec(s)
        ep_a = run_episode(rec, loadout, backend, tool_runner=runner_a)
        ep_b = run_episode(rec, loadout, backend, tool_runner=runner_b)
        fired = any(c in CHARTLIBRARY_TOOL_NAMES for c in ep_a["called"])
        records.append({
            "id": s["id"], "date": s.get("date"),
            "fired_chartlibrary": fired,
            "receipt_called": ep_a["called"],
            "receipt_final": ep_a["final"],
            "no_receipt_final": ep_b["final"],
            "judges": {},
        })
    return records


def judge_records(records, judges):
    """Each judge scores the SAME receipt/no-receipt pair for every record."""
    for r in records:
        for label, judge_fn in judges:
            rng = random.Random(f"recv|{label}|{r['id']}")
            r["judges"][label] = judge_fn(r, r["receipt_final"],
                                          r["no_receipt_final"], rng)
    return records


# ---------------------------------------------------------------------------
# Cost / key handling (mirrors probe_chaining.py)
# ---------------------------------------------------------------------------

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


def estimate_cost(n_subjects, n_judges):
    """Rough paid-call + dollar estimate for the live run. Two arms x ~2
    orchestrator calls/episode + n_judges judge calls per subject."""
    orch_calls = 2 * n_subjects * 2
    judge_calls = n_judges * n_subjects
    total = orch_calls + judge_calls
    # crude $: sonnet orchestrator ~ $0.03/call (tool loop), judge ~ $0.01-0.02/call.
    low = orch_calls * 0.03 + judge_calls * 0.01
    high = orch_calls * 0.06 + judge_calls * 0.03
    return total, low, high


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(rate):
    return "  n/a" if rate is None else f"{rate:>5.2f}"


def print_report(rep, n_dates, synthetic):
    print("\n-- Receipt-ablation: receipt-ON vs receipt-OFF (same toolkit) --")
    print(f"   subjects={rep['n']}   distinct anchor dates={n_dates}")
    for label in rep["labels"]:
        r = rep["per_judge"][label]
        if not r.get("n"):
            continue
        tag = "   [SYNTHETIC - plumbing only]" if r.get("synthetic") else ""
        ov = r["overall"]
        print(f"  judge={label}:{tag}")
        print(f"    OVERALL  receipt wins {ov['receipt_wins']} | ties {ov['ties']} | "
              f"no-receipt wins {ov['no_receipt_wins']}   "
              f"win-rate(decided)={_fmt(ov['receipt_win_rate_decided'])}")
        for k in DIMENSION_KEYS:
            d = r["per_dimension"][k]
            print(f"      {k:26s} win-rate(decided)={_fmt(d['receipt_win_rate_decided'])}  "
                  f"(w{d['receipt_wins']}/t{d['ties']}/l{d['no_receipt_wins']})")
    if rep.get("judge_agreement") is not None and len(rep["labels"]) >= 2:
        print(f"  inter-judge agreement (overall): {_fmt(rep['judge_agreement'])}  "
              f"({rep['labels'][0]} vs {rep['labels'][1]})")


def write_markdown(path, results):
    """Human-facing markdown summary alongside the JSON."""
    meta = results["meta"]
    rep = results["report"]
    lines = []
    lines.append("# Receipt-ablation eval v1 — results")
    lines.append("")
    lines.append(f"- mode: **{meta['mode']}**"
                 + ("  _(SYNTHETIC — plumbing only, NOT a result)_" if meta["synthetic"]
                    else "  _(LIVE)_"))
    lines.append(f"- subjects: {meta['n_subjects']}  |  distinct anchor dates: {meta['n_dates']}")
    lines.append(f"- orchestrator: `{meta['orchestrator_model']}`  |  "
                 f"judges: {', '.join(f'`{j}`' for j in meta['judges'])}")
    lines.append(f"- receipt keys stripped in Arm B: {', '.join('`'+k+'`' for k in RECEIPT_KEYS)} "
                 f"(+ nested `{_FIXTURE_BAND_KEY}` / calibration `{_FIXTURE_NOTE_KEY}` in fixtures)")
    lines.append(f"- timestamp: {meta['timestamp']}")
    lines.append("")
    lines.append("## Pre-registered predictions + honest-outcome handling")
    lines.append("")
    lines.append("```")
    lines.append(PRE_REGISTERED)
    lines.append("```")
    lines.append("")
    if meta["synthetic"]:
        lines.append("> **SYNTHETIC run.** The numbers below are deterministic mock "
                     "verdicts that prove the wiring only. No number may be cited until "
                     "`--mode live` has run.")
        lines.append("")
    lines.append("## Receipt-arm win-rates (decided)")
    lines.append("")
    header = "| dimension | " + " | ".join(rep["labels"]) + " |"
    sep = "|---|" + "|".join("---" for _ in rep["labels"]) + "|"
    lines.append(header)
    lines.append(sep)
    for k in ["overall"] + DIMENSION_KEYS:
        cells = []
        for lbl in rep["labels"]:
            pj = rep["per_judge"].get(lbl, {})
            block = pj.get("overall") if k == "overall" else pj.get("per_dimension", {}).get(k)
            wr = block.get("receipt_win_rate_decided") if isinstance(block, dict) else None
            cells.append("n/a" if wr is None else f"{wr:.2f}")
        bold = "**" if k == "overall" else ""
        lines.append(f"| {bold}{k}{bold} | " + " | ".join(cells) + " |")
    if rep.get("judge_agreement") is not None:
        lines.append("")
        lines.append(f"Inter-judge agreement (overall): {rep['judge_agreement']:.2f}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Receipt-ablation eval v1 "
                                             "(same toolkit, with/without the receipt)")
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--orchestrator-model", default=DEFAULT_ORCH)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--judge-model-2", default=DEFAULT_JUDGE_2,
                    help="second blind judge for the dual-judge inter-rater check "
                         "(set to '' to use a single judge)")
    ap.add_argument("--yes", action="store_true", help="authorize live spend")
    ap.add_argument("--env-file", default=None,
                    help="dotenv file to read ANTHROPIC_API_KEY from (never printed)")
    ap.add_argument("--real-chartlibrary", action="store_true",
                    help="in --mode live, call the LIVE chartlibrary endpoint for the "
                         "chartlibrary tools (recommended for the real run); fixtures otherwise")
    ap.add_argument("--limit", type=int, default=0, help="cap #subjects (0=all)")
    ap.add_argument("--out", default="results_receipt_ablation.json")
    ap.add_argument("--md-out", default="results_receipt_ablation.md")
    args = ap.parse_args()

    load_key(args.env_file)
    subjects = SUBJECTS[: args.limit] if args.limit else SUBJECTS
    n_dates = distinct_dates(subjects)

    judge_labels = [args.judge_model]
    if args.judge_model_2 and args.judge_model_2 != args.judge_model:
        judge_labels.append(args.judge_model_2)

    if args.mode == "live":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Live mode needs ANTHROPIC_API_KEY (use --env-file or set it). Aborting.")
            sys.exit(2)
        total, low, high = estimate_cost(len(subjects), len(judge_labels))
        if not args.yes:
            print(f"Live mode will make ~{total} paid Anthropic calls across both arms "
                  f"(orchestrator={args.orchestrator_model}, judges={', '.join(judge_labels)}).")
            print(f"Estimated cost: ~${low:.0f}-${high:.0f}.")
            print("Re-run with --yes to authorize the spend.")
            sys.exit(3)
        # authorized: still print the estimate before proceeding
        print(f"[live] authorized. ~{total} paid calls, est ~${low:.0f}-${high:.0f}.")

    synthetic = args.mode == "mock"
    banner = ("SYNTHETIC - plumbing only, NOT a result" if synthetic
              else "LIVE - real receipt ablation")
    print("=== Receipt-ablation eval v1 ===")
    print(f"mode: {args.mode}  ({banner})")
    print(f"subjects: {len(subjects)}   distinct anchor dates: {n_dates}   "
          f"orchestrator: {args.orchestrator_model}")
    print()
    print(PRE_REGISTERED)

    if args.mode == "mock":
        backend = MockModel("v2")
        base_runner = run_tool  # deterministic fixtures (carry a receipt)
        judges = [(args.judge_model, lambda r, rcv, nor, rng: mock_judge(r, rcv, nor, rng))]
        if len(judge_labels) > 1:
            judges.append((args.judge_model_2,
                           lambda r, rcv, nor, rng: mock_judge(r, rcv, nor, rng)))
    else:
        backend = LiveModel(args.orchestrator_model)
        base_runner = make_real_runner() if args.real_chartlibrary else run_tool
        judges = [(args.judge_model,
                   lambda r, rcv, nor, rng, m=args.judge_model:
                       live_judge(backend.client, m, r, rcv, nor, rng))]
        if len(judge_labels) > 1:
            judges.append((args.judge_model_2,
                           lambda r, rcv, nor, rng, m=args.judge_model_2:
                               live_judge(backend.client, m, r, rcv, nor, rng)))

    t0 = time.time()
    records = run_pair(subjects, backend, base_runner)
    records = judge_records(records, judges)
    rep = report_multi(records, judge_labels)

    print_report(rep, n_dates, synthetic)

    results = {
        "meta": {"mode": args.mode, "synthetic": synthetic,
                 "orchestrator_model": args.orchestrator_model,
                 "judges": judge_labels, "n_subjects": len(subjects),
                 "n_dates": n_dates, "receipt_keys_stripped": list(RECEIPT_KEYS),
                 "real_chartlibrary": args.real_chartlibrary and args.mode == "live",
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "pre_registered": PRE_REGISTERED,
        "report": rep,
        "records": records,
    }
    results["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_markdown(args.md_out, results)
    print(f"\nwrote {args.out} + {args.md_out}  ({results['meta']['elapsed_sec']}s)")

    if synthetic:
        print("\nNOTE: mock verdicts are synthetic and prove only that the harness runs "
              "end-to-end. No number may be cited until --mode live (paid, --yes).")


if __name__ == "__main__":
    main()
