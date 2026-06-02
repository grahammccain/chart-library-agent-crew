"""
Phase 0 CI gate.

Turns the harness's "suggested read" (run.py) into an ENFORCED exit code so CI can
fail when selection or answer-lift regresses. Reads a results.json produced by
run.py; it NEVER runs the model or spends money itself.

  python gate.py results.json                # enforce the default thresholds
  python gate.py results.json --require-live  # REFUSE to certify synthetic (mock) numbers

GO thresholds (defaults match run.py's suggested read):
  recall_overall          >= 0.80   the node fires when it should
  over-fire (fpr_overall)  <= 0.15   the node does NOT fire when it should not
  answer-lift (decided)    >= 0.60   the with-node answer wins the blind A/B
With multiple judges, EVERY judge must clear the lift bar (gate on the weakest).

Honesty guard: mock numbers are SYNTHETIC. By default a mock results file is a
PLUMBING check only (verifies a well-formed, gateable report; the thresholds are
informational, not a verdict). With --require-live a synthetic file FAILS - only
live numbers certify.

Exit: 0 = pass, 1 = gate failed, 2 = unreadable results / usage.
"""
from __future__ import annotations

import argparse
import json
import sys

RECALL_MIN = 0.80
OVERFIRE_MAX = 0.15
LIFT_MIN = 0.60


def _numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _pick_loadout(selection, want):
    if not selection:
        return None, None
    if want in selection:
        return want, selection[want]
    k = next(iter(selection))
    return k, selection[k]


def _lifts(ab_report):
    """Both A/B shapes -> {judge_label: decided_win_rate}.
    multi-judge (ab_report_multi): report.per_judge[label].with_win_rate_decided
    single-judge (legacy ab_report): flat report.with_win_rate_decided."""
    if not ab_report:
        return {}
    if "per_judge" in ab_report:
        return {lbl: ab_report["per_judge"].get(lbl, {}).get("with_win_rate_decided")
                for lbl in ab_report.get("labels", [])}
    if "with_win_rate_decided" in ab_report:
        return {"judge": ab_report.get("with_win_rate_decided")}
    return {}


def _chk(name, value, ok, detail):
    present = _numeric(value)
    return {"name": name, "present": present,
            "ok": bool(ok) if present else False, "detail": detail}


def _skip(name, detail):
    return {"name": name, "present": None, "ok": None, "detail": detail}


def evaluate_gate(res, loadout="7", recall_min=RECALL_MIN, overfire_max=OVERFIRE_MAX,
                  lift_min=LIFT_MIN, require_live=False):
    """Pure decision core (no I/O) so it is unit-testable. Returns a verdict dict
    with passed/exit_code/synthetic/checks/summary."""
    synthetic = bool(res.get("meta", {}).get("synthetic"))

    if synthetic and require_live:
        guard = _chk("synthetic-guard", 0, False,
                     "results are SYNTHETIC and --require-live set; mock never certifies")
        return {"passed": False, "exit_code": 1, "synthetic": True, "checks": [guard],
                "summary": "FAIL: refusing to certify synthetic numbers (--require-live)"}

    size_key, sel = _pick_loadout(res.get("selection") or {}, loadout)
    lifts = _lifts((res.get("ab") or {}).get("report") or {})
    checks = []

    if sel:
        recall = sel.get("recall_overall")
        overfire = sel.get("fpr_overall")
        checks.append(_chk(f"recall_overall (loadout {size_key})", recall,
                           _numeric(recall) and recall >= recall_min,
                           f"{recall} ({sel.get('recall_overall_n', '?')}); need >= {recall_min}"))
        checks.append(_chk(f"over-fire fpr (loadout {size_key})", overfire,
                           _numeric(overfire) and overfire <= overfire_max,
                           f"{overfire} ({sel.get('fpr_overall_n', '?')}); need <= {overfire_max}"))
    else:
        checks.append(_skip("selection", "selection pass not in results - NOT gated"))

    if lifts:
        for lbl, wr in lifts.items():
            checks.append(_chk(f"answer-lift decided [judge={lbl}]", wr,
                               _numeric(wr) and wr >= lift_min,
                               f"{wr}; need >= {lift_min}"))
    else:
        checks.append(_skip("answer-lift", "A/B not in results - NOT gated"))

    ran = [c for c in checks if c["present"] is not None]

    if synthetic:
        ok = bool(ran) and all(c["present"] for c in ran)
        return {"passed": ok, "exit_code": 0 if ok else 1, "synthetic": True, "checks": checks,
                "summary": ("PASS (plumbing): well-formed, gateable report" if ok
                            else "FAIL (plumbing): no well-formed gateable section")}

    if not ran:
        return {"passed": False, "exit_code": 1, "synthetic": False, "checks": checks,
                "summary": "FAIL: nothing to gate (no selection, no A/B in results)"}

    breaches = [c["name"] for c in ran if not c["ok"]]
    ok = not breaches
    return {"passed": ok, "exit_code": 0 if ok else 1, "synthetic": False, "checks": checks,
            "summary": (f"PASS: all {len(ran)} live checks cleared thresholds" if ok
                        else "FAIL: " + ", ".join(breaches))}


def _tag(c, info_mode):
    if c["present"] is None:
        return "[----]"
    if not c["present"]:
        return "[FAIL]"
    if info_mode:
        return "[mock]"
    return "[PASS]" if c["ok"] else "[FAIL]"


def main():
    ap = argparse.ArgumentParser(description="Phase 0 selection + answer-lift CI gate")
    ap.add_argument("results", help="results.json produced by run.py")
    ap.add_argument("--loadout", default="7", help="selection loadout to gate (default 7)")
    ap.add_argument("--recall-min", type=float, default=RECALL_MIN)
    ap.add_argument("--overfire-max", type=float, default=OVERFIRE_MAX)
    ap.add_argument("--lift-min", type=float, default=LIFT_MIN)
    ap.add_argument("--require-live", action="store_true",
                    help="FAIL on synthetic (mock) results; only live numbers certify")
    a = ap.parse_args()

    try:
        with open(a.results) as f:
            res = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[gate] cannot read {a.results}: {e}", file=sys.stderr)
        sys.exit(2)

    v = evaluate_gate(res, a.loadout, a.recall_min, a.overfire_max, a.lift_min, a.require_live)
    info_mode = v["synthetic"] and not a.require_live

    print("=== Phase 0 eval gate ===")
    print(f"results: {a.results}   mode: {res.get('meta', {}).get('mode', '?')}   "
          f"thresholds: recall>={a.recall_min}, over-fire<={a.overfire_max}, lift>={a.lift_min}")
    print()
    for c in v["checks"]:
        print(f"  {_tag(c, info_mode)} {c['name']}: {c['detail']}")
    print()
    if info_mode:
        print("NOTE: SYNTHETIC (mock) results - PLUMBING check only; thresholds above are")
        print("      informational. The real GO/NO-GO needs --mode live (paid).")
        print()
    print(v["summary"])
    sys.exit(v["exit_code"])


if __name__ == "__main__":
    main()
