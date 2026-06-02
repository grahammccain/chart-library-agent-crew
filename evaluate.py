"""
Metrics for the Phase 0 harness.

Two questions, two families of metric:

1. SELECTION  - did the orchestrator reach for chartlibrary at the right times?
   * recall      = of prompts that SHOULD use it, how many did      (higher better)
   * over-fire   = of prompts that should NOT,    how many did anyway (lower better)
   A node that is silently ignored fails recall; a node that fires on a pure-RSI
   question fails over-fire. Both are real failure modes from the tool-selection
   literature, so we score both — outcome-only evals would miss them.

2. ANSWER LIFT - with the node available vs. removed, which final answer is more
   grounded? Judged blind (randomised A/B order). The win-rate for the
   with-chartlibrary arm is both the proof-it-helps and the marketing receipt.

`mock_*` judges are SYNTHETIC (free, deterministic) and prove the plumbing only.
`live_*` judges make a real model call and are the validation-grade path.
"""

from __future__ import annotations

import json

from tools import CHARTLIBRARY_TOOL_NAMES


def used_chartlibrary(called) -> bool:
    return any(c in CHARTLIBRARY_TOOL_NAMES for c in called)


def _rate(num, den):
    return round(num / den, 3) if den else None


def selection_report(episodes) -> dict:
    """Recall on the should-fire sets, over-fire on the should-not sets."""
    buckets = {"base_rate": [], "composite": [], "pure_ta": [], "other": []}
    for e in episodes:
        buckets.setdefault(e["category"], []).append(e)

    def recall(cat):
        eps = buckets.get(cat, [])
        hits = sum(used_chartlibrary(e["called"]) for e in eps)
        return _rate(hits, len(eps)), hits, len(eps)

    def overfire(cat):
        eps = buckets.get(cat, [])
        bad = sum(used_chartlibrary(e["called"]) for e in eps)
        return _rate(bad, len(eps)), bad, len(eps)

    pos = [e for e in episodes if e["expects_chartlibrary"]]
    neg = [e for e in episodes if not e["expects_chartlibrary"]]
    recall_overall = _rate(sum(used_chartlibrary(e["called"]) for e in pos), len(pos))
    fpr_overall = _rate(sum(used_chartlibrary(e["called"]) for e in neg), len(neg))

    r_br, br_h, br_n = recall("base_rate")
    r_co, co_h, co_n = recall("composite")
    o_ta, ta_b, ta_n = overfire("pure_ta")
    o_ot, ot_b, ot_n = overfire("other")

    return {
        "recall_base_rate": r_br, "recall_base_rate_n": f"{br_h}/{br_n}",
        "recall_composite": r_co, "recall_composite_n": f"{co_h}/{co_n}",
        "recall_overall": recall_overall, "recall_overall_n": f"{sum(used_chartlibrary(e['called']) for e in pos)}/{len(pos)}",
        "overfire_pure_ta": o_ta, "overfire_pure_ta_n": f"{ta_b}/{ta_n}",
        "overfire_other": o_ot, "overfire_other_n": f"{ot_b}/{ot_n}",
        "fpr_overall": fpr_overall, "fpr_overall_n": f"{sum(used_chartlibrary(e['called']) for e in neg)}/{len(neg)}",
    }


# ---------------------------------------------------------------------------
# Answer-quality A/B judges
# ---------------------------------------------------------------------------

def mock_judge(rec, ans_with, ans_without, rng) -> dict:
    """SYNTHETIC verdict. Leans to the with-arm on prompts that genuinely want a
    base rate, ties otherwise. Free; for plumbing, not evidence."""
    if rec["expects_chartlibrary"]:
        roll = rng.random()
        winner = "with" if roll < 0.70 else ("tie" if roll < 0.85 else "without")
    else:
        roll = rng.random()
        winner = "tie" if roll < 0.70 else ("without" if roll < 0.85 else "with")
    return {"winner": winner, "synthetic": True}


_JUDGE_INSTRUCTIONS = (
    "You are grading two answers to the same markets question. Pick the one that is "
    "better grounded in verifiable historical fact and more useful, or 'tie'. Reward "
    "concrete base rates / calibrated ranges over vague qualitative talk; penalise "
    "made-up specifics. Reply with ONLY JSON: {\"winner\": \"A\" | \"B\" | \"tie\", "
    "\"reason\": \"<8 words>\"}."
)


def live_judge(client, model, rec, ans_with, ans_without, rng) -> dict:
    """Real model call, blind to which arm is which (randomised A/B order)."""
    swap = rng.random() < 0.5
    a, b = (ans_without, ans_with) if swap else (ans_with, ans_without)
    user = (f"QUESTION:\n{rec['prompt']}\n\n"
            f"ANSWER A:\n{a}\n\nANSWER B:\n{b}\n\n{_JUDGE_INSTRUCTIONS}")
    resp = client.messages.create(
        model=model, max_tokens=120,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(blk.text for blk in resp.content if blk.type == "text").strip()
    pick = _parse_winner(raw)
    if pick == "tie" or pick is None:
        winner = "tie"
    elif pick == "A":
        winner = "with" if not swap else "without"
    else:  # "B"
        winner = "without" if not swap else "with"
    return {"winner": winner, "synthetic": False, "raw": raw}


def _parse_winner(raw):
    try:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1])
        w = str(obj.get("winner", "")).strip().upper()
        return w if w in {"A", "B", "TIE"} else None
    except Exception:
        up = raw.upper()
        if "TIE" in up:
            return "TIE"
        if '"A"' in up or up.startswith("A"):
            return "A"
        if '"B"' in up or up.startswith("B"):
            return "B"
        return None


def ab_report(judgements) -> dict:
    """Aggregate win/tie/loss for the with-chartlibrary arm."""
    n = len(judgements)
    if not n:
        return {"n": 0}
    wins = sum(j["winner"] == "with" for j in judgements)
    ties = sum(j["winner"] == "tie" for j in judgements)
    losses = sum(j["winner"] == "without" for j in judgements)
    decided = wins + losses
    return {
        "n": n, "with_wins": wins, "ties": ties, "without_wins": losses,
        "with_win_rate_all": _rate(wins, n),
        "with_win_rate_decided": _rate(wins, decided),
        "synthetic": all(j.get("synthetic") for j in judgements),
    }


def ab_report_multi(records, labels) -> dict:
    """Per-judge win-rates over the SAME answer pairs, plus inter-judge agreement.

    Each record carries record['judges'][label] = {'winner': ..., ...}. Because
    every judge scores an identical with/without answer pair, agreement is a true
    inter-rater statistic (not confounded by different answers) — the receipt that
    the lift isn't an artifact of one judge model's quirks."""
    labels = list(labels)
    per = {label: ab_report([r["judges"][label] for r in records
                              if label in r.get("judges", {})])
           for label in labels}
    agreement = None
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        both = [r for r in records
                if a in r.get("judges", {}) and b in r.get("judges", {})]
        if both:
            same = sum(r["judges"][a]["winner"] == r["judges"][b]["winner"]
                       for r in both)
            agreement = _rate(same, len(both))
    return {"n": len(records), "labels": labels,
            "per_judge": per, "judge_agreement": agreement}
