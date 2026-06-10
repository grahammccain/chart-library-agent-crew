"""
Tests for the receipt-ablation eval (run: pytest -q).

Free + offline. Pins the load-bearing properties of the ablation so it can never
silently drift:

  1. strip-proxy removes EXACTLY the receipt keys and nothing else.
  2. mock e2e produces both-arm scores + a report + a markdown file.
  3. subject list has the required date diversity (>=40 distinct anchor dates).
  4. judge-blinding: arm labels ('receipt'/'no_receipt') never reach a judge prompt.
"""

from __future__ import annotations

import json

import pytest

import receipt_ablation as ra
from receipt_subjects import SUBJECTS, distinct_dates
from tools import run_tool


# ---------------------------------------------------------------------------
# 1. strip-proxy: removes exactly the receipt keys, nothing else
# ---------------------------------------------------------------------------

def _live_like_payload():
    """A faithful slice of the live cohort_analyze response shape."""
    return {
        "anchor": {"symbol": "AAPL", "date": "2026-05-22", "timeframe": "1d"},
        "cohort_id": "coh_abc123",
        "cohort_size_actual": 500,
        "outcome_distribution": {"5": {"p10": -4.1, "p50": 0.4, "p90": 5.2,
                                       "win_rate": 0.55, "n": 500}},
        "feature_importance": {"5": [{"feature": "relative_volume", "importance": 0.4}]},
        "regime_stratification": {"low": {"p50": 0.6}},
        "risk_profile": {"max_dd_p50": -3.2},
        "expected_move": {"5": 4.6},
        "cohort_tightness_score": 0.56,
        # --- the receipt (must be stripped) ---
        "calibration": {"calibrated_p10": -4.26, "calibrated_p90": 5.35,
                        "band_coverage_observed": 0.808, "calibration_n": 58207},
        "provenance": {"attribution_string": "Chart Library: 500 analogs...",
                       "n_analogs": 500, "as_of": "2026-05-22"},
    }


def test_strip_removes_exactly_receipt_keys():
    payload = _live_like_payload()
    before = set(payload.keys())
    stripped = json.loads(ra.strip_receipt(json.dumps(payload)))
    after = set(stripped.keys())
    removed = before - after
    # exactly the two top-level receipt keys were removed
    assert removed == set(ra.RECEIPT_KEYS) == {"calibration", "provenance"}
    # and nothing else changed
    kept = before - removed
    for k in kept:
        assert stripped[k] == payload[k], f"strip mutated non-receipt key {k}"


def test_strip_preserves_analysis_block():
    """The outcome distribution / features / regime / risk all survive — the agent
    keeps the analysis, loses only the receipt."""
    stripped = json.loads(ra.strip_receipt(json.dumps(_live_like_payload())))
    for k in ("outcome_distribution", "feature_importance", "regime_stratification",
              "risk_profile", "expected_move", "cohort_tightness_score", "cohort_id"):
        assert k in stripped, f"strip wrongly dropped analysis key {k}"
    assert "calibration" not in stripped and "provenance" not in stripped


def test_strip_on_real_fixture_removes_receipt():
    """On the actual mock fixture (tools.run_tool), the receipt (the calibrated
    band inside horizons + the calibration note) is present before and gone after."""
    raw = run_tool("chartlibrary_cohort_analyze",
                   {"symbol": "NVDA", "date": "2026-05-22", "timeframe": "1d"})
    assert ra.receipt_keys_present(raw), "fixture should carry a receipt"
    stripped = ra.strip_receipt(raw)
    assert not ra.receipt_keys_present(stripped), "strip should remove the fixture receipt"
    # the distribution itself is untouched
    a, b = json.loads(raw), json.loads(stripped)
    assert b["horizons"]["5d"]["p50"] == a["horizons"]["5d"]["p50"]
    assert b["horizons"]["5d"]["pct_positive"] == a["horizons"]["5d"]["pct_positive"]
    # only the band (+ note) left the horizon
    assert "calibrated_80_band" in a["horizons"]["5d"]
    assert "calibrated_80_band" not in b["horizons"]["5d"]


def test_strip_is_noop_on_non_receipt_results():
    """Search / rival results carry no receipt → strip is a byte-identical no-op."""
    for name in ("chartlibrary_search", "technical_analysis", "news_catalysts"):
        raw = run_tool(name, {"symbol": "AAPL", "query": "AAPL 2026-05-22"})
        assert ra.strip_receipt(raw) == raw, f"strip mutated {name}"


def test_strip_tolerates_malformed_input():
    assert ra.strip_receipt("not json at all") == "not json at all"
    assert ra.strip_receipt("[1, 2, 3]") == "[1, 2, 3]"  # non-dict JSON passes through


def test_arm_runner_only_strips_cohort_analyze():
    """make_arm_runner(strip=True) strips cohort_analyze and leaves every other
    tool identical to the strip=False arm."""
    on = ra.make_arm_runner(run_tool, strip=False)
    off = ra.make_arm_runner(run_tool, strip=True)
    args = {"symbol": "NVDA", "date": "2026-05-22", "timeframe": "1d"}
    # cohort_analyze differs (receipt stripped on the off arm)
    assert ra.receipt_keys_present(on("chartlibrary_cohort_analyze", args))
    assert not ra.receipt_keys_present(off("chartlibrary_cohort_analyze", args))
    # every other tool is byte-identical between arms
    for name in ("chartlibrary_search", "technical_analysis", "fundamentals"):
        a = {"symbol": "NVDA", "query": "NVDA 2026-05-22"}
        assert on(name, a) == off(name, a), f"arms differ on non-receipt tool {name}"


# ---------------------------------------------------------------------------
# 2. mock e2e: both-arm scores + report + markdown
# ---------------------------------------------------------------------------

def test_mock_e2e_produces_both_arm_scores_and_report(tmp_path):
    from harness import MockModel
    subjects = SUBJECTS[:6]
    backend = MockModel("v2")
    judges = [("haiku", lambda r, rcv, nor, rng: ra.mock_judge(r, rcv, nor, rng)),
              ("opus", lambda r, rcv, nor, rng: ra.mock_judge(r, rcv, nor, rng))]

    records = ra.run_pair(subjects, backend, run_tool)
    assert len(records) == len(subjects)
    for rec in records:
        # both arms produced a final answer
        assert rec["receipt_final"], "receipt arm produced no answer"
        assert rec["no_receipt_final"], "no-receipt arm produced no answer"

    records = ra.judge_records(records, judges)
    rep = ra.report_multi(records, ["haiku", "opus"])
    # both judges scored, overall + all 8 dimensions present
    for lbl in ("haiku", "opus"):
        pj = rep["per_judge"][lbl]
        assert pj["n"] == len(subjects)
        assert "receipt_win_rate_decided" in pj["overall"]
        assert set(pj["per_dimension"]) == set(ra.DIMENSION_KEYS)
        assert len(ra.DIMENSION_KEYS) == 8, "expected 6 prior + 2 new dimensions"
    assert rep["judge_agreement"] is not None

    # markdown writes and contains the pre-registered header + a dimension row
    md = tmp_path / "out.md"
    ra.write_markdown(str(md), {
        "meta": {"mode": "mock", "synthetic": True,
                 "orchestrator_model": "claude-sonnet-4-6", "judges": ["haiku", "opus"],
                 "n_subjects": len(subjects), "n_dates": ra.distinct_dates(subjects),
                 "timestamp": "2026-06-10T00:00:00Z"},
        "report": rep})
    text = md.read_text(encoding="utf-8")
    assert "Pre-registered predictions" in text
    assert "weak_comp_set_flagging" in text
    assert "SYNTHETIC" in text  # mock run is flagged in the markdown


def test_two_new_dimensions_present():
    assert "confidence_qualification" in ra.DIMENSION_KEYS
    assert "weak_comp_set_flagging" in ra.DIMENSION_KEYS
    # the six prior dimensions are still there
    for d in ("groundedness", "specificity", "calibration_honesty",
              "usefulness", "no_overclaim", "internal_consistency"):
        assert d in ra.DIMENSION_KEYS


# ---------------------------------------------------------------------------
# 3. subject list date diversity
# ---------------------------------------------------------------------------

def test_subjects_span_enough_distinct_dates():
    assert len(SUBJECTS) >= 50, f"expected ~50 subjects, got {len(SUBJECTS)}"
    n_dates = distinct_dates()
    assert n_dates >= 40, f"spec requires >=40 distinct anchor dates, got {n_dates}"


def test_subjects_reuse_prior_plus_fresh():
    sources = {s["source"] for s in SUBJECTS}
    assert "reused" in sources and "fresh" in sources
    # all anchors are settled past dates (live endpoint has EOD embeddings)
    assert all(s["date"] < "2026-05-30" for s in SUBJECTS)
    # ids unique
    ids = [s["id"] for s in SUBJECTS]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# 4. judge-blinding: arm labels never reach the judge prompt
# ---------------------------------------------------------------------------

class _SpyClient:
    """Captures the prompt sent to the judge so we can assert blinding."""
    def __init__(self):
        self.seen_prompts = []
        self.messages = self

    def create(self, model, max_tokens, messages):
        self.seen_prompts.append(messages[0]["content"])

        class _Blk:
            type = "text"
            text = ('{"dimensions": {"groundedness": "A", "specificity": "tie", '
                    '"calibration_honesty": "A", "usefulness": "tie", '
                    '"no_overclaim": "tie", "internal_consistency": "tie", '
                    '"confidence_qualification": "A", "weak_comp_set_flagging": "A"}, '
                    '"overall": "A", "reason": "x"}')

        class _Resp:
            content = [_Blk()]
        return _Resp()


def test_live_judge_prompt_is_blind():
    import random
    spy = _SpyClient()
    rec = {"id": "x", "prompt": "Is NVDA extended?"}
    ans_receipt = "RECEIPT ARM: calibrated 80% band -4% to +5%, 80.8% coverage."
    ans_noreceipt = "NO-RECEIPT ARM: roughly flat to up, wide range."
    for seed in range(20):  # exercise both swap orders deterministically
        rng = random.Random(seed)
        ra.live_judge(spy, "judge-model", rec, ans_receipt, ans_noreceipt, rng)

    assert len(spy.seen_prompts) == 20
    for prompt in spy.seen_prompts:
        # our internal arm-label tokens must NEVER appear in the judge prompt —
        # that would tell the judge which arm is which.
        assert "no_receipt" not in prompt
        assert "receipt_final" not in prompt and "no_receipt_final" not in prompt
        # the judge only ever sees neutral A / B framing + the raw answers
        assert "ANSWER A:" in prompt and "ANSWER B:" in prompt
        # and the judge instruction explicitly says it does not know how they differ
        assert "do NOT know how they differ" in prompt


def test_live_judge_maps_blind_pick_back_to_arm():
    """With a spy that always picks 'A', the mapped winner depends only on the
    swap — proving the de-blinding is the swap inverse, not a label leak."""
    import random
    spy = _SpyClient()
    rec = {"id": "x", "prompt": "q"}
    receipts = 0
    for seed in range(40):
        rng = random.Random(seed)
        # peek the swap the judge will use: live_judge draws rng first
        r2 = random.Random(seed)
        swap = r2.random() < 0.5
        out = ra.live_judge(spy, "m", rec, "RCV", "NOR", rng)
        # judge always says 'A'; A is receipt iff not swap
        expected = "no_receipt" if swap else "receipt"
        assert out["overall"] == expected
        receipts += out["overall"] == "receipt"
    assert 0 < receipts < 40, "swap should produce a mix of mapped winners"


def test_verdict_mapping():
    # swap=False: A=receipt, B=no_receipt
    assert ra._verdict_from_pick("A", False) == "receipt"
    assert ra._verdict_from_pick("B", False) == "no_receipt"
    assert ra._verdict_from_pick("tie", False) == "tie"
    # swap=True: A=no_receipt, B=receipt
    assert ra._verdict_from_pick("A", True) == "no_receipt"
    assert ra._verdict_from_pick("B", True) == "receipt"
