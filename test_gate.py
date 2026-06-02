"""Self-check for gate.py - run: python test_gate.py  (plain asserts, no pytest).

A gate you cannot trust is worse than no gate, so these pin the pass/fail edges:
threshold breaches, the weakest-judge rule, both A/B shapes, the boundary, and the
synthetic honesty guard.
"""
from gate import evaluate_gate

LIVE = {"meta": {"mode": "live", "synthetic": False}}
MOCK = {"meta": {"mode": "mock", "synthetic": True}}


def _sel(recall, fpr):
    return {"selection": {"7": {"recall_overall": recall, "recall_overall_n": "x/y",
                                "fpr_overall": fpr, "fpr_overall_n": "x/y"}}}


def _ab_multi(**judges):
    return {"ab": {"report": {
        "labels": list(judges),
        "per_judge": {k: {"with_win_rate_decided": v} for k, v in judges.items()}}}}


def t(name, res, expect_pass, **kw):
    v = evaluate_gate(res, **kw)
    assert v["passed"] is expect_pass, f"{name}: expected pass={expect_pass}, got {v}"
    print(f"ok  {name}")


# live full pass (the real-receipt shape: recall .933, fpr 0, lift .867)
t("live full pass", {**LIVE, **_sel(0.933, 0.0), **_ab_multi(haiku=0.867)}, True)
# recall below the floor -> fail
t("recall too low", {**LIVE, **_sel(0.70, 0.0), **_ab_multi(haiku=0.9)}, False)
# over-fire above the ceiling -> fail
t("over-fire too high", {**LIVE, **_sel(0.95, 0.30), **_ab_multi(haiku=0.9)}, False)
# weakest judge governs: one judge under the lift floor -> fail
t("one weak judge", {**LIVE, **_sel(0.95, 0.0), **_ab_multi(haiku=0.9, opus=0.50)}, False)
# legacy single-judge A/B shape still parses + passes
t("legacy single-judge", {**LIVE, **_sel(0.9, 0.0),
                          "ab": {"report": {"with_win_rate_decided": 0.8}}}, True)
# boundary is inclusive (>= / <=): recall .80, fpr .15, lift .60 -> pass
t("boundary inclusive", {**LIVE, **_sel(0.80, 0.15), **_ab_multi(haiku=0.60)}, True)
# mock -> plumbing passes even with bad synthetic numbers
t("mock plumbing ok", {**MOCK, **_sel(0.10, 0.90), **_ab_multi(mock=0.10)}, True)
# mock + require_live -> hard fail (never certify synthetic)
t("mock blocked by require-live",
  {**MOCK, **_sel(0.99, 0.0), **_ab_multi(mock=0.99)}, False, require_live=True)
# live but a metric is null (malformed) -> fail
t("malformed null metric",
  {**LIVE, "selection": {"7": {"recall_overall": None, "fpr_overall": None}},
   **_ab_multi(haiku=0.9)}, False)
# selection-only live run passes (lift not gated, only what ran)
t("selection only", {**LIVE, **_sel(0.9, 0.0)}, True)
# nothing gateable -> fail
t("empty live", {**LIVE}, False)

print("\nall gate self-checks passed")
