"""
Phase 1: the REAL chartlibrary tool runner.

Phase 0 fed the orchestrator deterministic FIXTURES for every tool. That is
faithful for the *selection* question (does the agent pick the tool — which
depends on the tool DESCRIPTION, not its output) but NOT for the *answer-lift*
question. The A/B asks: when the agent gets a REAL cohort distribution instead of
representative numbers, does the with-vs-without answer lift still hold?

This module swaps ONLY the two chartlibrary tools (search + cohort_analyze) for
live calls to the production endpoint. Every rival / padding tool keeps its
Phase 0 fixture — they have no real backend here, and leaving them fixed means
the A/B "without" arm is byte-identical to before and the "with" arm differs in
exactly one thing: the chartlibrary node is now real.

Calls are anonymous (no API key) over the same public path scripts/verify_introspect
uses. Gentle on prod: a known-good date fallback so we don't spray 404s, a small
retry chain for weekend/holiday anchors, response trimming via fields=, and a
short per-call timeout. stdlib only (urllib) so the reference repo gains no dep.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from tools import CHARTLIBRARY_INTROSPECT_NAME, _parse_symbol, run_tool

DEFAULT_BASE_URL = os.environ.get("CHARTLIBRARY_BASE_URL", "https://chartlibrary.io").rstrip("/")
# A settled trading day known to carry V5 embeddings (matches the default in
# scripts/verify_introspect). The model usually asks about "today"; today's EOD
# embedding does not exist until the overnight ingest runs, so we anchor "today's
# setup" to the most recent settled bar — faithful to how the live product reads
# "the current pattern".
DEFAULT_ANCHOR_DATE = "2026-05-22"
DEFAULT_TIMEFRAME = "1d"
_SETTLED_BEFORE = "2026-05-30"  # EOD embeddings for >= this date may not exist yet
# NB: do NOT send fields= — it is an allowlist that drops cohort_id (cohort_id is
# not an allowlistable key), and search needs the handle. We take the full faithful
# payload (incl. calibration) and bound its size client-side with _trim.
_TIMEOUT = 60


def _post(base_url: str, path: str, body: dict, timeout: int = _TIMEOUT):
    """POST JSON; return (status_code, parsed_json_or_error_dict). status 0 = transport error."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json",
                 "User-Agent": "chartlibrary-agent-crew/phase1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"error": raw[:300]}
    except Exception as e:  # transport / timeout / DNS
        return 0, {"error": f"{type(e).__name__}: {e}"}


def _candidate_date(model_date) -> str:
    """Honor a settled past date the model supplied; otherwise the known-good default."""
    d = (model_date or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and d < _SETTLED_BEFORE:
        return d
    return DEFAULT_ANCHOR_DATE


def _analyze_real(base_url: str, symbol: str, date, timeframe, stats: dict) -> dict:
    """Call /api/v1/cohort_analyze with a small retry chain. Returns the real
    response dict on success, or {"error": ...} on failure (honest degrade)."""
    tf = timeframe or DEFAULT_TIMEFRAME
    tried: list[str] = []

    def attempt(d: str):
        tried.append(d)
        body = {"anchor": {"symbol": symbol, "date": d, "timeframe": tf},
                "cohort_size": 500, "horizons": [1, 5, 10], "embedding": "v5"}
        return _post(base_url, "/api/v1/cohort_analyze", body)

    cand = _candidate_date(date)
    status, payload = attempt(cand)

    # weekend/holiday -> 422 carrying a previous_trading_day hint
    if status == 422 and isinstance(payload, dict):
        detail = payload.get("detail", payload)
        hint = detail.get("previous_trading_day") if isinstance(detail, dict) else None
        if hint and hint not in tried:
            status, payload = attempt(hint)

    # anything still wrong -> fall back to the known-good default once
    if status != 200 and DEFAULT_ANCHOR_DATE not in tried:
        status, payload = attempt(DEFAULT_ANCHOR_DATE)

    # success = a real cohort came back (cohort_id may be absent for tiny cohorts,
    # so key on the distribution / member count instead)
    if status == 200 and isinstance(payload, dict) and (
            payload.get("cohort_id") or payload.get("outcome_distribution")
            or payload.get("cohort_size_actual")):
        stats["ok"] += 1
        return payload

    stats["error"] += 1
    err = payload.get("detail") if isinstance(payload, dict) else payload
    return {"error": f"cohort_analyze unavailable (HTTP {status})", "detail": err, "_real": True}


def _trim(payload: dict, limit: int = 8000) -> str:
    """Generic size backstop for small payloads (introspect)."""
    s = json.dumps(payload)
    if len(s) <= limit:
        return s
    return json.dumps({"_trimmed": True, **{k: payload[k] for k in
                       ("error", "detail", "_real", "interpretation",
                        "comparison", "full_cohort_stats", "subset_stats")
                       if k in payload}})


# Every small, decision-relevant field is kept verbatim — including `calibration`
# (the calibrated 80% bands, our headline asset). Only `feature_importance` is
# compacted: the live payload returns ALL features with CIs per horizon (~32KB,
# 96% of the response); the agent only needs the top separators.
_CA_KEEP = ("anchor", "cohort_id", "cohort_size_actual", "outcome_distribution",
            "regime_stratification", "risk_profile", "expected_move", "calibration",
            "cohort_tightness_score", "cohort_score", "combined_conviction",
            "warnings", "error", "_real", "_degraded")


def _slim_cohort_analyze(payload: dict, top_n: int = 5) -> str:
    if not isinstance(payload, dict):
        return json.dumps(payload)
    out = {k: payload[k] for k in _CA_KEEP if k in payload}
    fi = payload.get("feature_importance")
    if isinstance(fi, dict):
        slim = {}
        for hz, feats in fi.items():
            if isinstance(feats, list):
                top = sorted((f for f in feats if isinstance(f, dict)),
                             key=lambda f: abs(f.get("importance") or 0), reverse=True)[:top_n]
                slim[hz] = [{"feature": f.get("feature"),
                             "importance": round(f.get("importance") or 0, 3),
                             "direction": f.get("direction")} for f in top]
        if slim:
            out["feature_importance"] = slim
    return json.dumps(out)


def make_real_runner(base_url: str = DEFAULT_BASE_URL):
    """Return a (name, args) -> json_str runner. The two chartlibrary tools hit
    the live endpoint; every other tool delegates to the Phase 0 fixture.

    The returned callable carries a `.stats` dict so the caller can report how
    many real calls succeeded / degraded / errored — transparency that lets us
    read the A/B honestly (a low lift from integration failures != a low lift
    from genuine no-value)."""
    stats = {"ok": 0, "error": 0, "degraded": 0, "fixture": 0}

    def runner(name: str, args: dict) -> str:
        args = args or {}

        if name == "chartlibrary_cohort_analyze":
            sym = (args.get("symbol") or _parse_symbol(str(args)) or "").upper()
            if not sym:
                stats["degraded"] += 1
                return json.dumps({
                    "error": "chartlibrary needs a specific ticker symbol (and ideally a "
                             "date) to find historical analogs; this question names none.",
                    "_real": True, "_degraded": "no_symbol"})
            payload = _analyze_real(base_url, sym, args.get("date"), args.get("timeframe"), stats)
            return _slim_cohort_analyze(payload)

        if name == "chartlibrary_search":
            q = args.get("query", "")
            sym = (_parse_symbol(q) or "").upper()
            if not sym:
                stats["degraded"] += 1
                return json.dumps({
                    "error": "chartlibrary_search needs a 'SYMBOL YYYY-MM-DD' query naming a ticker.",
                    "_real": True, "_degraded": "no_symbol"})
            m = re.search(r"\d{4}-\d{2}-\d{2}", q)
            payload = _analyze_real(base_url, sym, m.group(0) if m else None, "1d", stats)
            if payload.get("error"):
                return _trim(payload)
            return json.dumps({"cohort_id": payload.get("cohort_id"),
                               "anchor": payload.get("anchor"),
                               "n_matches": payload.get("cohort_size_actual"),
                               "survivorship": "ok", "_real": True})

        if name == CHARTLIBRARY_INTROSPECT_NAME:
            cid = args.get("cohort_id")
            if not cid:
                stats["degraded"] += 1
                return json.dumps({"error": "introspect needs a cohort_id from a prior call.",
                                   "_real": True})
            where = args.get("where")
            body = {"cohort_id": cid, "horizon": 5}
            # the harness passes a string token (e.g. 'relative_volume_top_quartile');
            # the real endpoint wants a structured filter — bridge the common one.
            if isinstance(where, dict):
                body["where"] = where
            elif isinstance(where, str) and "relative_volume" in where:
                body["where"] = {"technical.relative_volume": {"min": 1.0}}
            else:
                body["where"] = {}
            status, payload = _post(base_url, "/api/v1/cohort_introspect", body)
            if status == 200 and isinstance(payload, dict) and not payload.get("error"):
                stats["ok"] += 1
                return _trim(payload)
            stats["error"] += 1
            return json.dumps({"error": f"introspect unavailable (HTTP {status})",
                               "detail": payload, "_real": True})

        # every rival / padding tool keeps its Phase 0 fixture
        stats["fixture"] += 1
        return run_tool(name, args)

    runner.stats = stats
    return runner
