"""
Tool definitions for the Phase 0 selection harness.

Two chartlibrary tools (search + cohort_analyze) in two description arms:
  * v1 = the REAL production descriptions, copied (lightly trimmed of internal
    legacy-alias lines) from chart-library/mcp_server.py. This is what an agent
    actually sees today.
  * v2 = an IMPROVED arm: explicit Purpose + "use this when" activation criteria
    + a hard negative boundary against the technical-analysis rival, per the 2026
    "MCP tool descriptions are smelly" finding (Purpose + when-to-use ~= +5.85pp
    task success). We A/B v1 vs v2 to see if description quality changes selection.

Plus rival specialist tools so the orchestrator has real competition — most
importantly `technical_analysis`, the collision case our node must be distinct
from.

Tool OUTPUTS are deterministic fixtures so the harness runs offline, free, and
without touching production. Selection behaviour (does the orchestrator pick a
tool) depends on the *descriptions*, not the live data — so fixtures are faithful
for the selection question. The answer-quality A/B uses representative numbers;
swap in real prod data in Phase 1 via run.py --real-chartlibrary.
"""

from __future__ import annotations

import hashlib
import json
import re

# ---------------------------------------------------------------------------
# chartlibrary node — v1 (real production descriptions)
# ---------------------------------------------------------------------------

CHARTLIBRARY_V1 = {
    "chartlibrary_search": {
        "name": "chartlibrary_search",
        "description": (
            "Entry point: find similar historical patterns for a ticker+date and get a "
            "cohort handle. Returns {status, data: {cohort_id, anchor, n_matches, "
            "survivorship}, meta}. The cohort_id can be passed to chartlibrary_cohort_analyze "
            "to chain operations (sub-second, no kNN re-run).\n\n"
            "Args:\n"
            "  query: 'SYMBOL YYYY-MM-DD' (optional ' timeframe' suffix, e.g. 'NVDA 2024-06-18 rth_5d')\n"
            "  top_k: Cohort size to establish (10-2000, default 500)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "'SYMBOL YYYY-MM-DD' (optional ' timeframe' suffix)"},
                "top_k": {"type": "integer", "description": "Cohort size 10-2000 (default 500)"},
            },
            "required": ["query"],
        },
    },
    "chartlibrary_cohort_analyze": {
        "name": "chartlibrary_cohort_analyze",
        "description": (
            "Layer 3 cohort intelligence - V5 retrieval + Layer 2 metadata. Given a "
            "(symbol, date, timeframe) anchor, returns:\n"
            "  - outcome distribution per horizon (1d / 5d / 10d default)\n"
            "  - per-feature importance - which metadata features separated winners from "
            "losers within this specific cohort\n"
            "  - regime stratification - outcomes sliced by vol regime\n"
            "  - risk profile - drawdown / runup percentiles\n"
            "  - cohort tightness score\n\n"
            "Empirical-distribution analysis. Does NOT predict a single point return - "
            "surfaces what historical analogs did and which features mattered.\n\n"
            "Args:\n"
            "  symbol: Ticker (e.g. 'NVDA')\n"
            "  date: Anchor date, ISO YYYY-MM-DD\n"
            "  timeframe: One of 5m / 15m / 30m / 1h / 1d (default 1h)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "date": {"type": "string", "description": "ISO YYYY-MM-DD"},
                "timeframe": {"type": "string", "description": "5m/15m/30m/1h/1d (default 1h)"},
            },
            "required": ["symbol", "date"],
        },
    },
}

# ---------------------------------------------------------------------------
# chartlibrary node — v2 (improved: Purpose + use-when + negative boundary vs TA)
# ---------------------------------------------------------------------------

CHARTLIBRARY_V2 = {
    "chartlibrary_search": {
        "name": "chartlibrary_search",
        "description": (
            "PURPOSE: Find the cohort of the most similar historical chart setups to a "
            "(ticker, date) and return a handle for base-rate analysis.\n"
            "USE THIS WHEN you want to ground a question in historical precedent - it is "
            "the first step before asking 'what happened next' to setups like this one. "
            "Returns a cohort_id to pass to chartlibrary_cohort_analyze.\n"
            "Args: query='SYMBOL YYYY-MM-DD'; top_k (default 500)."
        ),
        "input_schema": CHARTLIBRARY_V1["chartlibrary_search"]["input_schema"],
    },
    "chartlibrary_cohort_analyze": {
        "name": "chartlibrary_cohort_analyze",
        "description": (
            "PURPOSE: Return the calibrated distribution of what ACTUALLY HAPPENED NEXT "
            "after the most similar historical chart setups (25M+ real analogs across 19K "
            "symbols and 10 years). Empirical base rates, not a forecast.\n"
            "USE THIS WHEN the question is about historical frequency or expected range: "
            "'how often does this follow through', 'what's the base rate after a setup like "
            "this', 'what's the typical 5-day range', 'how did similar breakouts / gaps / "
            "pullbacks resolve historically'. Returns the forward-return distribution "
            "(p10/p50/p90), split-conformal CALIBRATED 80% bands (validated ~80% coverage), "
            "per-feature importance, and a cohort tightness score.\n"
            "Do NOT use this to read the CURRENT chart. For live RSI / MACD / moving "
            "averages / support-resistance on today's chart, use `technical_analysis`. "
            "Rule of thumb: technical_analysis = what the chart looks like NOW; "
            "chartlibrary = what historically happened NEXT to charts that looked like this."
        ),
        "input_schema": CHARTLIBRARY_V1["chartlibrary_cohort_analyze"]["input_schema"],
    },
}

# ---------------------------------------------------------------------------
# Rival specialists (real competition for the orchestrator's attention)
# ---------------------------------------------------------------------------

_SYM_SCHEMA = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}

RIVALS = {
    "technical_analysis": {
        "name": "technical_analysis",
        "description": (
            "Compute LIVE technical indicators on the CURRENT chart for a symbol: RSI(14), "
            "MACD, moving averages (50/200), Bollinger bands, ATR, and current "
            "support/resistance levels. Use to describe the present technical state or "
            "setup of a stock right now (e.g. 'is it overbought', 'is it above its 50-day', "
            "'where is resistance')."
        ),
        "input_schema": _SYM_SCHEMA,
    },
    "news_catalysts": {
        "name": "news_catalysts",
        "description": (
            "Fetch recent news headlines and catalysts for a symbol plus a short "
            "narrative-change / sentiment read. Use for 'any news', catalysts, or why a "
            "stock is moving on headlines today."
        ),
        "input_schema": _SYM_SCHEMA,
    },
    "fundamentals": {
        "name": "fundamentals",
        "description": (
            "Company fundamentals: valuation (P/E, P/S, EV/EBITDA), revenue and earnings "
            "growth, margins, and balance-sheet health. Use for valuation and "
            "financial-health questions."
        ),
        "input_schema": _SYM_SCHEMA,
    },
    "macro_regime": {
        "name": "macro_regime",
        "description": (
            "Current market / macro regime: VIX level and term structure, trend regime, "
            "rates and credit backdrop, sector posture. Use for top-down market-context "
            "questions (no specific ticker required)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "risk_position": {
        "name": "risk_position",
        "description": (
            "Position-risk helper: given a stop distance and account size, suggest a "
            "position size and R-multiple and summarize downside. Use for sizing / stop "
            "questions."
        ),
        "input_schema": _SYM_SCHEMA,
    },
}

# Padding tools to reach a ~13-tool loadout for the count-stress condition.
PADDING = {
    name: {"name": name, "description": desc, "input_schema": _SYM_SCHEMA}
    for name, desc in {
        "options_flow": "Unusual options activity and implied-volatility snapshot for a symbol.",
        "insider_activity": "Recent insider buys/sells and Form 4 filings for a symbol.",
        "analyst_ratings": "Sell-side analyst ratings, price targets, and recent revisions for a symbol.",
        "sector_rotation": "Relative-strength and money-flow across sectors and the symbol's sector.",
        "earnings_calendar": "Next earnings date and recent earnings-reaction history for a symbol.",
        "economic_calendar": "Upcoming macro releases (CPI, FOMC, jobs) over the next two weeks.",
    }.items()
}

CHARTLIBRARY_TOOL_NAMES = {"chartlibrary_search", "chartlibrary_cohort_analyze"}

# Second-hop introspection tool (Phase 0.5 chaining probe). Kept OUT of
# CHARTLIBRARY_TOOL_NAMES so the existing selection/over-fire metrics are
# unchanged; the probe scores chaining separately.
CHARTLIBRARY_INTROSPECT_NAME = "chartlibrary_cohort_introspect"

CHARTLIBRARY_INTROSPECT_TOOL = {
    "name": CHARTLIBRARY_INTROSPECT_NAME,
    "description": (
        "PURPOSE: Second-hop drill-down on an EXISTING chartlibrary cohort. Given a "
        "cohort_id from a prior chartlibrary_search or chartlibrary_cohort_analyze call, "
        "reveal WHICH features (entry volume, days-since-ATH, sector relative strength, "
        "realized vol, regime) separated the winners from the losers INSIDE that cohort, "
        "and optionally filter to a sub-cohort to get a sharper, conditional distribution.\n"
        "USE THIS WHEN you already have a cohort and want to understand *why* its outcome "
        "distribution looks the way it does, or to condition on a sub-segment the user "
        "cares about (their volume / their regime / their setup variant). It turns a base "
        "rate into a conditional base rate, and is the natural follow-up when a cohort's "
        "range is wide or bimodal.\n"
        "Args: cohort_id (required, from a prior call); where (optional sub-cohort filter, "
        "e.g. 'relative_volume_top_quartile'); split_by (optional feature to stratify by)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cohort_id": {"type": "string", "description": "from a prior chartlibrary call"},
            "where": {"type": "string", "description": "optional sub-cohort filter expression"},
            "split_by": {"type": "string", "description": "optional feature to stratify by"},
        },
        "required": ["cohort_id"],
    },
}


def build_loadout(desc="v2", size=7, introspect=False):
    """Return the list of tool schemas the orchestrator sees.

    size=7  -> 2 chartlibrary + 5 rivals
    size=13 -> + 6 padding tools (count-stress condition)
    introspect=True appends the second-hop chartlibrary_cohort_introspect tool
    (used by the Phase 0.5 chaining probe).
    """
    cl = CHARTLIBRARY_V2 if desc == "v2" else CHARTLIBRARY_V1
    tools = dict(cl)
    tools.update(RIVALS)
    if size >= 13:
        tools.update(PADDING)
    schemas = list(tools.values())
    if introspect:
        schemas.append(CHARTLIBRARY_INTROSPECT_TOOL)
    return schemas


# ---------------------------------------------------------------------------
# Deterministic fixture outputs
# ---------------------------------------------------------------------------

def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def _parse_symbol(text: str):
    if not text:
        return None
    m = re.search(r"\b[A-Z]{2,5}\b", text)
    return m.group(0) if m else None


def _cohort_id(symbol: str, date: str) -> str:
    return "coh_" + hashlib.sha256(f"{symbol}|{date}".encode()).hexdigest()[:8]


def _cohort_fixture(symbol: str, date: str = "?", timeframe: str = "1h",
                    nudge: bool = False) -> dict:
    r = _seed(f"{symbol}|{date}")
    med5 = round(((r % 700) / 100.0) - 2.0, 2)        # -2.0 .. +5.0
    spread = round(4 + (r % 900) / 100.0, 1)           # 4.0 .. 13.0
    n = 300 + (r % 400)
    pos = round(0.46 + ((r >> 7) % 22) / 100.0, 2)     # 0.46 .. 0.67
    tight = round(0.40 + ((r >> 11) % 55) / 100.0, 2)
    lo, hi = round(med5 - spread, 1), round(med5 + spread, 1)
    feats = ["relative_volume", "days_since_ath", "sector_rs", "realized_vol"]
    cid = _cohort_id(symbol, date)
    out = {
        "cohort_id": cid,
        "anchor": {"symbol": symbol, "date": date, "timeframe": timeframe},
        "n_analogs": n,
        "horizons": {
            "5d": {"p10": lo, "p50": med5, "p90": hi,
                   "calibrated_80_band": [lo, hi], "pct_positive": pos},
            "10d": {"p50": round(med5 * 1.4, 2)},
        },
        "cohort_tightness": tight,
        "feature_importance": feats[: 1 + (r % 4)],
        "note": "empirical historical distribution; calibrated 80% band ~= 80% coverage; not a forecast",
        "_fixture": True,
    }
    if nudge:
        # Mirrors the #72 design: cohort_analyze's RESPONSE carries a concrete
        # next-hop pointer to invite the introspection second hop.
        out["suggested_introspections"] = [{
            "tool": "chartlibrary_cohort_introspect",
            "args": {"cohort_id": cid, "where": "relative_volume_top_quartile"},
            "why": ("the 5-day outcome is wide and looks bimodal - winners and losers "
                    "split sharply on entry volume; drilling into the sub-cohort tightens "
                    "the conditional range."),
        }]
    return out


def run_tool(name: str, args: dict, nudge: bool = False) -> str:
    """Return a fixture result (JSON string) for a tool call.

    nudge=True makes chartlibrary_cohort_analyze attach a `suggested_introspections`
    pointer to its response (the Phase 0.5 chaining-probe treatment arm).
    """
    args = args or {}
    sym = args.get("symbol") or _parse_symbol(args.get("query", "")) or _parse_symbol(str(args))
    if name == "chartlibrary_search":
        q = args.get("query", "")
        cid = "coh_" + hashlib.sha256(q.encode()).hexdigest()[:8]
        return json.dumps({"cohort_id": cid, "anchor": q,
                           "n_matches": 300 + _seed(q) % 400, "survivorship": "ok",
                           "_fixture": True})
    if name == "chartlibrary_cohort_analyze":
        return json.dumps(_cohort_fixture(sym or "UNKNOWN", args.get("date", "?"),
                                          args.get("timeframe", "1h"), nudge=nudge))
    if name == CHARTLIBRARY_INTROSPECT_NAME:
        r = _seed("intro|" + str(args.get("cohort_id", "")) + "|" + str(args.get("where", "")))
        med = round(((r % 600) / 100.0) - 1.0, 2)      # -1.0 .. +5.0
        return json.dumps({
            "cohort_id": args.get("cohort_id"),
            "where": args.get("where"),
            "split_by": args.get("split_by"),
            "n_sub": 60 + r % 140,
            "separating_features": [
                {"feature": "relative_volume", "winners": "high", "losers": "low", "importance": 0.41},
                {"feature": "days_since_ath", "winners": "few", "losers": "many", "importance": 0.22},
            ],
            "conditional_5d": {"p10": round(med - 3, 1), "p50": med,
                               "p90": round(med + 7, 1),
                               "pct_positive": round(0.55 + (r % 18) / 100.0, 2)},
            "note": "conditional sub-cohort; sharper than the parent base rate",
            "_fixture": True,
        })
    if name == "technical_analysis":
        r = _seed("ta" + (sym or ""))
        return json.dumps({"symbol": sym, "rsi14": 40 + r % 45,
                           "macd": "bullish_cross" if r % 2 else "bearish_cross",
                           "vs_ma50": "above" if r % 3 else "below",
                           "vs_ma200": "above" if r % 5 else "below", "_fixture": True})
    if name == "news_catalysts":
        return json.dumps({"symbol": sym,
                           "headlines": ["(fixture) product update", "(fixture) sell-side note"],
                           "narrative_change_score": round((_seed("nw" + (sym or "")) % 100) / 100, 2),
                           "_fixture": True})
    if name == "fundamentals":
        r = _seed("fn" + (sym or ""))
        return json.dumps({"symbol": sym, "pe": 12 + r % 40, "rev_growth_yoy_pct": r % 30,
                           "gross_margin_pct": 40 + r % 45, "_fixture": True})
    if name == "macro_regime":
        r = _seed("macro")
        return json.dumps({"vix": 14 + r % 18, "vix_term": "contango" if r % 2 else "backwardation",
                           "trend_regime": "risk_on" if r % 3 else "risk_off", "_fixture": True})
    if name == "risk_position":
        return json.dumps({"symbol": sym, "suggested_size_pct": 2, "r_multiple": 2.0,
                           "max_loss_pct": 1.0, "_fixture": True})
    return json.dumps({"tool": name, "args": args, "result": "(fixture stub)", "_fixture": True})
