"""
OpenAI Agents SDK port of the crew (Phase 2).

Same crew, a different orchestrator - and a different model VENDOR. crew.py is a
framework-free native loop; ports/langgraph_crew.py reuses its nodes inside a
LangGraph StateGraph; this file drives the SAME Chart Library node from OpenAI's
Agents SDK. It is the second proof that the node is plug-and-play: here an OpenAI
model does the orchestration and reaches for our anonymous public endpoint.

Design - idiomatic to this SDK, which is MODEL-driven (not graph-structural like
LangGraph):
  * the Chart Library node is exposed as two `@function_tool`s that call the EXACT
    real runner from `real_tools.make_real_runner()` (the live /api/v1/cohort_analyze
    over stdlib urllib) - byte-for-byte the node the eval measured;
  * one orchestrator `Agent` carries the VALIDATED v2 USE-WHEN / boundary language +
    the provenance mandate + "surface the distribution, never a single forecast";
  * `Runner.run_sync` lets the OpenAI model decide WHEN to reach for the node.

Two modes:
  * OFFLINE (default, FREE): reuses crew.py's deterministic canned path verbatim
    (`crew.run_crew(..., live=False)`) so you get the same plumbing proof with no key
    and no spend; the SDK `Agent` is still CONSTRUCTED, so import + wiring is verified.
    NOTE: offline deliberately BYPASSES the SDK's model-driven `Runner` (which needs a
    key). The free check proves wiring + the real node, not the OpenAI agent loop -
    that loop is exercised only by a (paid) --live run.
  * --live (PAID, needs OPENAI_API_KEY): real OpenAI orchestration + the real Chart
    Library node. This is the cross-vendor receipt.

Run:
    python ports/openai_agents_crew.py "what usually happens to NVDA after a breakout?"
        # OFFLINE: canned crew + node wiring constructed; no key, no spend (free).
    python ports/openai_agents_crew.py "Is NVDA extended, and what next?" --live --model gpt-4.1
        # real OpenAI Agents Runner + the real Chart Library node; needs OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Reuse the shared, validated crew from the repo root (this file lives in ports/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crew  # noqa: E402  (the framework-free crew - we reuse its canned path verbatim)
from real_tools import make_real_runner  # noqa: E402  (the real Chart Library node)

DEFAULT_OPENAI_MODEL = "gpt-4.1"  # overridable with --model; only used on a --live run


# The orchestrator's instructions: the SAME validated v2 boundary + provenance mandate
# the eval measured, phrased for a single model that plays the qualitative roles itself
# and reaches for the Chart Library tools for the historical-base-rate lane.
_LEAD_INSTRUCTIONS = (
    "You are the lead of a markets research crew answering a markets question.\n"
    "TOOLS: you have a Chart Library node - `chartlibrary_cohort_analyze` and "
    "`chartlibrary_search` - returning the REAL, calibrated historical base rates: what "
    "ACTUALLY HAPPENED NEXT after the most similar historical setups to a (ticker, date), "
    "from 25M+ analogs with calibrated 80% bands. USE THEM when the question is about "
    "historical frequency, 'what usually happens next', odds, or the expected range for a "
    "NAMED ticker. Do NOT use them to read the CURRENT chart (RSI/MACD/support-resistance) "
    "- that is live technical analysis, a different lane.\n"
    "OTHER PERSPECTIVES: play the technical / fundamental / news / macro / risk roles "
    "yourself from general knowledge, and say plainly those reads are qualitative / "
    "illustrative (no live feed) - the contrast is honest and makes the sourced numbers pop.\n"
    "PROVENANCE: attach provenance to every historical number, e.g. \"per Chart Library's N "
    "analogs (calibrated 80% band)\", so the numbers are trusted, not flagged as guessed.\n"
    "OUTPUT: one concise brief. Surface the DISTRIBUTION / range; do NOT issue a single "
    "directional price forecast. If the question names no ticker, say the node needs a symbol."
)


def build_openai_orchestrator(model):
    """Construct the orchestrator Agent + the real Chart Library runner. Importing +
    constructing needs NO key (only Runner.run_sync does), so this doubles as the
    free wiring check. Returns (agent, runner); runner.stats reports node usage."""
    from agents import Agent, function_tool

    runner = make_real_runner()

    @function_tool
    def chartlibrary_cohort_analyze(symbol: str, date: str = "", timeframe: str = "1d") -> str:
        """Calibrated HISTORICAL BASE RATES for a ticker: what ACTUALLY HAPPENED NEXT after
        the most similar historical setups (25M+ analogs, calibrated 80% bands). Pass a
        ticker symbol (and optional YYYY-MM-DD date). USE for 'what usually happens next',
        odds, or the expected range; do NOT use to read the current chart (RSI/MACD)."""
        return runner("chartlibrary_cohort_analyze",
                      {"symbol": symbol, "date": date or None, "timeframe": timeframe or "1d"})

    @function_tool
    def chartlibrary_search(query: str) -> str:
        """Find a Chart Library cohort handle for a 'SYMBOL YYYY-MM-DD' style query that
        names a ticker. Returns the cohort id + match count to anchor a base-rate read."""
        return runner("chartlibrary_search", {"query": query})

    agent = Agent(name="market-research-lead", instructions=_LEAD_INSTRUCTIONS,
                  tools=[chartlibrary_cohort_analyze, chartlibrary_search], model=model)
    return agent, runner


def run_crew_openai(question, live, model):
    # Always construct the SDK agent first - even offline - so a free run verifies the
    # import + wiring (the function_tool schemas, the Agent) without spending anything.
    agent, runner = build_openai_orchestrator(model)

    if not live:
        # FREE deterministic plumbing proof: reuse crew.py's canned path verbatim. (The
        # SDK Runner is model-driven and needs a key, so offline bypasses it by design.)
        r = crew.run_crew(question, False, model)
        r["orchestrator"] = "openai-agents (offline canned; SDK agent constructed, Runner bypassed)"
        return r

    from agents import Runner
    result = Runner.run_sync(agent, question)
    cc = dict(runner.stats)
    fired = (cc.get("ok", 0) + cc.get("error", 0) + cc.get("degraded", 0)) > 0
    return {"question": question, "live": True, "model": model, "orchestrator": "openai-agents",
            "plan": {"consult": ["(model-internal)"], "reason": "OpenAI Agents Runner chose the tools"},
            "memos": {}, "chartlibrary_fired": fired, "chartlibrary_calls": cc,
            "brief": str(getattr(result, "final_output", result))}


def _print_human(result):
    print("[orchestrator: OpenAI Agents SDK - Runner + Chart Library function_tools]\n")
    if not result["live"]:
        crew._print_human(result)  # offline result has crew.py's full structured shape
        return
    print("=== chart-library-agent-crew  [LIVE - OpenAI Agents SDK] ===")
    print(f"Q: {result['question']}\n")
    cc = result["chartlibrary_calls"]
    print(f"[chartlibrary node fired: {result['chartlibrary_fired']}  "
          f"real-calls ok={cc.get('ok',0)} err={cc.get('error',0)} degraded={cc.get('degraded',0)}]\n")
    print("-- BRIEF --")
    print(result["brief"])
    print("\n(receipt: eval/ shows a blind judge prefers the WITH-node answer "
          "~0.80-0.87 on base-rate questions - see README.)")


def main():
    # Windows consoles default to cp1252; live model output (em dashes, arrows) would
    # crash print(). Make stdout UTF-8 and never-fail (same fix as crew.py).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="OpenAI Agents SDK port of the market-research crew (same Chart Library node).")
    ap.add_argument("question", help="a markets question")
    ap.add_argument("--live", action="store_true",
                    help="real OpenAI Agents Runner + real Chart Library node (needs OPENAI_API_KEY; spends)")
    ap.add_argument("--model", default=DEFAULT_OPENAI_MODEL,
                    help=f"orchestrator model (default {DEFAULT_OPENAI_MODEL})")
    ap.add_argument("--json", action="store_true", help="print the structured result as JSON")
    args = ap.parse_args()

    if args.live and not os.environ.get("OPENAI_API_KEY"):
        print("--live needs OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(2)

    result = run_crew_openai(args.question, args.live, args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
