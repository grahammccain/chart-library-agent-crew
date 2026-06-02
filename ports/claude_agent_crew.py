"""
Claude Agent SDK port of the crew (Phase 2).

The most ON-THESIS port: it hands the Chart Library node to a Claude agent as an
in-process MCP server (`create_sdk_mcp_server` + `@tool`). Chart Library IS an MCP
server in production, so this is the closest a reference example gets to the real
thing - "wire Chart Library into your Claude agent as one MCP tool and watch the
agent reach for it." Same node, same validated boundary + provenance language as
crew.py.

Two modes (mirroring the other ports):
  * OFFLINE (default, FREE): reuses crew.py's deterministic canned path verbatim
    (`crew.run_crew(..., live=False)`); the in-process MCP server + `ClaudeAgentOptions`
    are still CONSTRUCTED, so import + wiring is verified with no key and no runtime.
    NOTE: offline BYPASSES the actual Claude agent loop (it needs a key + the Claude
    Agent SDK runtime). The free check proves wiring + the real node, not the agent run.
  * --live (PAID): a real Claude agent drives the in-process Chart Library MCP. Needs
    ANTHROPIC_API_KEY *and* the Claude Agent SDK runtime to be available.

Run:
    python ports/claude_agent_crew.py "what usually happens to NVDA after a breakout?"
        # OFFLINE: canned crew + MCP wiring constructed; no key, no spend (free).
    python ports/claude_agent_crew.py "Is NVDA extended, and what next?" --live
        # real Claude agent over the in-process Chart Library MCP; needs ANTHROPIC_API_KEY.
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

# Fully-qualified name the SDK assigns an in-process MCP tool: mcp__<server>__<tool>.
_TOOL_FQN = "mcp__chartlibrary__chartlibrary_cohort_analyze"

_LEAD_INSTRUCTIONS = (
    "You are the lead of a markets research crew answering a markets question.\n"
    "TOOL: you have a Chart Library MCP tool (chartlibrary_cohort_analyze) returning the "
    "REAL, calibrated historical base rates - what ACTUALLY HAPPENED NEXT after the most "
    "similar historical setups to a (ticker, date), from 25M+ analogs with calibrated 80% "
    "bands. USE IT when the question is about historical frequency, 'what usually happens "
    "next', odds, or the expected range for a NAMED ticker. Do NOT use it to read the "
    "CURRENT chart (RSI/MACD/support-resistance) - that is live technical analysis.\n"
    "OTHER PERSPECTIVES: play the technical / fundamental / news / macro / risk roles "
    "yourself from general knowledge, and say plainly those reads are qualitative / "
    "illustrative (no live feed).\n"
    "PROVENANCE: attach provenance to every historical number, e.g. \"per Chart Library's N "
    "analogs (calibrated 80% band)\", so the numbers are trusted, not flagged as guessed.\n"
    "OUTPUT: one concise brief. Surface the DISTRIBUTION / range; do NOT issue a single "
    "directional price forecast. If the question names no ticker, say the node needs a symbol."
)


def build_chartlibrary_mcp():
    """Build the in-process Chart Library MCP server (one @tool wrapping the EXACT real
    runner from real_tools). Constructing it needs no key/runtime, so this is also the
    free wiring check. Returns (server, runner); runner.stats reports node usage."""
    from claude_agent_sdk import tool, create_sdk_mcp_server

    runner = make_real_runner()

    @tool("chartlibrary_cohort_analyze",
          "Calibrated HISTORICAL BASE RATES for a ticker: what ACTUALLY HAPPENED NEXT after "
          "the most similar historical setups (25M+ analogs, calibrated 80% bands). USE for "
          "'what usually happens next', odds, or the expected range for a named ticker; do "
          "NOT use to read the current chart (RSI/MACD) - that is live technical analysis.",
          {"symbol": str, "date": str, "timeframe": str})
    async def chartlibrary_cohort_analyze(args):
        out = runner("chartlibrary_cohort_analyze",
                     {"symbol": args.get("symbol"), "date": args.get("date") or None,
                      "timeframe": args.get("timeframe") or "1d"})
        return {"content": [{"type": "text", "text": out}]}

    server = create_sdk_mcp_server(name="chartlibrary", version="1.0.0",
                                   tools=[chartlibrary_cohort_analyze])
    return server, runner


def build_options(model):
    """Wire the in-process Chart Library MCP into ClaudeAgentOptions. Construction only -
    no key or runtime needed - so an offline run verifies the whole wiring for free."""
    from claude_agent_sdk import ClaudeAgentOptions

    server, runner = build_chartlibrary_mcp()
    options = ClaudeAgentOptions(
        mcp_servers={"chartlibrary": server},
        allowed_tools=[_TOOL_FQN],
        system_prompt=_LEAD_INSTRUCTIONS,
        model=model)
    return options, runner


async def _run_live(question, model):
    from claude_agent_sdk import query, AssistantMessage, TextBlock

    options, runner = build_options(model)
    chunks = []
    async for message in query(prompt=question, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    brief = "\n".join(c for c in chunks if c).strip()
    cc = dict(runner.stats)
    fired = (cc.get("ok", 0) + cc.get("error", 0) + cc.get("degraded", 0)) > 0
    return {"question": question, "live": True, "model": model,
            "orchestrator": "claude-agent-sdk (in-process Chart Library MCP)",
            "plan": {"consult": ["(model-internal)"], "reason": "Claude agent chose the MCP tool"},
            "memos": {}, "chartlibrary_fired": fired, "chartlibrary_calls": cc, "brief": brief}


def run_crew_claude_agent(question, live, model):
    if not live:
        # FREE wiring proof: construct the in-process MCP server + options (no key/runtime),
        # then reuse crew.py's deterministic canned path for the brief.
        build_options(crew.DEFAULT_MODEL)
        r = crew.run_crew(question, False, model)
        r["orchestrator"] = ("claude-agent-sdk (offline canned; MCP server + options "
                             "constructed, agent loop not run)")
        return r
    import anyio
    return anyio.run(_run_live, question, model)


def _print_human(result):
    print("[orchestrator: Claude Agent SDK - in-process Chart Library MCP server]\n")
    if not result["live"]:
        crew._print_human(result)  # offline result has crew.py's full structured shape
        return
    print("=== chart-library-agent-crew  [LIVE - Claude Agent SDK] ===")
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
        description="Claude Agent SDK port of the market-research crew (Chart Library as in-process MCP).")
    ap.add_argument("question", help="a markets question")
    ap.add_argument("--live", action="store_true",
                    help="real Claude agent + real Chart Library MCP (needs ANTHROPIC_API_KEY + SDK runtime; spends)")
    ap.add_argument("--model", default=crew.DEFAULT_MODEL,
                    help=f"agent model (default {crew.DEFAULT_MODEL})")
    ap.add_argument("--json", action="store_true", help="print the structured result as JSON")
    args = ap.parse_args()

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--live needs ANTHROPIC_API_KEY in your environment.", file=sys.stderr)
        sys.exit(2)

    result = run_crew_claude_agent(args.question, args.live, args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
