"""
chart-library-agent-crew - a runnable, framework-FREE multi-agent research crew.

A thin native loop (no LangChain / no CrewAI / no agent framework - just the
Anthropic SDK + Python stdlib) that coordinates a small team of specialist agents
to answer a markets question, and shows what it looks like to drop the Chart
Library MCP node into that team as the calibrated historical-base-rate specialist.

Why so plain? Because the thing worth demonstrating is the NODE, not orchestration
glue. If a ~250-line native loop gets measurable lift from the node, so will your
LangGraph / OpenAI-Agents / Claude-Agent-SDK crew - the node is plug-and-play. The
eval/ harness beside this file is the receipt: with the node, a blind judge prefers
the crew's answer ~0.80-0.87 of the time on base-rate questions (see README).

Topology (one orchestrator + a tight roster of specialists - kept small on purpose;
tool-selection degrades as the bench grows):

    question
       │
       ▼  plan()        orchestrator picks the RELEVANT subset of specialists
       ▼  specialists   each = one focused Claude call:
       │                  • rivals reason qualitatively (no live data feed)
       │                  • chartlibrary_analyst runs the 2 REAL Chart Library
       │                    tools and grounds its memo in calibrated base rates,
       │                    WITH PROVENANCE so the rest of the crew trusts the numbers
       ▼  synthesize()  orchestrator merges the memos into one honest brief

Run:
    python crew.py "Is NVDA extended here, and what usually happens next?" --live
        # real Anthropic + real Chart Library node; needs ANTHROPIC_API_KEY; spends.
    python crew.py "..."                 # OFFLINE: canned specialists + fixture node,
                                         # no key, no spend - proves the plumbing.

The Chart Library node is anonymous (no Chart Library key needed) and talks to the
public endpoint over stdlib urllib, so this repo's only dependency is `anthropic`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Reuse the VALIDATED node + rival descriptions verbatim so the demo ships exactly
# what the eval measured: CHARTLIBRARY_V2 = the blessed tool descriptions (Purpose +
# USE-WHEN + the hard boundary vs. the technical analyst); make_real_runner = the
# real /api/v1/cohort_analyze call with size-trimming + date anchoring.
from real_tools import DEFAULT_ANCHOR_DATE, make_real_runner
from tools import CHARTLIBRARY_V2, _parse_symbol, run_tool

DEFAULT_MODEL = "claude-sonnet-4-6"   # capable + cheap; opus is overkill for a demo
MAX_TOOL_STEPS = 4

# --------------------------------------------------------------------------- #
# The roster.  key -> (display name, one-line capability shown to the planner).  #
# The chartlibrary line reuses the v2 USE-WHEN / boundary language so the node    #
# is described to the orchestrator exactly as it was when we measured the lift.   #
# --------------------------------------------------------------------------- #

SPECIALISTS = {
    "chartlibrary_analyst": (
        "Chart Library (historical base rates)",
        "Calibrated HISTORICAL BASE RATES: given a ticker (+date), returns what "
        "ACTUALLY HAPPENED NEXT after the most similar historical setups (25M+ real "
        "analogs, calibrated 80% bands). USE WHEN the question is about historical "
        "frequency, 'what usually happens next', odds, or expected range. Do NOT use "
        "it to read the CURRENT chart - that is the technical analyst's job."),
    "technical_analyst": (
        "Technical analyst",
        "Reads the CURRENT chart: RSI/MACD/moving averages/support-resistance - what "
        "the chart looks like NOW (not what happened next historically)."),
    "fundamentals_analyst": (
        "Fundamentals analyst",
        "Valuation (P/E, P/S, EV/EBITDA), revenue/earnings growth, margins, balance "
        "sheet health."),
    "news_analyst": (
        "News & catalysts analyst",
        "Recent headlines, catalysts, and a narrative-change / sentiment read."),
    "macro_strategist": (
        "Macro strategist",
        "Top-down regime: VIX & term structure, rates, credit, sector posture. No "
        "specific ticker required."),
    "risk_analyst": (
        "Risk & sizing analyst",
        "Position sizing, stop placement, R-multiple, and downside framing."),
}

# --------------------------------------------------------------------------- #
# Specialist role prompts.                                                      #
# --------------------------------------------------------------------------- #

# The four tool-less rivals reason from the model's own knowledge. They are told to
# be explicit that their read is QUALITATIVE (no live data feed) - both because it
# is honest and because it is exactly the contrast that makes the chartlibrary
# specialist's verifiable, sourced numbers stand out. A real deployment would wire
# each of these to its own data API; that is left to the reader.
_QUALITATIVE_TAIL = (
    " You have no live data feed in this demo, so be explicit that your read is "
    "qualitative/illustrative rather than sourced. Keep it to 3-4 sentences.")

ROLE_PROMPTS = {
    "technical_analyst":
        "You are the technical analyst on a markets research crew. Describe the likely "
        "current technical state of the stock - trend, momentum (RSI/MACD), key moving "
        "averages, support/resistance - i.e. what the chart looks like NOW." + _QUALITATIVE_TAIL,
    "fundamentals_analyst":
        "You are the fundamentals analyst on a markets research crew. Comment on "
        "valuation, growth, margins, and balance-sheet health relevant to the question."
        + _QUALITATIVE_TAIL,
    "news_analyst":
        "You are the news & catalysts analyst on a markets research crew. Note the kinds "
        "of recent catalysts or narrative shifts that would matter for the question."
        + _QUALITATIVE_TAIL,
    "macro_strategist":
        "You are the macro strategist on a markets research crew. Give the top-down "
        "regime context - volatility, rates/credit, risk-on vs risk-off, sector posture."
        + _QUALITATIVE_TAIL,
    "risk_analyst":
        "You are the risk & sizing analyst on a markets research crew. Frame the downside "
        "and how one might size/stop a position given the setup." + _QUALITATIVE_TAIL,
}

# The star.  It HAS real tools, and the PROVENANCE instruction is the product insight
# the eval surfaced: blind judges (and downstream agents) discount specific numbers as
# "hallucinated" unless they are sourced - so we make the specialist attribute every
# number to Chart Library and its sample sizes. That single change is the top remaining
# lever on the with-vs-without lift.
CHARTLIBRARY_ROLE = (
    "You are the historical base-rate analyst on a markets research crew. You have two "
    "Chart Library tools that return the REAL, calibrated distribution of what happened "
    "NEXT after the most similar historical setups to a given (ticker, date). Use them "
    "to answer; if no date is given, anchor to the latest settled session. "
    "CRITICAL - attach PROVENANCE to every number so the rest of the crew trusts it, "
    "e.g. 'per Chart Library's N historical analogs (calibrated 80% band)'. Report the "
    "base-rate of follow-through and the calibrated 80% range. Surface the DISTRIBUTION; "
    "do NOT collapse it into a single directional price forecast. If the question names "
    "no ticker, say so plainly - you need a symbol to anchor a cohort. Keep it to 5 "
    "sentences max.")

PLAN_SYSTEM = (
    "You are the lead of a markets research crew. Given the user's question and the "
    "roster of specialists, choose the SUBSET whose expertise is genuinely relevant to "
    "THIS question - do not consult everyone by default. Reply with ONLY JSON: "
    "{\"consult\": [\"<key>\", ...], \"reason\": \"<=12 words\"}.")

SYNTH_SYSTEM = (
    "You are the lead of a markets research crew writing the final brief from your "
    "specialists' memos. Integrate them into one concise, decision-useful answer. "
    "PRESERVE the provenance on any historical numbers (e.g. 'per Chart Library's N "
    "analogs'). Lead with the empirically grounded base rates where available, and be "
    "explicit about what is qualitative vs. sourced. Surface the distribution / range; "
    "do NOT issue a single directional price forecast.")


# --------------------------------------------------------------------------- #
# Anthropic plumbing (zero framework - one helper over the Messages API).        #
# --------------------------------------------------------------------------- #

def _client():
    import anthropic
    return anthropic.Anthropic()


def _message(client, model, system, messages, tools=None, max_tokens=900):
    """One Messages API call. Returns (text, tool_uses, stop_reason)."""
    kw = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
    if tools:
        kw["tools"] = tools
    resp = client.messages.create(**kw)
    text = "".join(b.text for b in resp.content if b.type == "text")
    tool_uses = [(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"]
    return text, tool_uses, resp.stop_reason


# --------------------------------------------------------------------------- #
# Step 1 - plan: which specialists does THIS question need?                      #
# --------------------------------------------------------------------------- #

def plan(question, live, client, model):
    if not live:
        return _offline_plan(question)
    roster = "\n".join(f"- {k}: {desc}" for k, (_, desc) in SPECIALISTS.items())
    user = f"QUESTION:\n{question}\n\nROSTER:\n{roster}"
    text, _, _ = _message(client, model, PLAN_SYSTEM,
                          [{"role": "user", "content": user}], max_tokens=200)
    consult, reason = _parse_plan(text)
    consult = [k for k in consult if k in SPECIALISTS] or list(SPECIALISTS)
    return consult, reason


def _parse_plan(raw):
    try:
        obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        consult = [str(x) for x in obj.get("consult", [])]
        return consult, str(obj.get("reason", "")).strip()
    except Exception:
        # be forgiving: scan for any roster keys mentioned in the text
        return [k for k in SPECIALISTS if k in raw], "(unparsed plan)"


def _offline_plan(question):
    t = question.lower()
    consult = []
    base_intent = any(k in t for k in (
        "usually", "historical", "base rate", "how often", "odds", "typical",
        "what happens", "probability", "follow through", "follow-through", "next",
        "distribution", "forward", "expected"))
    # "what does the chart look like NOW" signals -> the technical analyst's lane,
    # not the historical-base-rate node (mirrors the v2 boundary the live eval
    # validated at 0.00 over-fire on pure-TA prompts).
    ta_now = any(k in t for k in (
        "overbought", "oversold", "rsi", "macd", "resistance", "support",
        "moving average", "50-day", "200-day", "bollinger", "golden cross"))
    has_ticker = bool(_parse_symbol(question))
    if base_intent or (has_ticker and not (ta_now and not base_intent)):
        consult.append("chartlibrary_analyst")
    consult.append("technical_analyst")
    if any(k in t for k in ("news", "catalyst", "headline", "earnings")):
        consult.append("news_analyst")
    if any(k in t for k in ("valuation", "p/e", "margin", "growth", "fundamental")):
        consult.append("fundamentals_analyst")
    if any(k in t for k in ("macro", "vix", "regime", "rates", "market")):
        consult.append("macro_strategist")
    if any(k in t for k in ("size", "sizing", "stop", "risk", "position")):
        consult.append("risk_analyst")
    seen = set()
    consult = [k for k in consult if not (k in seen or seen.add(k))]
    return consult, "[offline] keyword routing"


# --------------------------------------------------------------------------- #
# Step 2 - specialists.                                                         #
# --------------------------------------------------------------------------- #

def tool_less_specialist(key, question, live, client, model):
    if not live:
        return _offline_specialist_memo(key, question)
    text, _, _ = _message(client, model, ROLE_PROMPTS[key],
                          [{"role": "user", "content": question}], max_tokens=400)
    return text.strip()


def chartlibrary_specialist(question, live, client, model):
    """The one specialist wired to a real data node. Returns (memo, fired, stats)."""
    if not live:
        return _offline_chartlibrary_memo(question)

    runner = make_real_runner()
    schemas = list(CHARTLIBRARY_V2.values())
    messages = [{"role": "user", "content": question}]
    fired = False

    for _ in range(MAX_TOOL_STEPS):
        text, tool_uses, _ = _message(client, model, CHARTLIBRARY_ROLE,
                                      messages, tools=schemas, max_tokens=900)
        if not tool_uses:
            return text.strip(), fired, dict(runner.stats)
        fired = True
        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": tid, "name": name, "input": inp}
            for (tid, name, inp) in tool_uses]
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": runner(name, inp)}
            for (tid, name, inp) in tool_uses]})

    # ran out of tool steps - make one last no-tools call to force a memo
    text, _, _ = _message(client, model, CHARTLIBRARY_ROLE, messages, max_tokens=900)
    return text.strip(), fired, dict(runner.stats)


# --------------------------------------------------------------------------- #
# Step 3 - synthesize the brief.                                                #
# --------------------------------------------------------------------------- #

def synthesize(question, memos, live, client, model):
    if not live:
        return _offline_synth(question, memos)
    blocks = "\n\n".join(f"## {SPECIALISTS[k][0]}\n{m}" for k, m in memos.items())
    user = f"QUESTION:\n{question}\n\nSPECIALIST MEMOS:\n{blocks}"
    text, _, _ = _message(client, model, SYNTH_SYSTEM,
                          [{"role": "user", "content": user}], max_tokens=1100)
    return text.strip()


# --------------------------------------------------------------------------- #
# Offline canned content (free; clearly SYNTHETIC; proves the plumbing only).    #
# --------------------------------------------------------------------------- #

_OFFLINE_MEMOS = {
    "technical_analyst": "qualitative read of trend/momentum; no live feed in offline mode.",
    "fundamentals_analyst": "qualitative valuation/growth/margin notes; no live feed offline.",
    "news_analyst": "qualitative catalyst/narrative read; no live feed offline.",
    "macro_strategist": "qualitative regime/volatility context; no live feed offline.",
    "risk_analyst": "qualitative downside/sizing framing; no live feed offline.",
}


def _offline_specialist_memo(key, question):
    return f"[SYNTHETIC] {_OFFLINE_MEMOS.get(key, 'qualitative note offline.')}"


def _offline_chartlibrary_memo(question):
    sym = _parse_symbol(question)
    if not sym:
        memo = ("[SYNTHETIC] No ticker named, so I can't anchor a cohort - Chart Library "
                "needs a (symbol, date) to find historical analogs. (This is the honest "
                "no-ticker boundary; with a ticker I'd return calibrated base rates.)")
        return memo, False, {"ok": 0, "error": 0, "degraded": 1, "fixture": 0}
    fix = json.loads(run_tool("chartlibrary_cohort_analyze",
                              {"symbol": sym, "date": DEFAULT_ANCHOR_DATE, "timeframe": "1d"}))
    h = fix["horizons"]["5d"]
    memo = (f"[SYNTHETIC] Per Chart Library's {fix['n_analogs']} historical analogs for "
            f"{sym}-like setups (calibrated 80% band), the 5-day outcome historically ran "
            f"{h['p10']}% to {h['p90']}% (median {h['p50']}%), with {int(h['pct_positive']*100)}% "
            f"closing higher; cohort tightness {fix['cohort_tightness']}. Empirical base "
            f"rates with provenance - not a forecast.")
    return memo, True, {"ok": 0, "error": 0, "degraded": 0, "fixture": 1}


def _offline_synth(question, memos):
    lines = [f"[SYNTHETIC crew brief]  Q: {question}", ""]
    for k, m in memos.items():
        lines.append(f"- {SPECIALISTS[k][0]}: {m}")
    lines += ["", "(Offline canned synthesis - run with --live for a real integrated brief.)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Drive the crew + CLI.                                                          #
# --------------------------------------------------------------------------- #

def run_crew(question, live, model):
    client = _client() if live else None
    consult, reason = plan(question, live, client, model)
    memos, fired, stats = {}, False, {}
    for key in consult:
        if key == "chartlibrary_analyst":
            memos[key], fired, stats = chartlibrary_specialist(question, live, client, model)
        else:
            memos[key] = tool_less_specialist(key, question, live, client, model)
    brief = synthesize(question, memos, live, client, model)
    return {"question": question, "live": live, "model": model,
            "plan": {"consult": consult, "reason": reason},
            "memos": memos, "chartlibrary_fired": fired,
            "chartlibrary_calls": stats, "brief": brief}


def _print_human(r):
    mode = "LIVE" if r["live"] else "OFFLINE (canned, no spend)"
    print(f"=== chart-library-agent-crew  [{mode}] ===")
    print(f"Q: {r['question']}\n")
    print(f"-- plan --  consult: {', '.join(r['plan']['consult'])}")
    print(f"            reason: {r['plan']['reason']}\n")
    for k, m in r["memos"].items():
        print(f"-- {SPECIALISTS[k][0]} --")
        print(f"{m}\n")
    if "chartlibrary_analyst" in r["memos"]:
        cc = r["chartlibrary_calls"]
        print(f"[chartlibrary node fired: {r['chartlibrary_fired']}  "
              f"real-calls ok={cc.get('ok',0)} err={cc.get('error',0)} "
              f"degraded={cc.get('degraded',0)} fixture={cc.get('fixture',0)}]\n")
    print("-- BRIEF --")
    print(r["brief"])
    print("\n(receipt: eval/ shows a blind judge prefers the WITH-node answer "
          "~0.80-0.87 on base-rate questions - see README.)")


def main():
    # Windows consoles default to cp1252; live model output (em dashes, arrows, …)
    # would crash print(). Make stdout UTF-8 and never-fail.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="Framework-free multi-agent market-research crew demo "
                    "(Chart Library MCP node as the historical-base-rate specialist).")
    ap.add_argument("question", help="a markets question, e.g. 'what usually happens after NVDA gaps up 5%%?'")
    ap.add_argument("--live", action="store_true",
                    help="real Anthropic calls + real Chart Library node (needs ANTHROPIC_API_KEY; spends money)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"orchestrator/specialist model (default {DEFAULT_MODEL})")
    ap.add_argument("--json", action="store_true", help="print the structured result as JSON")
    args = ap.parse_args()

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--live needs ANTHROPIC_API_KEY in your environment.", file=sys.stderr)
        sys.exit(2)

    result = run_crew(args.question, args.live, args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
