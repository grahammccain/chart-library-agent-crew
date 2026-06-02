"""
LangGraph port of the crew (Phase 2).

Same crew, different orchestrator. This file imports the EXACT validated nodes
from the framework-free `crew.py` (the same planner, the same tool-less rivals, and
the same Chart Library specialist with its provenance mandate + real
`/api/v1/cohort_analyze` node) and wires them into an idiomatic LangGraph
`StateGraph`:

    START -> plan -> (fan-out: one specialist node per consulted role, in parallel)
          -> synthesize -> END

The point of the port is to show that the Chart Library node drops into a real
framework UNCHANGED - we reuse crew.py's functions verbatim, so whatever the eval
measured for the native loop holds here too. The only new dependency is `langgraph`
itself (kept out of the repo-root requirements; see ports/requirements.txt).

Run:
    python ports/langgraph_crew.py "what usually happens to NVDA after a breakout?"
        # OFFLINE: canned specialists + fixture node; no key, no spend (free).
    python ports/langgraph_crew.py "Is NVDA extended, and what happens next?" --live
        # real Anthropic + real Chart Library node; needs ANTHROPIC_API_KEY; spends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Annotated, TypedDict

# Import the shared, validated crew nodes from the repo root (this file lives in
# ports/). One path insert keeps the example copy-paste runnable from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crew  # noqa: E402  (the framework-free crew - we reuse its nodes verbatim)

from langgraph.graph import START, END, StateGraph  # noqa: E402

try:  # the Send fan-out primitive moved across langgraph versions
    from langgraph.types import Send  # noqa: E402
except Exception:  # pragma: no cover - older langgraph
    from langgraph.constants import Send  # noqa: E402


def _merge(a, b):
    """Reducer so the parallel specialist branches can each write their own memo
    into the shared dict without clobbering one another."""
    out = dict(a or {})
    out.update(b or {})
    return out


class CrewState(TypedDict, total=False):
    question: str
    live: bool
    model: str
    consult: list
    reason: str
    memos: Annotated[dict, _merge]
    chartlibrary_fired: bool
    chartlibrary_calls: dict
    brief: str


def build_crew_graph(live, model):
    """Compile the LangGraph crew. The Anthropic client (live only) is created once
    here and closed over by the nodes - exactly one client per run, like crew.py."""
    client = crew._client() if live else None

    def plan_node(state: CrewState):
        consult, reason = crew.plan(state["question"], live, client, model)
        return {"consult": consult, "reason": reason}

    def fan_out(state: CrewState):
        # dynamic fan-out: one specialist branch per consulted role, in parallel
        return [Send("specialist", {"key": k, "question": state["question"]})
                for k in state["consult"]]

    def specialist_node(state):
        # state here is the per-branch Send payload {"key", "question"}.
        key, q = state["key"], state["question"]
        if key == "chartlibrary_analyst":
            memo, fired, stats = crew.chartlibrary_specialist(q, live, client, model)
            return {"memos": {key: memo}, "chartlibrary_fired": fired,
                    "chartlibrary_calls": stats}
        return {"memos": {key: crew.tool_less_specialist(key, q, live, client, model)}}

    def synth_node(state: CrewState):
        # order memos by the plan so the brief is deterministic + faithful to crew.py
        memos = {k: state["memos"][k] for k in state["consult"] if k in state["memos"]}
        return {"brief": crew.synthesize(state["question"], memos, live, client, model)}

    g = StateGraph(CrewState)
    g.add_node("plan", plan_node)
    g.add_node("specialist", specialist_node)
    g.add_node("synthesize", synth_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["specialist"])
    g.add_edge("specialist", "synthesize")  # barrier: runs once, after all branches
    g.add_edge("synthesize", END)
    return g.compile()


def run_crew_langgraph(question, live, model):
    graph = build_crew_graph(live, model)
    final = graph.invoke({"question": question, "live": live, "model": model,
                          "consult": [], "reason": "", "memos": {},
                          "chartlibrary_fired": False, "chartlibrary_calls": {},
                          "brief": ""})
    memos = {k: final["memos"][k] for k in final["consult"] if k in final["memos"]}
    return {"question": question, "live": live, "model": model, "orchestrator": "langgraph",
            "plan": {"consult": final["consult"], "reason": final["reason"]},
            "memos": memos, "chartlibrary_fired": final.get("chartlibrary_fired", False),
            "chartlibrary_calls": final.get("chartlibrary_calls", {}),
            "brief": final["brief"]}


def main():
    # Windows consoles default to cp1252; live model output (em dashes, arrows) would
    # crash print(). Make stdout UTF-8 and never-fail (same fix as crew.py).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="LangGraph port of the market-research crew (same Chart Library node).")
    ap.add_argument("question", help="a markets question")
    ap.add_argument("--live", action="store_true",
                    help="real Anthropic + real Chart Library node (needs ANTHROPIC_API_KEY; spends)")
    ap.add_argument("--model", default=crew.DEFAULT_MODEL,
                    help=f"orchestrator/specialist model (default {crew.DEFAULT_MODEL})")
    ap.add_argument("--json", action="store_true", help="print the structured result as JSON")
    args = ap.parse_args()

    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--live needs ANTHROPIC_API_KEY in your environment.", file=sys.stderr)
        sys.exit(2)

    result = run_crew_langgraph(args.question, args.live, args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("[orchestrator: LangGraph StateGraph - plan -> fan-out specialists -> synthesize]\n")
        crew._print_human(result)


if __name__ == "__main__":
    main()
