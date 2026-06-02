"""
Phase 0 orchestrator loop.

One neutral orchestrator is given a question plus a loadout of specialist tools
and decides which (if any) to call. We measure whether it reaches for the
chartlibrary tools when it should (recall) and leaves them alone when it should
not (over-fire) — the core "does the agent actually like chartlibrary" question.

Two interchangeable backends behind one `Turn` shape:
  * MockModel  - deterministic, free, offline. Scripts a plausible-but-SYNTHETIC
                 tool selection per prompt so the metrics/plumbing can be proven
                 without spending. Its numbers are NOT a validation result.
  * LiveModel  - the real Anthropic Messages API doing genuine tool selection.
                 This is the only backend whose numbers answer GO/NO-GO.

The system prompt is deliberately NEUTRAL — it never names chartlibrary or hints
that history/base-rates are special — so selection reflects the tool DESCRIPTIONS,
not a thumb on the scale.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from tools import (
    CHARTLIBRARY_INTROSPECT_NAME,
    CHARTLIBRARY_TOOL_NAMES,
    _parse_symbol,
    run_tool,
)

SYSTEM = (
    "You are a markets research orchestrator. The user asks a question about a "
    "stock or the market. You have a set of specialist tools, each covering one "
    "area. Decide which tool or tools, if any, are appropriate for THIS question, "
    "call them, then give a concise, grounded final answer based only on what they "
    "return. Call a tool only when it is relevant to the question; if none is "
    "relevant, answer directly. Do not call every tool by default."
)


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    text: str = ""
    tool_uses: list = field(default_factory=list)
    stop_reason: str = "end_turn"


# ---------------------------------------------------------------------------
# Mock backend (free, deterministic) — SYNTHETIC selection for plumbing only
# ---------------------------------------------------------------------------

def _infer_expected(rec) -> list:
    """The tool set a reasonable agent 'should' reach for, before noise."""
    cat = rec["category"]
    text = rec["prompt"].lower()
    if cat == "base_rate":
        return ["chartlibrary_search", "chartlibrary_cohort_analyze"]
    if cat == "composite":
        tools = ["chartlibrary_search", "chartlibrary_cohort_analyze"]
        if any(k in text for k in ("catalyst", "news", "earnings", "headline")):
            tools.append("news_catalysts")
        if any(k in text for k in ("technical", "setup", "chart", "entry")):
            tools.append("technical_analysis")
        if "macro" in text:
            tools.append("macro_regime")
        if any(k in text for k in ("size", "sizing", "trim", "exposure")):
            tools.append("risk_position")
        return tools
    if cat == "pure_ta":
        return ["technical_analysis"]
    if cat == "other":
        if any(k in text for k in ("p/e", "margin", "revenue", "valuation", "growth")):
            return ["fundamentals"]
        if any(k in text for k in ("news", "headline", "catalyst", "moving on")):
            return ["news_catalysts"]
        if any(k in text for k in ("vix", "risk-on", "risk-off", "macro", "term structure")):
            return ["macro_regime"]
        return ["fundamentals"]
    return []


class MockModel:
    """Deterministic stand-in. Noise rates differ by description arm so the
    smoke output shows the metric *can* separate v1 from v2 — this is an
    illustration of the measurement, not evidence about the real tools."""

    def __init__(self, desc="v2"):
        # v2 (Purpose + use-when + negative boundary) is scripted to miss less
        # and over-fire less than v1. Synthetic — to exercise the A/B math.
        self.miss = 0.06 if desc == "v2" else 0.14
        self.over_fire = 0.10 if desc == "v2" else 0.20
        self.spurious = 0.06
        self.rec = None
        self.rng = random.Random(0)
        self.step = 0

    def begin(self, rec):
        self.rec = rec
        self.rng = random.Random(rec["id"] + "|" + str(self.miss))
        self.step = 0

    def create(self, messages, tool_schemas) -> Turn:
        self.step += 1
        names = {t["name"] for t in tool_schemas}
        if self.step >= 2:
            called = _called_so_far(messages)
            # Chaining smoke (SYNTHETIC): if cohort_analyze surfaced a
            # suggested_introspections pointer and the introspect tool is on the
            # bench and we haven't chained yet, take the second hop. This lets the
            # mock demonstrate that the probe separates the plain vs nudge arms;
            # it is NOT evidence about real agent behaviour.
            if (CHARTLIBRARY_INTROSPECT_NAME in names
                    and CHARTLIBRARY_INTROSPECT_NAME not in called
                    and _has_suggestion(messages)):
                return Turn(tool_uses=[ToolUse(
                    id=f"mock_intro_{self.step}",
                    name=CHARTLIBRARY_INTROSPECT_NAME,
                    input={"cohort_id": _last_cohort_id(messages),
                           "where": "relative_volume_top_quartile"})],
                    stop_reason="tool_use")
            return Turn(text=_mock_answer(self.rec, called))

        chosen = []
        for t in _infer_expected(self.rec):
            if t not in names:
                continue
            if t in CHARTLIBRARY_TOOL_NAMES and self.rng.random() < self.miss:
                continue  # silent-ignore the node it should have used
            chosen.append(t)

        if not self.rec["expects_chartlibrary"] and self.rng.random() < self.over_fire:
            cl = "chartlibrary_cohort_analyze"
            if cl in names and cl not in chosen:
                chosen.append(cl)  # over-fire onto a pure-TA / off-lane prompt

        if self.rng.random() < self.spurious:
            pool = [n for n in names if n not in chosen and n not in CHARTLIBRARY_TOOL_NAMES]
            if pool:
                chosen.append(self.rng.choice(pool))

        uses = [ToolUse(id=f"mock_{i}", name=n, input=_mock_args(n, self.rec))
                for i, n in enumerate(chosen)]
        if not uses:
            return Turn(text=_mock_answer(self.rec, []))
        return Turn(tool_uses=uses, stop_reason="tool_use")


def _mock_args(name, rec) -> dict:
    sym = _parse_symbol(rec["prompt"]) or "SPY"
    if name == "chartlibrary_search":
        return {"query": f"{sym} 2026-05-29"}
    if name == "chartlibrary_cohort_analyze":
        return {"symbol": sym, "date": "2026-05-29", "timeframe": "1d"}
    if name == "macro_regime":
        return {}
    return {"symbol": sym}


def _called_so_far(messages) -> list:
    called = []
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    called.append(b["name"])
    return called


def _tool_result_contents(messages):
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content", "")
                    if isinstance(c, str):
                        yield c


def _has_suggestion(messages) -> bool:
    return any("suggested_introspections" in c for c in _tool_result_contents(messages))


def _last_cohort_id(messages) -> str:
    cid = None
    for c in _tool_result_contents(messages):
        if "cohort_id" in c:
            try:
                cid = json.loads(c).get("cohort_id", cid)
            except Exception:
                pass
    return cid or "coh_unknown"


def _mock_answer(rec, called) -> str:
    sym = _parse_symbol(rec["prompt"]) or "the market"
    if "chartlibrary_cohort_analyze" in called:
        return (f"[SYNTHETIC] {sym}: grounded in the historical cohort — calibrated 80% "
                f"5-day band and base-rate of follow-through give an empirical range.")
    return f"[SYNTHETIC] {sym}: qualitative read from available signals; no historical base rate."


# ---------------------------------------------------------------------------
# Live backend (real Anthropic tool selection) — the only validation-grade path
# ---------------------------------------------------------------------------

class LiveModel:
    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def begin(self, rec):
        pass

    def create(self, messages, tool_schemas) -> Turn:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            tools=tool_schemas,
            messages=messages,
        )
        text, tool_uses = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=block.input))
        return Turn(text=text, tool_uses=tool_uses, stop_reason=resp.stop_reason)


# ---------------------------------------------------------------------------
# Episode driver (shared by both backends)
# ---------------------------------------------------------------------------

def run_episode(rec, tool_schemas, backend, max_steps=6, tool_runner=None) -> dict:
    runner = tool_runner or run_tool
    backend.begin(rec)
    messages = [{"role": "user", "content": rec["prompt"]}]
    called, final = [], ""

    for _ in range(max_steps):
        turn = backend.create(messages, tool_schemas)
        if turn.tool_uses:
            content = []
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            for tu in turn.tool_uses:
                called.append(tu.name)
                content.append({"type": "tool_use", "id": tu.id,
                                "name": tu.name, "input": tu.input})
            messages.append({"role": "assistant", "content": content})

            results = [{"type": "tool_result", "tool_use_id": tu.id,
                        "content": runner(tu.name, tu.input)}
                       for tu in turn.tool_uses]
            messages.append({"role": "user", "content": results})
        else:
            final = turn.text
            break

    return {"id": rec["id"], "category": rec["category"],
            "expects_chartlibrary": rec["expects_chartlibrary"],
            "called": called, "final": final}
