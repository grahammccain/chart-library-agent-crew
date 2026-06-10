# chartlibrary agent-crew — reference research crew + eval harness

A reference research crew — plus the tiny, honest eval harness that justifies it —
built around one question:

> **When a neutral markets orchestrator is given a realistic question and a bench
> of competing specialist tools, does it actually reach for chartlibrary at the
> right times — and does having it produce a better-grounded answer?**

This repo is **two things**: the runnable reference crew (`crew.py` + three
framework ports) that wires our node alongside the other specialists, **and** the
honest eval harness that justified building it — the measurement proving a neutral
orchestrator actually reaches for chartlibrary at the right times, and answers
better when it does. The harness ships alongside the crew so you can re-run the
receipt yourself. It is *not* a framework: our node drops into whatever
orchestrator you already use.

> **The writeup:** the rationale behind this repo is on the Chart Library blog —
> start with **[How to Build a Market-Research Agent Crew in 2026](https://chartlibrary.io/blog/market-research-agent-crew-2026)**
> (frameworks, data costs, and the missing primitive). Per-post links are in
> [Further reading](#further-reading).

## Why this exists

We are the *calibrated historical-analog* slot in a multi-agent trading stack —
not the orchestration glue. The reference app's job is to show developers a clean
pattern for wiring our MCP node alongside the other specialists they already have
(fundamentals, technicals, news, macro, risk). But a demo only helps adoption if
the agent genuinely *uses* our node. So we measure that first, on the substrate
every framework shares: model-native tool selection.

## What it measures

1. **Selection** — per prompt, did the orchestrator call chartlibrary?
   * `recall` on the prompts that *should* use it (base-rate / composite questions)
   * `over-fire` on the prompts that should *not* — most importantly the
     **pure-technical-analysis** collision set ("what's AAPL's RSI(14)?"), where a
     live-indicator tool is the right call and our historical-cohort node is wrong.
   Scoring both catches the two real failure modes — silent-ignore and
   over-firing — that an outcome-only eval would miss.

2. **Answer lift (A/B)** — for the should-use prompts, run the orchestrator
   *with* chartlibrary available vs *with it removed*, then have a blind judge
   pick the better-grounded final answer. The with-node win-rate is both the
   proof-it-helps and the marketing receipt.

3. **Count-stress** — run the same suite at a 7-tool and a 13-tool loadout to see
   if selection degrades as the bench grows (the "skill shadowing" effect).

It also A/Bs two **description arms** for our two tools:
* `v1` — the real production MCP descriptions (what an agent sees today).
* `v2` — improved: explicit Purpose + "use this when" + a hard negative boundary
  against the technical-analysis tool. Tells us if better copy alone moves
  selection.

## Run the crew (the reference app)

`crew.py` is the runnable demo the harness exists to justify: a framework-free
thin native loop (just the Anthropic SDK + stdlib) that wires the Chart Library
MCP node into a small crew of specialists as the *historical base-rate* analyst.
No LangChain, no CrewAI — the point is that the **node** is plug-and-play, so the
same wiring drops into whatever orchestrator you already use.

```bash
# OFFLINE: canned specialists + fixture node — free, no key, proves the plumbing
python crew.py "what usually happens to NVDA the week after a high-volume breakout?"

# LIVE: real Anthropic orchestrator + the real Chart Library node
# (anonymous to Chart Library; needs ANTHROPIC_API_KEY; spends money)
python crew.py "is NVDA extended here, and what usually happens next?" --live
```

The orchestrator picks which specialists a question needs; the Chart Library node
runs on the real public endpoint and grounds its memo in calibrated base rates
**with provenance** ("per Chart Library's N analogs…") so the rest of the crew
trusts the numbers; the lead then writes one honest brief — base rates and ranges,
never a single directional forecast. Name no ticker and the node says so plainly
instead of guessing.

## Run a framework port (same node, three real orchestrators)

The whole point is that the Chart Library node is **plug-and-play** — drop it into
whatever orchestrator you already run. `ports/` proves it across **three** of them,
each reusing crew.py's *exact* validated pieces (the same node, the same provenance
mandate, the same real `/api/v1/cohort_analyze` call, the same v2 USE-WHEN / boundary
language) wired into the framework **unchanged**:

| port | framework | how the node plugs in |
|------|-----------|-----------------------|
| `ports/langgraph_crew.py` | **LangGraph** | `StateGraph`: plan → parallel `Send` fan-out → synthesize; reuses crew.py's node functions verbatim |
| `ports/openai_agents_crew.py` | **OpenAI Agents SDK** | one `Agent` + Chart Library `@function_tool`s; an OpenAI model decides when to call the node (cross-vendor) |
| `ports/claude_agent_crew.py` | **Claude Agent SDK** | Chart Library exposed as an in-process MCP server (`create_sdk_mcp_server` + `@tool`) — the closest to the real product |

```bash
pip install -r ports/requirements.txt   # the three frameworks (install only what you try)

# OFFLINE (free, no key): canned crew + the node wiring constructed — same routing as crew.py
python ports/langgraph_crew.py      "what usually happens to NVDA after a breakout?"
python ports/openai_agents_crew.py  "what usually happens to NVDA after a breakout?"
python ports/claude_agent_crew.py   "what usually happens to NVDA after a breakout?"

# LIVE (paid): real orchestrator + the real Chart Library node
python ports/langgraph_crew.py      "is NVDA extended, and what next?" --live          # ANTHROPIC_API_KEY
python ports/openai_agents_crew.py  "is NVDA extended, and what next?" --live --model gpt-4.1  # OPENAI_API_KEY
python ports/claude_agent_crew.py   "is NVDA extended, and what next?" --live          # ANTHROPIC_API_KEY + SDK runtime
```

Offline, every port routes identically to the native loop: a base-rate question pulls
the Chart Library node (fired, with provenance), a no-ticker question makes it decline
honestly, and a pure-technical question (`AAPL's RSI(14)?`) never touches it. (LangGraph's
orchestration is graph-structural, so its offline run is a full end-to-end; the two
model-driven SDKs run the same canned path offline and construct their agent/MCP wiring
for free — the paid `--live` run is what exercises each model's own tool-selection loop.)

> **Port-by-port walkthrough:** [How to Add a Stock Base-Rate MCP Node to LangGraph,
> the OpenAI Agents SDK, and the Claude Agent SDK](https://chartlibrary.io/blog/stock-mcp-node-langgraph-openai-claude)
> walks through each of the three ports above, with the code.

## Run the harness (the receipt)

Mock mode is **free and offline** — it proves the harness and metrics run
end-to-end with a deterministic stand-in model. Its numbers are **synthetic**
and clearly labelled; they are *not* a validation result.

```bash
pip install -r requirements.txt

# free, offline plumbing check (synthetic numbers):
python run.py --mode mock --desc v2 --loadout both --ab
```

Live mode makes **real, paid** Anthropic tool-selection calls — this is the only
path whose numbers answer GO/NO-GO. It refuses to spend without a key *and* an
explicit `--yes`:

```bash
# PowerShell:  $env:ANTHROPIC_API_KEY = "<key>"
python run.py --mode live --desc v2 --loadout 7 --ab --yes
# compare arms:
python run.py --mode live --desc v1 --ab --yes
```

Useful flags: `--orchestrator-model` (default `claude-sonnet-4-6`),
`--judge-model` (default `claude-haiku-4-5-20251001`), `--limit N`,
`--out results.json`.

## Receipt-ablation eval (does the receipt itself change reasoning?)

`receipt_ablation.py` answers a narrower, harder question than the with-vs-without
A/B above: **holding the toolkit fixed, does the calibration receipt change the
agent's reasoning quality?** Both arms use the *same* `chartlibrary_cohort_analyze`
tool; the only delta is a thin response filter (`strip_receipt`) that deletes
exactly the receipt keys — the top-level `calibration` (the split-conformal 80%
band + empirical coverage + `calibration_n`) and `provenance` (the attribution
string) — on Arm B. The agent still gets the full outcome distribution, feature
importance, regime stratification and risk profile on both arms; it loses *only*
the receipt.

A blind dual-judge scores eight reasoning dimensions — the six prior plus two new
ones the receipt should move: **confidence-qualification** (does the answer
correctly qualify its confidence?) and **weak-comp-set flagging** (does it
flag/refuse when the cohort is thin?). Judges never learn which arm is which.
**Predictions and honest-outcome handling are pre-registered** and printed in the
report header: the receipt arm is predicted to win the two qualification
dimensions, parity is expected on breadth, and *either* outcome is publishable
(receipt-wins → a marketing number; parity/loss → reweight toward human-facing
receipt surfaces). No number may be cited until a live run.

```bash
# free, offline end-to-end (50 subjects across 50 distinct anchor dates; SYNTHETIC):
python receipt_ablation.py --mode mock

# paid, real — prints the estimated cost (~$10-30) and refuses to spend without --yes:
python receipt_ablation.py --mode live --yes --env-file .env --real-chartlibrary
```

Writes `results_receipt_ablation.json` + a `…​.md` summary. Useful flags:
`--judge-model` / `--judge-model-2` (dual blind judges, default haiku + opus),
`--real-chartlibrary` (live cohort pulls instead of fixtures, for the paid run),
`--limit N`.

## The CI gate (the regression guard)

`gate.py` turns the GO thresholds into an enforced exit code, so a regression
fails the build instead of sliding by. It reads a `results.json` written by
`run.py` — it never calls the model or spends money itself.

```bash
# free: prove the harness + metrics + gate all run (mock numbers are synthetic)
python run.py --mode mock --desc v2 --loadout both --ab --out results.json
python gate.py results.json

# paid: enforce the thresholds on REAL tool-selection numbers
python run.py --mode live --desc v2 --loadout 7 --ab --yes --out results.json
python gate.py results.json --require-live
```

Defaults (all overridable): `recall_overall >= 0.80`, `over-fire (fpr) <= 0.15`,
`answer-lift (decided) >= 0.60`; with multiple judges *every* judge must clear the
lift bar. `--require-live` refuses to certify synthetic mock numbers, so a free
per-PR mock run proves the plumbing without ever masquerading as validation.

An example GitHub Actions workflow (below) wires both: a **free mock** plumbing job
on every push/PR, and a **paid live** validation job on manual dispatch only (needs
the `ANTHROPIC_API_KEY` repo secret). There is deliberately no schedule — every
paid run stays a human decision. `test_gate.py` self-checks the gate's pass/fail
edges (run `python test_gate.py`). Copy the workflow below to
`.github/workflows/eval-gate.yml` to enable CI.

<details>
<summary>example <code>.github/workflows/eval-gate.yml</code></summary>

```yaml
name: chartlibrary eval gate

# Two gates over the eval harness:
#   * plumbing   - FREE mock run on every push/PR. Proves the harness + metrics +
#                  gate all execute end-to-end. Its numbers are SYNTHETIC (not a
#                  verdict) - gate.py treats a mock file as a plumbing check only.
#   * validation - PAID live run, MANUAL trigger only (workflow_dispatch). Real
#                  Anthropic tool-selection; enforces the GO thresholds on real
#                  numbers via `gate.py --require-live`. Needs the ANTHROPIC_API_KEY
#                  repo secret.
#
# Deliberately NO `schedule:` - live runs spend money, so each paid run stays a
# human decision (matches the repo's cost-gate posture). Add a schedule later only
# if a recurring public receipt is wanted.
#
# NOTE: paths assume the harness lives at the repo root (current layout). If you
# move it under a subdir, set `working-directory:` on the jobs.

on:
  push:
    branches: [ main ]
  pull_request:
  workflow_dispatch:

jobs:
  plumbing:
    name: plumbing (free, mock, synthetic)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: gate self-check
        run: python test_gate.py
      - name: run harness (mock, free)
        run: python run.py --mode mock --desc v2 --loadout both --ab --out results.json
      - name: gate (plumbing only)
        run: python gate.py results.json

  validation:
    name: validation (paid, live, real numbers)
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: run harness (live, paid)
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python run.py --mode live --desc v2 --loadout 7 --ab --yes --out results.json
      - name: gate (enforce live thresholds)
        run: python gate.py results.json --require-live
      - name: upload receipt
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-receipt
          path: results.json
```

</details>

## Files

| file | role |
|------|------|
| `crew.py` | **the reference app** — framework-free thin native loop: orchestrator + specialists + the real Chart Library node (offline + `--live`) |
| `real_tools.py` | the real Chart Library node — live `/api/v1/cohort_analyze` over stdlib urllib, size-trimmed, date-anchored (shared by the crew and the A/B) |
| `tools.py` | chartlibrary tool schemas (v1 real / v2 improved) + 5 rival specialists + 6 padding tools; deterministic fixture outputs |
| `prompts.py` | 29 labelled prompts: base-rate & composite (should use), pure-TA & off-lane (should not) |
| `harness.py` | neutral orchestrator loop; mock + live backends behind one `Turn` |
| `evaluate.py` | selection metrics + with/without answer-lift judges |
| `run.py` | CLI, cost guard, report, JSON output |
| `receipt_ablation.py` | **receipt-ablation eval** — same toolkit both arms; a thin `strip_receipt` proxy removes only the `calibration` + `provenance` receipt keys on Arm B. Blind dual-judge over 8 dimensions (incl. confidence-qualification + weak-comp-set flagging); pre-registered predictions; mock e2e free, live prints cost. Tested by `test_receipt_ablation.py` |
| `receipt_subjects.py` | 50 anchored subjects (15 reused for comparability + 35 fresh) spanning 50 distinct anchor dates, incl. deliberately thin names for the weak-comp-set dimension |
| `gate.py` | **CI gate** — enforces recall / over-fire / answer-lift thresholds on a `results.json` as an exit code (mock-aware; `--require-live` refuses synthetic). Self-checked by `test_gate.py`. Wire it into CI with the example workflow in [The CI gate](#the-ci-gate-the-regression-guard) |
| `ports/langgraph_crew.py` | **framework port** — the same crew on a LangGraph `StateGraph` (reuses `crew.py`'s nodes verbatim; offline + `--live`). Proves the node drops into a real orchestrator unchanged |
| `ports/openai_agents_crew.py` | **framework port** — the same node on the OpenAI Agents SDK; an OpenAI model orchestrates and reaches for the Chart Library `function_tool`s (cross-vendor receipt). Offline constructs the `Agent`; `--live` runs the model loop |
| `ports/claude_agent_crew.py` | **framework port** — Chart Library as an in-process MCP server to a Claude agent (`create_sdk_mcp_server` + `@tool`); the closest port to the real product. Offline constructs the server + options; `--live` runs the agent |

## Honest caveats

* Fixtures stand in for live data. Selection depends on the **descriptions**, so
  fixtures are faithful for the selection question; the answer-lift A/B uses
  representative numbers (swap in real prod data in Phase 1).
* Mock numbers are synthetic plumbing only. Never quote them as evidence.
* A single judge model has its own biases; treat the A/B as directional and
  consider a second judge before leaning on it for marketing.

## What's here, and what's next

Phase 0 passed — the orchestrator reaches for the node and answers better with it —
so this repo ships the proof, not just the test: the reference crew (`crew.py`,
Phase 1) and three framework ports — LangGraph, the OpenAI Agents SDK, and the
Claude Agent SDK (Phase 2) — all reusing the same validated node. The eval harness
stays in the repo so you (or a CI gate) can re-run the receipt any time.

## Further reading

The design decisions behind this crew are written up on the Chart Library blog:

* [How to Build a Market-Research Agent Crew in 2026](https://chartlibrary.io/blog/market-research-agent-crew-2026) — the overview: crew anatomy, frameworks, and the missing primitive.
* [How to Add a Stock Base-Rate MCP Node to LangGraph, the OpenAI Agents SDK, and the Claude Agent SDK](https://chartlibrary.io/blog/stock-mcp-node-langgraph-openai-claude) — the port-by-port walkthrough for `ports/`.
* [What Does It Cost to Build an AI Trading Agent in 2026?](https://chartlibrary.io/blog/ai-trading-agent-data-cost-2026) — the specialist data-stack budget, and which lanes are free.
* [The Oracle Fallacy](https://chartlibrary.io/blog/oracle-fallacy-trading-agent-backtest) — why a backtest flatters a trading agent, and what calibrated base rates fix.
* [Technical Analysis vs. Historical Base Rates](https://chartlibrary.io/blog/technical-analysis-vs-base-rates) — the boundary this crew draws between the live-TA node and the base-rate node.
