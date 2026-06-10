"""
Subjects for the receipt-ablation eval (research/specs/receipt_ablation_eval_v1).

Each subject is an out-of-sample markets question anchored to a concrete
(symbol, date) so the eval has by-date discipline. The spec asks for ~50 subjects
spanning >=40 DISTINCT anchor dates, reusing the prior subject list for
comparability plus fresh post-2026-05 anchors.

Composition:
  * REUSED   — the should-use prompts from prompts.py (the prior eval's positive
               set: base_rate + composite). These give comparability with the
               retired study. We attach a concrete anchor date to each (the prior
               prompts were date-free; the ablation needs an anchor so the live
               arm can pull a real cohort and so the by-date count is honest).
  * FRESH    — additional post-2026-05 anchors across a spread of symbols/dates,
               including a few deliberately THIN/edge anchors (small or obscure
               names) to exercise the weak-comp-set dimension.

In --mode mock the (symbol, date) only drives the deterministic fixture; in
--mode live --real-chartlibrary it anchors a real cohort pull. Every date here is
a settled past trading day (< 2026-05-30) so the live endpoint has EOD embeddings.

`date` is the anchor (ISO YYYY-MM-DD). `category` mirrors the prior taxonomy.
`source` = 'reused' | 'fresh' for provenance.
"""

from __future__ import annotations

# --- REUSED: the prior eval's should-use positives, each given a distinct anchor
# (text trimmed to the question; symbol/date carry the anchor). Dates are spread
# so reused subjects alone already contribute many distinct dates.
_REUSED = [
    ("rbr01", "NVDA", "2024-06-18", "base_rate",
     "NVDA broke out to a new all-time high on roughly 3x average volume. "
     "Historically, how often does that kind of breakout follow through, and "
     "what's the typical range over the next 5 days?"),
    ("rbr02", "TSLA", "2024-04-24", "base_rate",
     "TSLA gapped up about 9% on earnings. What have similar earnings gaps done "
     "over the following week historically?"),
    ("rbr03", "AMD", "2024-08-06", "base_rate",
     "AMD pulled back to its 50-day moving average after a big run. What's the "
     "historical base rate that the uptrend resumes from here?"),
    ("rbr04", "SPY", "2024-10-31", "base_rate",
     "SPY has closed down five days in a row. Historically, what tends to happen "
     "over the next five days after losing streaks like this?"),
    ("rbr05", "PLTR", "2024-11-12", "base_rate",
     "PLTR is up roughly 40% in a month. How often does momentum like that keep "
     "going versus mean-revert, based on history?"),
    ("rbr06", "AAPL", "2025-01-15", "base_rate",
     "AAPL has formed a textbook bull flag on the daily chart. Historically, how "
     "do setups like this tend to resolve?"),
    ("rbr07", "MARA", "2025-02-19", "base_rate",
     "A stock just printed its highest-volume up day in a year. What's the "
     "historical forward-return distribution after a day like that?"),
    ("rbr08", "META", "2025-03-26", "base_rate",
     "META just reclaimed its 200-day after months below it. What's the base "
     "rate that the reclaim holds over the next few weeks?"),
    ("rbr09", "COIN", "2025-04-23", "base_rate",
     "COIN broke out today. What is the typical max drawdown over the next 10 "
     "days after breakouts that look like this, historically?"),
    ("rbr10", "SOFI", "2025-05-21", "base_rate",
     "A small-cap squeezed 20% intraday. Historically, what's the next-day and "
     "next-week behavior after moves like that?"),
    ("rco01", "MSFT", "2025-06-18", "composite",
     "I'm holding MSFT into earnings next week. Walk me through the setup and the "
     "risk, including what history says about setups like this."),
    ("rco02", "AMD", "2025-07-23", "composite",
     "Is now a good entry on AMD after this pullback? Consider the current "
     "technical setup and the historical odds."),
    ("rco03", "NVDA", "2025-08-20", "composite",
     "Give me a full second opinion on NVDA here: the current chart, the "
     "historical base rate for this setup, and any catalysts."),
    ("rco04", "COIN", "2025-09-17", "composite",
     "I want to size a swing trade in COIN off today's breakout. What do the "
     "historical analogs say about drawdown, and what size makes sense?"),
    ("rco05", "SPY", "2025-10-22", "composite",
     "Should I trim my SPY exposure given the macro backdrop and what history "
     "says about down-streaks like the current one?"),
]

# --- FRESH: post-2026-05 anchors (all settled, < 2026-05-30). A spread of
# symbols/dates; the last few are deliberately THIN names to test weak-comp-set
# flagging. Distinct dates throughout.
_FRESH = [
    ("fbr01", "GOOGL", "2026-05-01", "base_rate",
     "GOOGL just made a new 52-week high on heavy volume. Historically, how have "
     "moves like this resolved over the next 1, 5 and 10 days?"),
    ("fbr02", "AMZN", "2026-05-04", "base_rate",
     "AMZN gapped up on a cloud-revenue beat. What's the historical base rate "
     "that gaps like this hold versus fade over the following week?"),
    ("fbr03", "QQQ", "2026-05-05", "base_rate",
     "QQQ has gone straight up for eight sessions. Historically, what tends to "
     "follow runs that extended this far this fast?"),
    ("fbr04", "NFLX", "2026-05-06", "base_rate",
     "NFLX broke out of a multi-month base. What's the typical 10-day follow-"
     "through and drawdown after base breakouts like this?"),
    ("fbr05", "AVGO", "2026-05-07", "base_rate",
     "AVGO pulled back to the 20-day after a strong trend. Historically, how "
     "often does the trend resume from a shallow pullback like this?"),
    ("fbr06", "JPM", "2026-05-08", "base_rate",
     "JPM reclaimed its 50-day on a strong up day. What's the base rate the "
     "reclaim sticks over the next two weeks, historically?"),
    ("fbr07", "XOM", "2026-05-11", "base_rate",
     "XOM is breaking out as energy rotates in. What have similar sector-rotation "
     "breakouts done over the next 5 days historically?"),
    ("fbr08", "WMT", "2026-05-12", "base_rate",
     "WMT made a quiet new high with declining volume. Historically, how do "
     "low-volume new highs tend to resolve?"),
    ("fbr09", "BAC", "2026-05-13", "base_rate",
     "BAC put in a higher-volume reversal day off support. What's the historical "
     "forward distribution after reversal days like this?"),
    ("fbr10", "DIS", "2026-05-14", "base_rate",
     "DIS gapped down then recovered the gap intraday. Historically, what tends "
     "to happen the next 5 days after a gap-and-recover?"),
    ("fbr11", "INTC", "2026-05-15", "base_rate",
     "INTC bounced hard off a multi-year low on volume. What's the historical "
     "base rate for follow-through after capitulation bounces like this?"),
    ("fbr12", "UBER", "2026-05-18", "base_rate",
     "UBER broke to a new high after a long consolidation. How do post-"
     "consolidation breakouts like this resolve historically?"),
    ("fco01", "CRM", "2026-05-19", "composite",
     "I'm eyeing CRM into earnings. Walk me through the technical setup and what "
     "history says about pre-earnings setups like this."),
    ("fco02", "SHOP", "2026-05-20", "composite",
     "Is SHOP a buy here after the breakout? Weigh the current chart against the "
     "historical odds and any catalyst."),
    ("fco03", "MU", "2026-05-21", "composite",
     "Considering a swing in MU off this momentum. What do historical analogs say "
     "about drawdown, and how should I size it?"),
    ("fco04", "SMCI", "2026-05-22", "composite",
     "SMCI is volatile here. Give me the current setup plus the historical "
     "base rate, and flag if the comparison set is thin."),
    ("fco05", "DKNG", "2026-05-26", "composite",
     "Should I add to DKNG after this pullback? Consider the chart and what "
     "history says about pullbacks like this."),
    # additional fresh anchors (earlier 2026, all settled) to reach ~50 subjects
    # and >=40 distinct dates
    ("fbr13", "GOOG", "2026-01-08", "base_rate",
     "GOOG broke out from a tight base on rising volume. Historically, how do "
     "tight-base breakouts like this resolve over 5 and 10 days?"),
    ("fbr14", "TSM", "2026-01-15", "base_rate",
     "TSM gapped up on a guidance raise. What's the historical base rate that "
     "guidance gaps like this hold rather than fade?"),
    ("fbr15", "ORCL", "2026-01-22", "base_rate",
     "ORCL made a new high after a long sideways stretch. What tends to follow "
     "breakouts from extended consolidations historically?"),
    ("fbr16", "ADBE", "2026-01-29", "base_rate",
     "ADBE failed at resistance and reversed lower on volume. What's the "
     "historical forward distribution after failed-breakout reversals?"),
    ("fbr17", "PYPL", "2026-02-05", "base_rate",
     "PYPL bounced off a long-term low. Historically, how often do bounces from "
     "multi-year lows follow through over the next two weeks?"),
    ("fbr18", "BA", "2026-02-12", "base_rate",
     "BA gapped down on a headline then stabilized. What tends to happen the "
     "following week after gap-down-and-stabilize days historically?"),
    ("fbr19", "GS", "2026-02-19", "base_rate",
     "GS pushed to a new high with broadening participation. What's the typical "
     "5-day range after new highs like this historically?"),
    ("fbr20", "CAT", "2026-02-26", "base_rate",
     "CAT reclaimed its 200-day after a long stretch below. What's the base rate "
     "the reclaim holds over the next few weeks?"),
    ("fco06", "ABNB", "2026-03-05", "composite",
     "Thinking about ABNB into its print. Walk me through the setup and what "
     "history says about pre-earnings setups like this."),
    ("fco07", "SNOW", "2026-03-12", "composite",
     "Is SNOW worth a swing here off the breakout? Weigh the current chart "
     "against the historical odds and any catalyst."),
    ("fco08", "PANW", "2026-03-19", "composite",
     "Considering PANW after this momentum push. What do historical analogs say "
     "about drawdown, and how should I size it?"),
    ("fco09", "NKE", "2026-03-26", "composite",
     "Should I add to NKE after this pullback to support? Consider the chart and "
     "what history says about pullbacks like this."),
    ("fbr21", "WDC", "2026-04-02", "base_rate",
     "WDC printed its highest-volume up day in months. What's the historical "
     "forward-return distribution after volume-thrust days like this?"),
    ("fbr22", "FCX", "2026-04-09", "base_rate",
     "FCX broke out as commodities firmed. Historically, what have commodity-"
     "driven breakouts done over the next 5 and 10 days?"),
    ("fbr23", "DAL", "2026-04-16", "base_rate",
     "DAL gapped up on a strong booking outlook. What's the base rate the gap "
     "holds versus fades over the following week historically?"),
    # deliberately THIN / edge anchors — exercise weak-comp-set flagging
    ("fth01", "RGTI", "2026-05-27", "base_rate",
     "RGTI, a thinly-traded small-cap, spiked 30% today. Historically, what "
     "follows moves like this — and is the comparison set even reliable here?"),
    ("fth02", "BBAI", "2026-05-28", "base_rate",
     "BBAI, a low-float name, broke out on a news spike. What's the historical "
     "base rate, and how confident should I be given how few analogs exist?"),
    ("fth03", "IONQ", "2026-05-29", "base_rate",
     "IONQ ran 25% on a contract headline. What do historical analogs say, and "
     "is the cohort large enough to trust?"),
]

SUBJECTS = [
    {"id": sid, "symbol": sym, "date": date, "category": cat,
     "prompt": f"{prompt} (Anchor: {sym} {date}.)", "source": src}
    for src, group in (("reused", _REUSED), ("fresh", _FRESH))
    for (sid, sym, date, cat, prompt) in group
]


def distinct_dates(subjects=None) -> int:
    subjects = SUBJECTS if subjects is None else subjects
    return len({s["date"] for s in subjects})
