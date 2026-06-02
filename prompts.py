"""
Labeled evaluation prompts for the Phase 0 selection harness.

category:
  base_rate  -> chartlibrary SHOULD be used (recall set)
  pure_ta    -> chartlibrary should NOT be used; technical_analysis should
                (the over-fire / collision negatives)
  other      -> a non-chartlibrary specialist is the right call
                (fundamentals / news / macro) -> chartlibrary should NOT fire
  composite  -> multi-tool question; chartlibrary should be among the tools used

`expects_chartlibrary` is the ground-truth label the metrics score against.
"""

PROMPTS = [
    # ---- base_rate: expects chartlibrary -------------------------------------
    {"id": "br01", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "NVDA broke out to a new all-time high today on roughly 3x average volume. "
               "Historically, how often does that kind of breakout follow through, and what's "
               "the typical range over the next 5 days?"},
    {"id": "br02", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "TSLA gapped up about 9% on earnings. What have similar earnings gaps done over "
               "the following week historically?"},
    {"id": "br03", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "AMD pulled back to its 50-day moving average after a big run. What's the "
               "historical base rate that the uptrend resumes from here?"},
    {"id": "br04", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "SPY has closed down five days in a row. Historically, what tends to happen over "
               "the next five days after losing streaks like this?"},
    {"id": "br05", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "PLTR is up roughly 40% in a month. How often does momentum like that keep going "
               "versus mean-revert, based on history?"},
    {"id": "br06", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "AAPL has formed a textbook bull flag on the daily chart. Historically, how do "
               "setups like this tend to resolve?"},
    {"id": "br07", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "A stock just printed its highest-volume up day in a year. What's the historical "
               "forward-return distribution after a day like that?"},
    {"id": "br08", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "META just reclaimed its 200-day after months below it. What's the base rate that "
               "the reclaim holds over the next few weeks?"},
    {"id": "br09", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "COIN broke out today. What is the typical max drawdown over the next 10 days "
               "after breakouts that look like this, historically?"},
    {"id": "br10", "category": "base_rate", "expects_chartlibrary": True,
     "prompt": "A small-cap squeezed 20% intraday. Historically, what's the next-day and "
               "next-week behavior after moves like that?"},

    # ---- pure_ta: must NOT call chartlibrary (collision negatives) ------------
    {"id": "ta01", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "What's AAPL's current RSI(14) and MACD reading?"},
    {"id": "ta02", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "Is TSLA trading above or below its 50-day and 200-day moving averages right now?"},
    {"id": "ta03", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "Where are the current support and resistance levels on NVDA?"},
    {"id": "ta04", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "Is SPY overbought on the daily right now?"},
    {"id": "ta05", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "What's the current Bollinger band width on AMD?"},
    {"id": "ta06", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "Give me the current ATR for MSFT."},
    {"id": "ta07", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "Has QQQ's MACD just crossed bullish or bearish?"},
    {"id": "ta08", "category": "pure_ta", "expects_chartlibrary": False,
     "prompt": "What's the 14-day RSI on GOOGL today?"},

    # ---- other: a different specialist is right; chartlibrary should NOT fire -
    {"id": "ot01", "category": "other", "expects_chartlibrary": False,
     "prompt": "What's AMZN's current P/E and year-over-year revenue growth?"},
    {"id": "ot02", "category": "other", "expects_chartlibrary": False,
     "prompt": "Any notable news catalysts for PLTR today?"},
    {"id": "ot03", "category": "other", "expects_chartlibrary": False,
     "prompt": "What's the current VIX level, and is the market risk-on or risk-off?"},
    {"id": "ot04", "category": "other", "expects_chartlibrary": False,
     "prompt": "What are AAPL's gross and operating margins?"},
    {"id": "ot05", "category": "other", "expects_chartlibrary": False,
     "prompt": "Why is GME moving on headlines today?"},
    {"id": "ot06", "category": "other", "expects_chartlibrary": False,
     "prompt": "Is the market risk-on or risk-off right now, and what does the VIX term structure say?"},

    # ---- composite: multi-tool; chartlibrary should be among them ------------
    {"id": "co01", "category": "composite", "expects_chartlibrary": True,
     "prompt": "I'm holding MSFT into earnings next week. Walk me through the setup and the risk."},
    {"id": "co02", "category": "composite", "expects_chartlibrary": True,
     "prompt": "Is now a good entry on AMD after this pullback? Consider the current technical "
               "setup and the historical odds."},
    {"id": "co03", "category": "composite", "expects_chartlibrary": True,
     "prompt": "Give me a full second opinion on NVDA here: the current chart, the historical "
               "base rate for this setup, and any catalysts."},
    {"id": "co04", "category": "composite", "expects_chartlibrary": True,
     "prompt": "I want to size a swing trade in COIN off today's breakout. What do the historical "
               "analogs say about drawdown, and what size makes sense?"},
    {"id": "co05", "category": "composite", "expects_chartlibrary": True,
     "prompt": "Should I trim my SPY exposure given the macro backdrop and what history says about "
               "down-streaks like the current one?"},
]
