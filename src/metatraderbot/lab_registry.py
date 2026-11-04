"""Strategy registry for the live laboratory: hundreds of configs, enumerated.

WHY A REGISTRY RATHER THAN A HAND-PICKED LIST
---------------------------------------------
The lab's purpose is breadth -- run many strategies forward on live data and
see what survives. That only works if the config set is generated
systematically, so nobody's prior about what "should" work quietly narrows the
search before it starts.

STATISTICAL CONSTRAINT THIS ENCODES
-----------------------------------
Detecting an edge needs trades PER STRATEGY, not trades in aggregate. Measured
against a per-trade R standard deviation of ~1.0, with a fixed budget of 2,000
trades/day split K ways:

    K=10  -> 200/day each -> +0.064R detectable after a month
    K=50  ->  40/day each -> +0.157R
    K=500 ->   4/day each -> +0.563R   (7 years to resolve a +0.05R edge)

Hence the two tiers. SHADOW configs cost nothing to run -- their outcome is
scored from subsequent bars, not from a fill -- so breadth is free there and
the count can be in the hundreds. Only the few that survive shadow are promoted
to real orders, where the budget actually binds.

Every config carries the multiple-comparison problem with it: with 400 configs
of zero true edge, roughly 20 will clear p<0.05 by chance. lab_report.py
applies Benjamini-Hochberg and refuses to rank anything until its own sample is
large enough to be readable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any


@dataclass(frozen=True)
class LabConfig:
    """One strategy instance under test.

    ``cid`` deliberately EXCLUDES the symbol. A strategy is family + params +
    timeframe + target, and it is evaluated on every symbol in the cohort with
    the results POOLED into one sample.

    That is not a convenience -- it is what makes 500 strategies testable at
    all. A strategy on a single symbol signals roughly 0.5-5 times a day, so
    splitting a trade budget 500 ways leaves ~4 trades/day each and a +0.05R
    edge takes about seven years to resolve. Pooling 12 symbols multiplies each
    strategy's sample 12-fold for the same wall-clock time, bringing that to
    months.

    It is also the right hypothesis. The claim under test is "this rule has an
    edge", not "this rule has an edge specifically on GBPJPY" -- and the latter,
    tested 12 times over, is precisely the multiple-comparison trap that
    manufactures winners out of noise.
    """
    cid: str                 # stable id -- must not change between restarts or
                             # the accumulated sample for it is orphaned
    family: str
    timeframe: str
    params: dict
    tp_r: float
    symbols: tuple           # every symbol this strategy trades; results pooled
    tier: str = "shadow"     # "shadow" (scored from bars) or "live" (real fills)

    def key(self) -> str:
        return self.cid


# Symbols confirmed fillable on this account -- see the deal-history scan in
# scratch/_trial_rework.py. Indices, WTI, Brent, copper, platinum and crypto are
# excluded: trade_mode=0 with live quotes, zero fills ever.
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
           "AUDUSD", "NZDUSD", "GBPJPY", "EURJPY", "CHFJPY", "AUDJPY"]

# The same breadth on Lucid's approved CME futures (VERTEX_BROKER=ninjatrader):
# the four equity-index micros, micro gold, micro crude and the six FX majors.
# Micros where they exist -- the lab trades nothing, and a micro's price is
# its mini's, so this is the same market at the size the bots actually trade.
FUTURES_SYMBOLS = ["MES", "MNQ", "MYM", "M2K", "MGC", "MCL",
                   "6E", "6B", "6J", "6A", "6C", "6S"]

_FAMILY_CLASSES = {
    "donchian": ("strategies_donchian", "DonchianBreakoutStrategy"),
    "macd": ("strategies_macd", "MACDTrendStrategy"),
    "rsi": ("strategies_rsi", "RSIReversionStrategy"),
    "bollinger": ("strategies_bollinger", "BollingerBounceStrategy"),
    "supertrend": ("strategies_supertrend", "SupertrendStrategy"),
    "stochastic": ("strategies_stochastic", "StochasticStrategy"),
    "ema_cross": ("strategies_ema_cross", "EMACrossoverStrategy"),
    "ma_cross": ("strategies_ma_cross", "SMACrossStrategy"),
    "orb": ("strategies_orb", "ORBStrategy"),
    "london_breakout": ("strategies_london_breakout", "LondonBreakoutStrategy"),
    "vwap": ("strategies_vwap_reversion", "VWAPReversionStrategy"),
    "sweep": ("strategies_sweep_reclaim", "SweepReclaimStrategy"),
    "tod_reversion": ("strategies_tod_reversion", "TODReversionStrategy"),
    "asian_fade": ("strategies_asian_fade", "AsianFadeStrategy"),
    "sr_retest": ("strategies_sr_retest", "SRBreakoutRetestStrategy"),
    "bos_pullback": ("strategies_bos_pullback", "TrendPullbackBOSStrategy"),
    "orb_trend": ("strategies_orb_trend", "ORBTrendStrategy"),
}


def instantiate(family: str, params: dict, tp_r: float, symbol: str,
                risk_usd: float = 100.0):
    """A strategy object for one family/params on one symbol -- the single
    place the lab, the futures research and the funded bot build strategies,
    so a config means exactly the same thing in all three."""
    import importlib
    mod_name, cls_name = _FAMILY_CLASSES[family]
    cls = getattr(importlib.import_module(f"metatraderbot.{mod_name}"), cls_name)
    return cls({"symbol": symbol,
                "capital": {"risk_usd_per_trade": risk_usd},
                "strategy": {"session": None,
                             "exit": {"tp_r_multiple": tp_r, "sl_swing_lookback": 10},
                             **params}})

# Intraday only -- this is what the lab exists to test. The 4h/1d families are
# already deployed as v7 and do not need a live bake-off.
TIMEFRAMES = ["5m", "15m", "30m", "1h"]


def _grid(family: str, spec: dict[str, list]) -> list[dict]:
    """Cartesian product of one family's parameter lists."""
    if not spec:
        return [{}]
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in product(*(spec[k] for k in keys))]


# Parameter neighbourhoods per family. Deliberately coarse: the lab is a
# screen, not a tuner, and a fine grid would multiply K without adding
# information -- each extra config costs statistical power for every other one.
FAMILIES: dict[str, dict[str, list]] = {
    "donchian":        {"period": [10, 20, 55]},
    "macd":            {"fast_period": [8, 12], "slow_period": [26]},
    "rsi":             {"period": [7, 14], "oversold": [25, 30]},
    "bollinger":       {"period": [14, 20], "std_dev": [2.0, 2.5]},
    "supertrend":      {"period": [7, 10], "multiplier": [2.0, 3.0]},
    "stochastic":      {"k_period": [9, 14]},
    "ema_cross":       {"fast_period": [9, 20], "slow_period": [50]},
    "ma_cross":        {"fast_period": [10, 20], "slow_period": [50]},
    "orb":             {"range_start_hour_utc": [7, 13], "range_minutes": [15, 30]},
    "london_breakout": {"lookback_bars": [48, 96]},
    "vwap":            {},
    "sweep":           {},
    "tod_reversion":   {},
    "asian_fade":      {},
    "sr_retest":       {"level_lookback": [20, 40]},
    "bos_pullback":    {"swing_lookback": [8, 20]},
}

TP_LEVELS = [1.0, 2.0]


def build(symbols: list[str] | None = None,
          timeframes: list[str] | None = None,
          families: list[str] | None = None,
          max_configs: int | None = None) -> list[LabConfig]:
    """Enumerate the shadow cohort.

    Ordered so that truncating with ``max_configs`` still yields a spread across
    families and symbols rather than every config of the first family -- a
    truncated-but-lopsided cohort would silently bias the whole experiment.
    """
    syms = symbols or SYMBOLS
    tfs = timeframes or TIMEFRAMES
    fams = families or list(FAMILIES)

    per_family: dict[str, list[LabConfig]] = {}
    for fam in fams:
        if fam not in FAMILIES:
            continue
        out = []
        for params in _grid(fam, FAMILIES[fam]):
            for tf, tp in product(tfs, TP_LEVELS):
                pstr = "-".join("%s%s" % (k[:3], v) for k, v in sorted(params.items()))
                cid = "%s|%s|%s|%gR" % (fam, pstr or "def", tf, tp)
                out.append(LabConfig(cid=cid, family=fam, timeframe=tf,
                                     params={fam: dict(params)}, tp_r=tp,
                                     symbols=tuple(syms)))
        per_family[fam] = out

    # round-robin across families so truncation stays balanced
    merged: list[LabConfig] = []
    i = 0
    while any(i < len(v) for v in per_family.values()):
        for fam in fams:
            v = per_family.get(fam, [])
            if i < len(v):
                merged.append(v[i])
        i += 1

    if max_configs:
        merged = merged[:max_configs]
    return merged


@dataclass
class Stat:
    """Running per-config sample. Kept small: this is serialised every poll."""
    n: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    wins: int = 0
    n_long: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    open_signals: list = field(default_factory=list)

    def add(self, r: float, is_long: bool, ts: str) -> None:
        self.n += 1
        self.sum_r += r
        self.sum_r2 += r * r
        self.wins += 1 if r > 0 else 0
        self.n_long += 1 if is_long else 0
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

    @property
    def mean(self) -> float:
        return self.sum_r / self.n if self.n else 0.0

    @property
    def sd(self) -> float:
        if self.n < 2:
            return 0.0
        v = (self.sum_r2 - self.n * self.mean ** 2) / (self.n - 1)
        return v ** 0.5 if v > 0 else 0.0

    @property
    def t(self) -> float:
        """t against zero. The only honest headline for a config under test."""
        if self.n < 2 or self.sd <= 0:
            return 0.0
        return self.mean / (self.sd / self.n ** 0.5)
