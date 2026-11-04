"""CME futures contract specs for trading through Lucid Trading.

Lucid is a futures-only prop firm: it does not offer forex/CFD symbols, and it
does not connect to MetaTrader 5. This module is the futures counterpart of
instruments.py -- the static facts the strategies, the backtester and the
NinjaTrader engine need about each contract.

Sources (checked 2026-09-29):
  * Approved product list and per-side commissions: Lucid's own help centre,
    "Approved Products and Commissions" (34 CME Group contracts). Anything not
    in LUCID_APPROVED is not tradeable on a Lucid account -- notably Bitcoin,
    micro FX, Treasuries, VIX and every FX cross (GBPJPY, GBPCHF, ...).
  * Point values: CME contract sizes. These are exact.
  * Tick sizes: CME minimum fluctuations as of this writing. Used ONLY for
    backtest spread-cost estimates -- live order prices are rounded by
    NinjaTrader's own MasterInstrument.RoundToTickSize, never by this table.

Live sizing reads the point value from NinjaTrader and cross-checks it against
this table (NinjaTraderEngine.usd_per_point), the same drift gate mt5_engine
applies after the 2026-08-28 gold incident.
"""
from __future__ import annotations

from datetime import date, timedelta

# --- contract table ----------------------------------------------------------
#
# point_value:  USD P&L per contract per 1.0 move in the quoted price
# tick_size:    minimum price increment (backtest cost estimate only, see above)
# commission:   Lucid's published commission, USD PER SIDE per contract
# roll:         key into _ROLL_RULES, or None when the contract needs an
#               explicit month (agricultural cycles are not automated here)
# pip:          the price distance the pip-based strategy parameters mean on
#               this contract (see pip_size in instruments.py). For FX futures
#               quoted inversely to the spot pair (6C, 6S, 6J) this is the
#               spot pip converted to the futures quote, so a "3 pip buffer"
#               is the same size of move on either instrument.
FUTURES_SPECS: dict[str, dict] = {
    # Equity indices
    "ES":  {"point_value": 50.0,  "tick_size": 0.25, "commission": 1.75, "roll": "quarterly_index", "pip": 1.0},
    "MES": {"point_value": 5.0,   "tick_size": 0.25, "commission": 0.50, "roll": "quarterly_index", "pip": 1.0},
    "NQ":  {"point_value": 20.0,  "tick_size": 0.25, "commission": 1.75, "roll": "quarterly_index", "pip": 1.0},
    "MNQ": {"point_value": 2.0,   "tick_size": 0.25, "commission": 0.50, "roll": "quarterly_index", "pip": 1.0},
    "RTY": {"point_value": 50.0,  "tick_size": 0.10, "commission": 1.75, "roll": "quarterly_index", "pip": 1.0},
    "M2K": {"point_value": 5.0,   "tick_size": 0.10, "commission": 0.50, "roll": "quarterly_index", "pip": 1.0},
    "YM":  {"point_value": 5.0,   "tick_size": 1.0,  "commission": 1.75, "roll": "quarterly_index", "pip": 1.0},
    "MYM": {"point_value": 0.50,  "tick_size": 1.0,  "commission": 0.50, "roll": "quarterly_index", "pip": 1.0},
    "NKD": {"point_value": 5.0,   "tick_size": 5.0,  "commission": 1.75, "roll": "quarterly_nkd",   "pip": 5.0},
    # Currencies (full-size only -- Lucid excludes the micro FX contracts)
    "6A":  {"point_value": 100_000.0,    "tick_size": 0.00005,   "commission": 2.40, "roll": "quarterly_fx", "pip": 0.0001},
    "6B":  {"point_value": 62_500.0,     "tick_size": 0.0001,    "commission": 2.40, "roll": "quarterly_fx", "pip": 0.0001},
    "6C":  {"point_value": 100_000.0,    "tick_size": 0.00005,   "commission": 2.40, "roll": "quarterly_fx", "pip": 0.00005},
    "6E":  {"point_value": 125_000.0,    "tick_size": 0.00005,   "commission": 2.40, "roll": "quarterly_fx", "pip": 0.0001},
    "6J":  {"point_value": 12_500_000.0, "tick_size": 0.0000005, "commission": 2.40, "roll": "quarterly_fx", "pip": 0.0000005},
    "6S":  {"point_value": 125_000.0,    "tick_size": 0.00005,   "commission": 2.40, "roll": "quarterly_fx", "pip": 0.0001},
    "6N":  {"point_value": 100_000.0,    "tick_size": 0.00005,   "commission": 2.40, "roll": "quarterly_fx", "pip": 0.0001},
    # Energy
    "CL":  {"point_value": 1_000.0,  "tick_size": 0.01,  "commission": 2.00, "roll": "crude",  "pip": 0.01},
    "MCL": {"point_value": 100.0,    "tick_size": 0.01,  "commission": 0.50, "roll": "crude",  "pip": 0.01},
    "QM":  {"point_value": 500.0,    "tick_size": 0.025, "commission": 2.00, "roll": "crude",  "pip": 0.01},
    "NG":  {"point_value": 10_000.0, "tick_size": 0.001, "commission": 2.00, "roll": "natgas", "pip": 0.001},
    "QG":  {"point_value": 2_500.0,  "tick_size": 0.005, "commission": 1.30, "roll": "natgas", "pip": 0.001},
    # Metals
    "GC":  {"point_value": 100.0,    "tick_size": 0.10,   "commission": 2.30, "roll": "gold",     "pip": 0.10},
    "MGC": {"point_value": 10.0,     "tick_size": 0.10,   "commission": 0.80, "roll": "gold",     "pip": 0.10},
    "SI":  {"point_value": 5_000.0,  "tick_size": 0.005,  "commission": 2.30, "roll": "silver",   "pip": 0.005},
    "SIL": {"point_value": 1_000.0,  "tick_size": 0.005,  "commission": 1.60, "roll": "silver",   "pip": 0.005},
    "HG":  {"point_value": 25_000.0, "tick_size": 0.0005, "commission": 2.30, "roll": "silver",   "pip": 0.0005},
    "PL":  {"point_value": 50.0,     "tick_size": 0.10,   "commission": 2.30, "roll": "platinum", "pip": 0.10},
    # Agriculture (quoted in cents for the grains; roll month must be explicit)
    "HE":  {"point_value": 400.0, "tick_size": 0.025, "commission": 2.80, "roll": None, "pip": 0.025},
    "LE":  {"point_value": 400.0, "tick_size": 0.025, "commission": 2.80, "roll": None, "pip": 0.025},
    "ZS":  {"point_value": 50.0,  "tick_size": 0.25,  "commission": 2.80, "roll": None, "pip": 0.25},
    "ZC":  {"point_value": 50.0,  "tick_size": 0.25,  "commission": 2.80, "roll": None, "pip": 0.25},
    "ZL":  {"point_value": 600.0, "tick_size": 0.01,  "commission": 2.80, "roll": None, "pip": 0.01},
    "ZM":  {"point_value": 100.0, "tick_size": 0.10,  "commission": 2.80, "roll": None, "pip": 0.10},
    "ZW":  {"point_value": 50.0,  "tick_size": 0.25,  "commission": 2.80, "roll": None, "pip": 0.25},
}

LUCID_APPROVED = frozenset(FUTURES_SPECS)

# Lucid caps position size as "4 minis or 40 micros" (LucidPro 50K). Counting
# in micro units makes mixed books exact: a micro is 1 unit, anything else 10.
MICRO_ROOTS = frozenset({"MES", "MNQ", "M2K", "MYM", "MCL", "MGC", "SIL"})
UNITS_PER_MINI = 10

# Where each CFD the MT5 research used lives on Lucid's list:
#   (full-size root, micro root or None, quoted inversely to the spot pair)
# Inverse quoting matters for reading old results, not for trading: every
# strategy runs on the futures contract's OWN bars, so a USDCAD short signal is
# simply a 6C long signal found directly on 6C.
CFD_TO_FUTURES: dict[str, tuple[str, str | None, bool]] = {
    "EURUSD": ("6E", None, False),
    "GBPUSD": ("6B", None, False),
    "AUDUSD": ("6A", None, False),
    "NZDUSD": ("6N", None, False),
    "USDJPY": ("6J", None, True),
    "USDCAD": ("6C", None, True),
    "USDCHF": ("6S", None, True),
    "XAUUSD": ("GC", "MGC", False),
    "XAGUSD": ("SI", "SIL", False),
    "XPTUSD": ("PL", None, False),
    "COPPER": ("HG", None, False),
    "WTI":    ("CL", "MCL", False),
    "US500":  ("ES", "MES", False),
    "NAS100": ("NQ", "MNQ", False),
    "US30":   ("YM", "MYM", False),
    "JP225":  ("NKD", None, False),
}

# CFDs from the MT5 rosters with NO Lucid-approved equivalent.
NOT_ON_LUCID: dict[str, str] = {
    "BTCUSD": "crypto futures are excluded by Lucid",
    "ETHUSD": "crypto futures are excluded by Lucid",
    "DE40":   "Eurex product, not on Lucid's CME list",
    "BRENT":  "ICE product, not on Lucid's CME list",
    "GBPJPY": "FX cross -- no CME contract on Lucid's list",
    "GBPCHF": "FX cross -- no CME contract on Lucid's list",
    "EURCHF": "FX cross -- no CME contract on Lucid's list",
    "EURJPY": "FX cross -- no CME contract on Lucid's list",
    "NZDJPY": "FX cross -- no CME contract on Lucid's list",
    "AUDJPY": "FX cross -- no CME contract on Lucid's list",
    "CHFJPY": "FX cross -- no CME contract on Lucid's list",
    "AUDCHF": "FX cross -- no CME contract on Lucid's list",
    "EURAUD": "FX cross -- no CME contract on Lucid's list",
    "AUDCAD": "FX cross -- no CME contract on Lucid's list",
}


def root_of(name: str) -> str:
    """'MGC 12-26' -> 'MGC'; a bare root passes through."""
    return name.strip().split(" ")[0].upper()


def spec(name: str) -> dict:
    root = root_of(name)
    s = FUTURES_SPECS.get(root)
    if s is None:
        hint = ""
        if root in CFD_TO_FUTURES:
            hint = f" -- the Lucid equivalent of {root} is {CFD_TO_FUTURES[root][0]}"
        elif root in NOT_ON_LUCID:
            hint = f" -- {NOT_ON_LUCID[root]}"
        raise ValueError(f"{root!r} is not a Lucid-approved futures contract{hint}")
    return s


def point_value(name: str) -> float:
    return float(spec(name)["point_value"])


def commission_per_side(name: str) -> float:
    return float(spec(name)["commission"])


def units(name: str) -> int:
    """Lucid contract-cap units for ONE contract: micro = 1, anything else = 10."""
    spec(name)  # unknown contracts must fail loudly, not count as micro
    return 1 if root_of(name) in MICRO_ROOTS else UNITS_PER_MINI


def futures_for_cfd(cfd: str, prefer_micro: bool = True) -> str:
    """The Lucid contract to trade instead of an MT5 CFD symbol."""
    s = cfd.upper()
    if s in NOT_ON_LUCID:
        raise ValueError(f"{s} cannot be traded on Lucid: {NOT_ON_LUCID[s]}")
    if s not in CFD_TO_FUTURES:
        raise ValueError(f"no futures mapping for {s}")
    full, micro, _inverted = CFD_TO_FUTURES[s]
    return micro if (prefer_micro and micro) else full


# --- front-month selection ---------------------------------------------------
#
# The bot has to name a specific contract ("MGC 12-26") to place an order.
# These rules pick the contract most volume trades in, rolling several days
# BEFORE the liquidity moves rather than after -- trading the old month in its
# last days means wider spreads and thinner stops. Exchange holidays are
# ignored, which shifts a computed date by a day or two at most; the roll
# lead absorbs that. Override any contract explicitly with an engine
# symbol_map entry (e.g. {"MGC": "MGC 02-27"}) if NinjaTrader's rollover date
# disagrees.

_ROLL_RULES = {
    "quarterly_index": {"months": (3, 6, 9, 12)},
    "quarterly_fx":    {"months": (3, 6, 9, 12)},
    "quarterly_nkd":   {"months": (3, 6, 9, 12)},
    "gold":            {"months": (2, 4, 6, 8, 10, 12)},
    "silver":          {"months": (3, 5, 7, 9, 12)},
    "platinum":        {"months": (1, 4, 7, 10)},
    "crude":           {"months": tuple(range(1, 13))},
    "natgas":          {"months": tuple(range(1, 13))},
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _prev_business_day(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _business_days_before(d: date, n: int) -> date:
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _month_before(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _last_day(year: int, month: int) -> date:
    y, m = (year + 1, 1) if month == 12 else (year, month + 1)
    return date(y, m, 1) - timedelta(days=1)


def roll_date(rule: str, year: int, month: int) -> date:
    """The first day on which contract (year, month) is no longer the front."""
    if rule == "quarterly_index":
        # Expires the 3rd Friday; the market rolls 8 days earlier (2nd Thursday).
        return _nth_weekday(year, month, 4, 3) - timedelta(days=8)
    if rule == "quarterly_fx":
        # Last trade is 2 business days before the 3rd Wednesday.
        expiry = _business_days_before(_nth_weekday(year, month, 2, 3), 2)
        return expiry - timedelta(days=7)
    if rule == "quarterly_nkd":
        # Last trade is the business day before the 2nd Friday.
        expiry = _business_days_before(_nth_weekday(year, month, 4, 2), 1)
        return expiry - timedelta(days=7)
    if rule in ("gold", "silver", "platinum"):
        # First notice day is the last business day of the prior month;
        # liquidity has moved on about a week before it.
        py, pm = _month_before(year, month)
        return _prev_business_day(_last_day(py, pm)) - timedelta(days=5)
    if rule == "crude":
        # Last trade: 3 business days before the 25th of the prior month
        # (counted from the business day on/before the 25th).
        py, pm = _month_before(year, month)
        anchor = _prev_business_day(date(py, pm, 25))
        return _business_days_before(anchor, 3) - timedelta(days=7)
    if rule == "natgas":
        # Last trade: 3 business days before the 1st of the contract month.
        return _business_days_before(date(year, month, 1), 3) - timedelta(days=7)
    raise ValueError(f"unknown roll rule {rule!r}")


def front_contract(name: str, today: date) -> str:
    """NinjaTrader contract name of the front month, e.g. 'MGC 12-26'."""
    root = root_of(name)
    rule = spec(root)["roll"]
    if rule is None:
        raise ValueError(
            f"{root} has no automatic roll rule -- name the contract explicitly, "
            f"e.g. symbol_map={{'{root}': '{root} 12-26'}}")
    months = _ROLL_RULES[rule]["months"]
    y, m = today.year, today.month
    for _ in range(36):
        if m in months and today < roll_date(rule, y, m):
            return f"{root} {m:02d}-{y % 100:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    raise RuntimeError(f"no front month found for {root} from {today}")  # unreachable


def contracts_for_risk(risk_usd: float, sl_distance: float, point_value_usd: float) -> int:
    """Whole contracts risking at most `risk_usd` over `sl_distance`.
    Floors -- never rounds up; rounding up silently increases risk."""
    per_contract = sl_distance * point_value_usd
    if risk_usd <= 0 or per_contract <= 0:
        return 0
    return int(risk_usd / per_contract + 1e-9)
