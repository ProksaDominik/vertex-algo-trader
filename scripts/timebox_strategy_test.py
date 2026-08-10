"""FAST-RESOLUTION test: does a hard time stop keep an edge alive?

THE PROBLEM
-----------
An FTMO 10-day challenge needs +10% inside ~10 trading days. The only two
strategies in this project that survived the train/OOS gate (RSI(14) 30/70 and
Bollinger(20,2.0), both on DAILY bars) hold for a MEDIAN of 10-18 days. In a
10-day window barely one trade matures, so the distribution of outcomes is
dominated by "did the single open trade happen to be up on day 10", not by the
strategy's expectancy. Speed of resolution, not maximum expectancy, is the
binding constraint.

HYPOTHESES TESTED HERE
----------------------
A. TIME-STOPPED DAILY MEAN REVERSION (the high-value test).
   Take RSI and Bollinger exactly as they passed validation, add a hard
   max-hold of N = 1, 2, 3, 5 bars (daily bars, Sundays dropped, so N bars ~ N
   trading days), and force a mark-to-market close at that bar's close.
   N = None is the unmodified control.
   Question: do they still have positive expectancy in BOTH halves?

B. SESSION BREAKOUT WITH FORCED END-OF-DAY CLOSE (hold < 1 day by construction).
   Measure a session range (London variant: 00:00-07:00 UTC "Asian range";
   NY variant: 07:00-13:00 UTC "London range"), trade the first hourly close
   that breaks it inside the following session, and force-close at the end of
   the day. Two stop geometries (opposite side of range / half range) x two
   targets (1R, 2R).

NO-LOOKAHEAD DISCIPLINE
-----------------------
A previous "stop after a losing day" rule in this repo was lookahead: it used a
trade's final outcome to gate later entries. Everything here is causal:
  - the time stop reads only (current bar index - entry bar index);
  - the session range is CLOSED before the trade window opens;
  - the breakout triggers on the CURRENT bar's close, entry at that close;
  - EOD close fires on the first bar with hour >= eod_hour, or the first bar
    whose calendar date differs from the entry date - both knowable live.

HOW THE TIME STOP IS IMPLEMENTED
--------------------------------
NOT by post-processing the trade list. `ForcedExitBacktest` subclasses
`metatraderbot.backtest.Backtest` and appends a forced-exit pass AFTER the
engine's own SL/TP/trail pass on each bar. That ordering matters: on the final
permitted bar, an SL or TP touched intrabar still wins over the time stop
(conservative), and because the position is genuinely removed from
`self.open`, the freed `max_positions` slot changes which later signals are
taken - exactly as it would live. Post-processing would have missed that.
The forced exit fills at the bar CLOSE and pays the same spread+commission the
engine charges every other exit.

=============================== RESULT ===============================
TIME STOPS DESTROY THE EDGE. Full numbers in the tables this prints, but:

  * RSI and Bollinger only survive both halves at LONG targets (3R-5R) with NO
    time stop. Every N in {1,2,3,5} that was tested turns at least one half
    negative, or turns positive-in-both only at a target so small that the
    cost drag eats it. There is no (target, N) cell that is positive in both
    halves AND has a median hold under 3 days AND enough trades to matter.
  * The mechanism is visible in the exit mix: with N<=3 the great majority of
    trades exit "timestop" at a small random mark-to-market, so the strategy
    stops being "small loss / big win" and becomes a coin flip minus costs.
    The 1R-target variants keep a decent win rate but each stop-out costs
    ~1.05R, so they cannot pay for themselves.
  * Session breakouts with an EOD close resolve fast (median hold ~0.3-0.5
    days, as designed) and are NEGATIVE out of sample on every variant tested.
    Intraday stops are 3-8x tighter than daily swing stops, so the same fixed
    spread+commission costs 3-8x more in R terms - reported here as cost_R.

CONCLUSION: fast resolution and positive expectancy are not simultaneously
available in this data with these strategies. Do not run a 10-day FTMO
challenge on a time-stopped version of the daily roster.

Usage:
    python scripts/timebox_strategy_test.py                 # both parts
    python scripts/timebox_strategy_test.py --part A
    python scripts/timebox_strategy_test.py --part B --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest, ClosedPosition  # noqa: E402
from metatraderbot.instruments import contract_mult  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategy import Direction, TradeSpec  # noqa: E402

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]

# --------------------------------------------------------------------------
# Engine extension: hard forced exits
# --------------------------------------------------------------------------


class ForcedExitBacktest(Backtest):
    """`Backtest` plus two causal forced exits, applied after SL/TP/trail.

    max_hold_bars : close at the close of the bar `max_hold_bars` bars after
                    the entry bar (None = disabled).
    eod_hour      : close at the close of the first bar whose UTC hour is
                    >= eod_hour, or whose calendar date is past the entry
                    date (None = disabled). Only meaningful intraday.

    Both read exclusively from the current bar and the entry bar, so a live
    runner could reproduce every exit without knowing the future.
    """

    def __init__(self, config, bars, strategy=None, max_hold_bars=None,
                 eod_hour=None):
        super().__init__(config, bars, strategy)
        self.max_hold_bars = max_hold_bars
        self.eod_hour = eod_hour
        # timestamp -> row index, so "bars held" is bar count, not calendar days
        self._row_of_ts = {
            pd.Timestamp(t): i for i, t in enumerate(self.bars["timestamp"])
        }

    def _force_close(self, pos, bar, bar_ts, reason: str) -> None:
        px = float(bar["close"])
        pnl = self._raw_pnl(pos, px) - self._costs(pos.lots)
        self.equity += pnl
        self.closed.append(ClosedPosition(
            direction=pos.direction, entry_price=pos.entry_price,
            exit_price=px, stop_loss=pos.stop_loss,
            take_profit=pos.take_profit, lots=pos.lots,
            entered_at=pos.entered_at, exited_at=bar_ts,
            exit_reason=reason, pnl_usd=pnl,
        ))
        d = bar_ts.date()
        self.daily_pnl[d] = self.daily_pnl.get(d, 0.0) + pnl

    def _check_exits(self, bar, bar_ts, avg_range: float = 0.0):
        # SL / TP / trail / skyrocket first: an intrabar SL or TP on the final
        # permitted bar takes precedence over the forced exit (conservative).
        super()._check_exits(bar, bar_ts, avg_range)
        if not self.open or (self.max_hold_bars is None and self.eod_hour is None):
            return

        i_now = self._row_of_ts.get(bar_ts)
        still = []
        for pos in self.open:
            reason = None
            if self.max_hold_bars is not None and i_now is not None:
                i_in = self._row_of_ts.get(pd.Timestamp(pos.entered_at))
                if i_in is not None and (i_now - i_in) >= self.max_hold_bars:
                    reason = "timestop"
            if reason is None and self.eod_hour is not None:
                ts_in = pd.Timestamp(pos.entered_at)
                if bar_ts.hour >= self.eod_hour or bar_ts.date() > ts_in.date():
                    reason = "eod"
            if reason is None:
                still.append(pos)
            else:
                self._force_close(pos, bar, bar_ts, reason)
        self.open = still


# --------------------------------------------------------------------------
# Session breakout strategy (hypotheses B1 / B2)
# --------------------------------------------------------------------------


class SessionBreakoutStrategy:
    """Range-of-previous-session breakout, one shot per day, EOD exit.

    Parameters (config["strategy"]["session_breakout"]):
        range_start_h / range_end_h : UTC hours [start, end) forming the range.
        trade_start_h / trade_end_h : UTC hours [start, end) in which a break
                                      may be taken. trade_start_h must be >=
                                      range_end_h so the range is closed.
        sl_mode : "range"  -> stop at the opposite side of the range
                  "half"   -> stop at half the range width from entry
    Entry is the CURRENT bar's close on the first bar that closes outside the
    range. One trade per calendar day per symbol.
    """

    def __init__(self, config: dict):
        s = config["strategy"]
        p = s.get("session_breakout") or {}
        self.range_start_h = int(p.get("range_start_h", 0))
        self.range_end_h = int(p.get("range_end_h", 7))
        self.trade_start_h = int(p.get("trade_start_h", 7))
        self.trade_end_h = int(p.get("trade_end_h", 15))
        self.sl_mode = str(p.get("sl_mode", "range"))
        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self._last_signal_date = None

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        # Cheap rejection first: this runs on every hourly bar of 10.5 years.
        cur = bars.iloc[bar_index]
        ts = pd.Timestamp(cur["timestamp"])
        if not (self.trade_start_h <= ts.hour < self.trade_end_h):
            return []
        today = ts.date()
        if self._last_signal_date == today:
            return []

        # Today's bars only, bounded lookback (<= 36 hourly bars back).
        lo = max(0, bar_index - 36)
        win = bars.iloc[lo:bar_index + 1]
        wts = pd.to_datetime(win["timestamp"])
        same_day = wts.dt.date.values == today
        hours = wts.dt.hour.values
        in_range = same_day & (hours >= self.range_start_h) & (hours < self.range_end_h)
        if in_range.sum() < 3:          # need a real session, not a stub
            return []

        r_hi = float(win["high"].values[in_range].max())
        r_lo = float(win["low"].values[in_range].min())
        width = r_hi - r_lo
        if width <= 0:
            return []

        entry = float(cur["close"])
        if entry > r_hi:
            direction = Direction.LONG
            sl = r_lo if self.sl_mode == "range" else entry - width * 0.5
            sl_distance = entry - sl
        elif entry < r_lo:
            direction = Direction.SHORT
            sl = r_hi if self.sl_mode == "range" else entry + width * 0.5
            sl_distance = sl - entry
        else:
            return []

        if sl_distance <= 0:
            return []
        lots = self.risk_usd / (sl_distance * self.contract_mult)
        if lots <= 0:
            return []
        tp = entry + sl_distance * self.tp_r_multiple if direction == Direction.LONG \
            else entry - sl_distance * self.tp_r_multiple
        self._last_signal_date = today
        return [TradeSpec(direction=direction, entry_price=entry, stop_loss=sl,
                          take_profit=tp, lots=lots, signal_bar_index=bar_index)]


# --------------------------------------------------------------------------
# Cell runners
# --------------------------------------------------------------------------

DAILY_STRATS = {
    "RSI":       (RSIReversionStrategy,
                  {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    "Bollinger": (BollingerBounceStrategy, {"bollinger": {}}),
}
DAILY_TPS = [1.0, 2.0, 3.0, 4.0, 5.0]
DAILY_HOLDS = [1, 2, 3, 5, None]        # None = unmodified control

SESSION_VARIANTS = {
    #  label            range      trade window
    "London_ORB": {"range_start_h": 0, "range_end_h": 7,
                   "trade_start_h": 7, "trade_end_h": 15},
    "NY_ORB":     {"range_start_h": 7, "range_end_h": 13,
                   "trade_start_h": 13, "trade_end_h": 19},
}
SESSION_SL_MODES = ["range", "half"]
SESSION_TPS = [1.0, 2.0]
EOD_HOUR = 20                            # last full hourly bar of the FX day


def _trade_rows(bt) -> list[dict]:
    """Per-trade (R, hold in calendar days, exit reason, cost as fraction of R)."""
    rows = []
    for t in bt.closed:
        hold_d = (pd.Timestamp(t.exited_at) - pd.Timestamp(t.entered_at)) / pd.Timedelta(days=1)
        cost = bt._costs(t.lots)
        rows.append({"r": t.pnl_usd / RISK_USD, "hold": float(hold_d),
                     "why": t.exit_reason, "cost_r": cost / RISK_USD})
    return rows


def _base_cfg(sym: str, params: dict, tp_r: float, max_positions: int = 2) -> dict:
    strat = {"session": None,
             "exit": {"tp_r_multiple": tp_r, "sl_swing_lookback": 10}}
    strat.update(params)
    return {"symbol": sym,
            "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                        "max_positions": max_positions,
                        "daily_stop_loss_usd": -1e9},
            "strategy": strat}


def _cell_daily(job):
    """(strategy, tp_r, max_hold_bars, symbol) on 1d bars, train + OOS."""
    name, tp_r, hold, sym = job
    cls, params = DAILY_STRATS[name]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]      # partial Sunday bar
        split = int(len(bars) * 0.70)
        cfg = _base_cfg(sym, params, tp_r)
        out = {"kind": "daily", "name": name, "tp_r": tp_r, "hold": hold, "sym": sym}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 250:
                continue
            bt = ForcedExitBacktest(cfg, b, strategy=Windowed(cls(cfg)),
                                    max_hold_bars=hold)
            bt.run()
            out[half] = {"trades": _trade_rows(bt),
                         "days": len(set(b.index.date))}
        return out
    except Exception as e:                          # noqa: BLE001
        return {"kind": "daily", "name": name, "tp_r": tp_r, "hold": hold,
                "sym": sym, "error": repr(e)}


def _cell_session(job):
    """(variant, sl_mode, tp_r, symbol) on 1h bars with a forced EOD close."""
    variant, sl_mode, tp_r, sym = job
    try:
        bars = load(sym, "1h")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        split = int(len(bars) * 0.70)
        params = {"session_breakout": {**SESSION_VARIANTS[variant], "sl_mode": sl_mode}}
        cfg = _base_cfg(sym, params, tp_r, max_positions=1)
        out = {"kind": "session", "name": f"{variant}/{sl_mode}", "tp_r": tp_r,
               "hold": "EOD", "sym": sym}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 2000:
                continue
            # No Windowed wrapper: the strategy already bounds its own lookback
            # to <= 36 bars, and wrapping would slice 400 rows on every one of
            # ~45k hourly bars for no benefit.
            bt = ForcedExitBacktest(cfg, b, strategy=SessionBreakoutStrategy(cfg),
                                    eod_hour=EOD_HOUR)
            bt.run()
            out[half] = {"trades": _trade_rows(bt),
                         "days": len(set(b.index.date))}
        return out
    except Exception as e:                          # noqa: BLE001
        return {"kind": "session", "name": f"{variant}/{sl_mode}", "tp_r": tp_r,
                "hold": "EOD", "sym": sym, "error": repr(e)}


# --------------------------------------------------------------------------
# Aggregation / reporting
# --------------------------------------------------------------------------


def agg(cells, half):
    trades = [t for c in cells for t in c.get(half, {}).get("trades", [])]
    days = max((c.get(half, {}).get("days", 0) for c in cells), default=0)
    if not trades:
        return None
    rs = [t["r"] for t in trades]
    holds = [t["hold"] for t in trades]
    wins = [x for x in rs if x > 0]
    gl = -sum(x for x in rs if x <= 0)
    why: dict[str, int] = {}
    for t in trades:
        why[t["why"]] = why.get(t["why"], 0) + 1
    return {
        "n": len(rs),
        "per_day": len(rs) / days if days else 0.0,
        "win": len(wins) / len(rs),
        "e": float(np.mean(rs)),
        "total": float(sum(rs)),
        "pf": (sum(wins) / gl) if gl > 0 else float("inf"),
        "med_hold": float(np.median(holds)),
        "cost_r": float(np.mean([t["cost_r"] for t in trades])),
        "why": {k: round(v / len(rs), 3) for k, v in
                sorted(why.items(), key=lambda kv: -kv[1])},
    }


HDR = (f"{'strategy':<18}{'TP':>5}{'hold':>6}{'n':>7}{'/day':>7}{'win%':>7}"
       f"{'E_tr':>9}{'E_oos':>9}{'PF_oos':>8}{'medDay':>8}{'cost_R':>8}  verdict")


def _row(label, tp, hold, t, o) -> str:
    both = bool(o and t and o["e"] > 0 and t["e"] > 0)
    verdict = "SURVIVOR" if both else ("oos+ only" if o and o["e"] > 0 else
                                       ("train+ only" if t and t["e"] > 0 else "FAIL"))
    pf = "-" if not o else (f"{o['pf']:.2f}" if o["pf"] != float("inf") else "inf")
    return (f"{label:<18}{tp:>5.1f}{str(hold):>6}"
            f"{(o['n'] if o else 0):>7}{(o['per_day'] if o else 0):>7.2f}"
            f"{(o['win'] * 100 if o else 0):>7.1f}"
            f"{(t['e'] if t else float('nan')):>+9.3f}"
            f"{(o['e'] if o else float('nan')):>+9.3f}{pf:>8}"
            f"{(o['med_hold'] if o else float('nan')):>8.2f}"
            f"{(o['cost_r'] if o else float('nan')):>8.3f}  {verdict}")


def run_part(part: str, jobs, fn, workers: int) -> list[dict]:
    print(f"\nRunning {len(jobs)} part-{part} cells on {workers} workers...", flush=True)
    cells, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in ex.map(fn, jobs, chunksize=4):
            done += 1
            if c and not c.get("error"):
                cells.append(c)
            elif c and c.get("error") and done < 6:
                print(f"  [ERROR] {c['name']} {c['sym']}: {c['error']}", flush=True)
            if done % 100 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}", flush=True)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    workers = min(args.workers, os.cpu_count() or 4)

    print("=" * 118)
    print("FAST-RESOLUTION TEST — hard time stops on the validated daily edge, "
          "and session breakouts with a forced EOD close")
    print(f"{len(SYMBOLS)} symbols   train = first 70%   OOS = last 30%   "
          f"1R = ${RISK_USD:.0f}   Sundays excluded")
    print("=" * 118)

    results: list[dict] = []

    if args.part in ("A", "both"):
        jobs = [(n, tp, h, s) for n in DAILY_STRATS for tp in DAILY_TPS
                for h in DAILY_HOLDS for s in SYMBOLS]
        cells = run_part("A", jobs, _cell_daily, workers)
        print("\n" + "=" * 118)
        print("PART A — DAILY RSI / BOLLINGER WITH A HARD MAX-HOLD "
              "(hold = bars, ~trading days; None = unmodified control)")
        print("=" * 118)
        print(HDR)
        print("-" * 118)
        for n in DAILY_STRATS:
            for tp in DAILY_TPS:
                for h in DAILY_HOLDS:
                    mine = [c for c in cells if c["name"] == n and c["tp_r"] == tp
                            and c["hold"] == h]
                    t, o = agg(mine, "train"), agg(mine, "oos")
                    if not o:
                        continue
                    print(_row(n, tp, h, t, o))
                    results.append({"part": "A", "name": n, "tp_r": tp,
                                    "hold": h, "train": t, "oos": o,
                                    "both_positive": bool(t and o and t["e"] > 0
                                                          and o["e"] > 0)})
                print()
        # Exit mix makes the mechanism visible.
        print("exit-reason mix (OOS, fraction of trades):")
        for r in results:
            if r["part"] == "A" and r["tp_r"] in (1.0, 3.0):
                print(f"  {r['name']:<10} TP {r['tp_r']:<4} hold "
                      f"{str(r['hold']):<5} {r['oos']['why']}")

    if args.part in ("B", "both"):
        jobs = [(v, m, tp, s) for v in SESSION_VARIANTS for m in SESSION_SL_MODES
                for tp in SESSION_TPS for s in SYMBOLS]
        cells = run_part("B", jobs, _cell_session, workers)
        print("\n" + "=" * 118)
        print(f"PART B — SESSION BREAKOUT, FORCED CLOSE AT {EOD_HOUR}:00 UTC "
              "(1h bars, one shot per day)")
        print("=" * 118)
        print(HDR)
        print("-" * 118)
        for v in SESSION_VARIANTS:
            for m in SESSION_SL_MODES:
                for tp in SESSION_TPS:
                    label = f"{v}/{m}"
                    mine = [c for c in cells if c["name"] == label and c["tp_r"] == tp]
                    t, o = agg(mine, "train"), agg(mine, "oos")
                    if not o:
                        continue
                    print(_row(label, tp, "EOD", t, o))
                    results.append({"part": "B", "name": label, "tp_r": tp,
                                    "hold": "EOD", "train": t, "oos": o,
                                    "both_positive": bool(t and o and t["e"] > 0
                                                          and o["e"] > 0)})
            print()

    print("=" * 118)
    print("SURVIVORS — positive expectancy in BOTH halves, n>=100 OOS, "
          "median hold < 3 days")
    print("=" * 118)
    keep = [r for r in results if r["both_positive"] and r["oos"]["n"] >= 100
            and r["oos"]["med_hold"] < 3.0]
    if not keep:
        print("  NONE. No configuration is fast AND profitable in both halves.")
        slow = [r for r in results if r["both_positive"] and r["oos"]["n"] >= 100]
        if slow:
            print("\n  For contrast, configurations that pass both halves but are "
                  "TOO SLOW for a 10-day challenge:")
            for r in sorted(slow, key=lambda x: -x["oos"]["e"]):
                print(f"    {r['name']:<18} TP {r['tp_r']:<4} hold "
                      f"{str(r['hold']):<5} E_oos={r['oos']['e']:+.3f}R  "
                      f"median hold {r['oos']['med_hold']:.1f} days")
    else:
        for r in sorted(keep, key=lambda x: -x["oos"]["e"]):
            print(f"  {r['name']:<18} TP {r['tp_r']:<4} hold {str(r['hold']):<5} "
                  f"E_train={r['train']['e']:+.3f}R  E_oos={r['oos']['e']:+.3f}R  "
                  f"{r['oos']['per_day']:.2f}/day  median hold "
                  f"{r['oos']['med_hold']:.2f} days")

    out = ROOT / "backtest" / "timebox_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {len(results)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
