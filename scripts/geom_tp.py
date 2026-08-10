import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
ROOT = os.path.join(os.path.dirname(__file__), "..")
import pandas as pd
from metatraderbot.backtest import Backtest
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy

SYMS = [("USDJPY", "30m", 100), ("US30", "30m", 100), ("EURAUD", "30m", 100), ("XAUUSD", "30m", 30)]


class W:
    def __init__(s, i): s.i = i
    def on_bar(s, i, b):
        st = max(0, i - 349); v = b.iloc[st:i + 1].reset_index(drop=True); return s.i.on_bar(len(v) - 1, v)


def load(sym, tf):
    f = os.path.join(ROOT, "data", f"{sym.lower()}_15m.parquet")
    if not os.path.exists(f): return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0]); df.index = pd.to_datetime(df.index, utc=True)
    if tf == "30m":
        df = df.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return df.tail(6000)


def cfg(sym, risk, frac, tp):
    d = {"period": 10}
    if frac < 1.0: d.update({"sl_mode": "channel_frac", "sl_channel_frac": frac})
    return {"symbol": sym, "capital": {"risk_usd_per_trade": risk, "initial_usd": 10000, "max_positions": 2,
            "daily_stop_loss_usd": -1e9, "trail_start_r": 1.0, "trail_lock_fraction": 0.5},
            "strategy": {"donchian": d, "exit": {"tp_r_multiple": tp}, "session": None, "trend": {},
                         "signal": {}, "entry": {}, "bollinger": {}, "orb": {}}}


rows = []
for sym, tf, risk in SYMS:
    b = load(sym, tf)
    if b is None: print(f"[skip]{sym}", flush=True); continue
    v = b.iloc[int(len(b) * 0.70):]
    print(f"\n==== {sym} {tf} validation n={len(v)} ====", flush=True)
    print(f"{'stop':10}{'TP':>5}{'trades':>8}{'WR%':>7}{'exp$':>9}{'PF':>7}{'pnl$':>10}{'maxDD':>10}", flush=True)
    for fr, fn in [(1.0, "channel"), (0.5, "chan0.5")]:
        for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            c = cfg(sym, risk, fr, tp); bt = Backtest(c, v, strategy=W(DonchianBreakoutStrategy(c))); bt.run(); r = bt.report()
            w = sum(t.pnl_usd for t in bt.closed if t.pnl_usd > 0); l = -sum(t.pnl_usd for t in bt.closed if t.pnl_usd < 0)
            pf = (w / l) if l > 0 else 0.0
            rows.append((sym, fn, tp, r["trades"], r["win_rate_pct"], r["expectancy_usd_per_trade"], pf, r["total_pnl_usd"], r["max_drawdown_usd"]))
            print(f"{fn:10}{tp:5.1f}{r['trades']:8d}{r['win_rate_pct']:7.1f}{r['expectancy_usd_per_trade']:9.2f}{pf:7.2f}{r['total_pnl_usd']:10.2f}{r['max_drawdown_usd']:10.2f}", flush=True)
pd.DataFrame(rows, columns=["symbol", "stop", "tp", "trades", "wr", "exp", "pf", "pnl", "dd"]).to_csv(os.path.join(ROOT, "backtest", "geometry_sweep.csv"), index=False)
print("\nDONE", flush=True)
