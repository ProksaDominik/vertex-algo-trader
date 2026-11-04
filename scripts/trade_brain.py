"""Trade brain: the bots' long-term memory, written to the owner's Obsidian vault (2026-10-04).

Owner's design: "a ledger that shows what happened, why and when, and logs every
single trade; and a learning file: when a trade loses it analyses it, finds the
mistake and never makes it again -- a brain that learns over time."

  notes/Trade ledger.md  every trade (trial + funded): when, what, why, how it went,
                         the market around it, execution checks.
  notes/Learning.md      every loss analysed. Two kinds, kept apart on purpose:
     * EXECUTION mistakes -- late / missed entry, an order that lost its protection,
       a lost trade record, a simulator fill Lucid would not have given. Real errors:
       each gets its fix and goes on the "never again" list.
     * MARKET losses -- the race loses ~6 in 10 even when everything works. Their
       conditions (Monday, VIX rising, sellers yesterday, high volatility, ...) are
       counted, and each condition is checked against the race's history (real
       NinjaTrader bars 2021-2026). A condition becomes a rule only when history
       confirms it (z <= -2); until then it is a hypothesis -- one unlucky day must
       not teach the bot a superstition. Confirmed lessons are candidates for the
       prediction lab, never switched on blindly.
  data/brain.json        the same memory, machine-readable.

Runs after every trading day from run_prediction_lab.py, or by hand:
    .venv/Scripts/python.exe scripts/trade_brain.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402
import lucid_vix_tick_search as vt  # noqa: E402
from metatraderbot import futures, lucid_rules  # noqa: E402

VAULT = ROOT.parents[1]
LEDGER_NOTE = VAULT / "notes" / "Trade ledger.md"
LEARN_NOTE = VAULT / "notes" / "Learning.md"
MEMORY = ROOT / "data" / "brain.json"
F = ROOT / "data" / "futures"
ET = lucid_rules.ET
PRAGUE = "Europe/Prague"
BOTS = {"Trial": ROOT / "data" / "lucid_futures_trades.jsonl",
        "Funded": ROOT / "data" / "lucid_futures_funded_trades.jsonl"}

# Execution mistakes the bots have made, how each is detected, and its permanent fix.
FIXES = {
    "late_entry": ("Late or missed start", "The race went in long after 19:00 New York because the bot or "
                   "the laptop was not running.", "Watchdog restarts anything stopped every 15 min, and the PC "
                   "wakes at 00:55 Mon-Fri when plugged in (2026-10-04)."),
    "protection_failed": ("Order lost its protection", "NinjaTrader could not place the take-profit / safety "
                          "stop and the trade was closed at once.", "Bridge 1.2 (press F5 once in NinjaTrader) and "
                          "an automatic retry, up to 3 times a day (2026-10-01)."),
    "not_watching": ("Bot not watching", "The outcome happened while the laptop slept; it was booked from the "
                     "price history.", "History replay books what Lucid would have done (2026-10-01); sleep no "
                     "longer stops the bots (2026-10-03)."),
    "lost_record": ("Trade record lost", "NinjaTrader restarted and no longer knew the trade.", "Lost trades are "
                    "settled from the price history (2026-10-03)."),
    "sim_vs_lucid": ("Simulator ≠ Lucid", "The simulator's fill differed from what Lucid would have done "
                     "(e.g. take-profit filled after the fail line was touched).", "Every bracket fill is "
                     "re-checked against the price history (2026-10-03)."),
}
TAGS = {   # condition at entry -> plain name
    "monday": "Monday", "event": "Fed / CPI / jobs-report day", "vix_up": "VIX rose yesterday",
    "vix_high": "VIX above 20", "sellers": "sellers dominated yesterday (NYSE TICK)",
    "high_vol": "high volatility (top third of the past year)", "drop_before": "NQ fell > 0.3% between 16:00 and entry",
    "below_avg": "NQ below its 20-day average", "fell_yesterday": "NQ fell yesterday (09:30-16:00)",
}


# --- data ----------------------------------------------------------------------

def _bars() -> pd.DataFrame:
    for name in ("lab_NQ_15m.parquet", "NQ_15m_long_nt.parquet"):
        if (F / name).exists():
            df = pd.read_parquet(F / name)
            df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
            return df.sort_index()
    raise FileNotFoundError("no NQ 15-minute bars")


def _bars_1m() -> pd.DataFrame | None:
    """1-minute NQ bars for the best / worst move inside each trade: fresh from NinjaTrader
    (read-only) merged into a cache, so older trades keep their minutes."""
    cache = F / "brain_NQ_1m.parquet"
    old = pd.read_parquet(cache) if cache.exists() else None
    try:
        from metatraderbot import free_prices
        if free_prices.in_use():                                          # paper mode: the internet's minutes
            new = free_prices.bars("1m", 12_000, "NQ").iloc[:-1]
        else:
            from metatraderbot.nt_engine import NinjaTraderEngine
            e = NinjaTraderEngine(account=os.environ.get("NT_ACCOUNT", "Sim101"), timeout_s=60.0)
            new = e.get_bars("NQ", "1m", 12_000).iloc[:-1]              # drop the forming stand-in
        new.index = pd.DatetimeIndex(new.index).tz_convert("UTC")
        df = new if old is None else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(cache)
        return df
    except Exception:  # noqa: BLE001 -- NinjaTrader closed: use what is cached
        return old


def _refresh_vix() -> None:
    """Keep the daily VIX current (read-only bridge call); silently skipped without NinjaTrader."""
    try:
        from metatraderbot import free_prices
        if free_prices.in_use():                                          # paper mode: the internet's VIX
            free_prices.bars("1d", 1500, "^VIX").to_parquet(F / "VIX_1d_nt.parquet")
            return
        from metatraderbot.nt_engine import NinjaTraderEngine
        e = NinjaTraderEngine(account=os.environ.get("NT_ACCOUNT", "Sim101"), timeout_s=30.0)
        d = e._call("bars", instrument="^VIX", minutes=1440, count=1500)
        df = pd.DataFrame(d.get("bars") or [], columns=["t", "open", "high", "low", "close", "volume"])
        if len(df) > 100:
            df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df.drop(columns="t").set_index("timestamp").to_parquet(F / "VIX_1d_nt.parquet")
    except Exception:  # noqa: BLE001
        pass


def _tick_session() -> pd.Series:
    parts = []
    for name in ("TICK_15m", "lab_TICK_15m"):
        p = F / (f"{name}_nt.parquet" if not name.startswith("lab") else f"{name}.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
            parts.append(df)
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    s = vt.rth_15m_stats(df)["session"]
    s.index = pd.DatetimeIndex(s.index)
    return s


class Context:
    """Market conditions known BEFORE each Lucid trading day's race (same definitions as the research)."""

    def __init__(self):
        _refresh_vix()
        self.g = lp.load_grid("NQ", path=(F / "lab_NQ_15m.parquet") if (F / "lab_NQ_15m.parquet").exists()
                              else F / "NQ_15m_long_nt.parquet")
        self.idx = pd.to_datetime(self.g.dates)
        d = self.g.daily.reindex(self.idx)
        vix = vt.load("VIX_1d")["close"] if (F / "VIX_1d_nt.parquet").exists() else pd.Series(dtype=float)
        vix_prev = vt.known_before(vix, self.idx)
        vix_chg = vt.known_before(vix.pct_change(), self.idx)
        tick = vt.known_before(_tick_session(), self.idx)
        atr_pct = d["atr14"] / d["prev_close"]
        vol_rank = atr_pct.rolling(252, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
        ev = json.loads((F / "event_dates.json").read_text(encoding="utf-8"))
        evd = pd.DatetimeIndex([x + pd.offsets.BDay(0) for k in ("fomc", "cpi", "nfp") for x in pd.to_datetime(ev[k])])
        move_19 = pd.Series(self.g.O[:, lp.S(19)], index=self.idx) / d["prev_close"] - 1
        self.frame = pd.DataFrame({
            "vix": vix_prev, "vix_chg": vix_chg, "tick": tick, "atr_pct": atr_pct, "vol_rank": vol_rank,
            "prev_ret": d["prev_rth_ret"], "below_avg": d["prev_close"] < d["sma20"],
            "event": pd.Series(self.idx, index=self.idx).isin(evd), "move_19": move_19,
            "dow": self.idx.dayofweek}, index=self.idx)
        self.tags = pd.DataFrame({
            "monday": self.frame["dow"] == 0, "event": self.frame["event"],
            "vix_up": self.frame["vix_chg"] > 0, "vix_high": self.frame["vix"] > 20,
            "sellers": self.frame["tick"] < 0, "high_vol": self.frame["vol_rank"] >= 2 / 3,
            "drop_before": self.frame["move_19"] < -0.003, "below_avg": self.frame["below_avg"],
            "fell_yesterday": self.frame["prev_ret"] < 0}, index=self.idx)
        lab = ss.race_labels(self.g, lp.S(19))
        self.race = pd.Series(lab[1], index=self.idx)                 # the deployed race's own outcome

    def at(self, day: str) -> dict:
        d = pd.Timestamp(day)
        if d not in self.frame.index:
            return {}
        r = self.frame.loc[d]
        out = {k: (None if pd.isna(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
               for k, v in r.items()}
        out["tags"] = [k for k in TAGS if bool(self.tags.loc[d, k]) is True]
        return out

    def history(self, tag: str) -> dict:
        """Race win rate on days with / without the condition (resolved days, 2021-2026 real bars)."""
        t = self.tags[tag].fillna(False).to_numpy(bool)
        y = self.race.to_numpy()
        ok = ~np.isnan(y)
        w1, n1 = float(np.mean(y[ok & t])) if (ok & t).any() else float("nan"), int((ok & t).sum())
        w0, n0 = float(np.mean(y[ok & ~t])) if (ok & ~t).any() else float("nan"), int((ok & ~t).sum())
        p = float(np.mean(y[ok]))
        z = (w1 - p) / math.sqrt(p * (1 - p) / n1) if n1 else float("nan")
        return {"with": w1, "n_with": n1, "without": w0, "n_without": n0, "z": z}


# --- trades ---------------------------------------------------------------------

def load_trades() -> list[dict]:
    rows = []
    paths = [(bot, path) for bot, path in BOTS.items()]
    # every further funded account keeps its own trade log (run_lucid_futures_funded.py --account)
    paths += [("Funded", p) for p in sorted(BOTS["Funded"].parent.glob("lucid_futures_funded_trades_*.jsonl"))]
    for bot, path in paths:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                r = json.loads(line)
                r["bot"] = bot
                r["key"] = f"{bot}|{r['opened_ts']}"
                rows.append(r)
    return sorted(rows, key=lambda r: r["opened_ts"])


def result_of(r: dict) -> str:
    ex, tp, ln, why = r["exit_price"], r.get("tp"), r.get("line"), (r.get("reason") or "")
    if tp and ex >= tp - 0.25:
        return "target"
    if ln and ex <= ln + 0.25:
        return "loss limit"
    if "exit" in why or "time" in why:
        return "16:00 close"
    return "closed"


def analyse(r: dict, prev: dict | None, bars: pd.DataFrame, ctx: Context, m1: pd.DataFrame | None = None) -> dict:
    o, c = pd.Timestamp(r["opened_ts"]), pd.Timestamp(r["closed_ts"])
    day = lucid_rules.trading_day(o.to_pydatetime())
    planned = pd.Timestamp(datetime.combine(pd.Timestamp(day).date() - timedelta(days=1),
                                            datetime.min.time().replace(hour=19)), tz=ET)
    if pd.Timestamp(day).dayofweek == 0:
        planned = planned - pd.Timedelta(days=0)        # Monday's race opens Sunday 19:00 ET (= day - 1)
    delay_min = (o - planned.tz_convert("UTC")).total_seconds() / 60
    restart = bool(prev and prev["bot"] == r["bot"] and
                   0 <= (o - pd.Timestamp(prev["closed_ts"])).total_seconds() <= 30 * 60)
    pv = futures.point_value(r["sym"])
    lots = float(r["lots"])
    if m1 is not None and len(m1) and m1.index[0] <= o:                 # minutes inside the trade only
        w = m1[(m1.index >= o.ceil("min")) & (m1.index < c.floor("min"))]
    else:
        w = bars[(bars.index >= o.floor("15min")) & (bars.index < c)]
    entry = float(r["entry_price"])
    sg = -1.0 if r.get("direction") == "short" else 1.0                # a short gains when the price falls
    if len(w):
        far_up, far_down = float(w["high"].max()) - entry, entry - float(w["low"].min())
        best, worst = (max(0.0, far_up), max(0.0, far_down)) if sg > 0 else (max(0.0, far_down), max(0.0, far_up))
    else:
        best = worst = 0.0
    if r.get("line") and result_of(r) == "loss limit":                  # Lucid closes the account at the line
        to_line = sg * (entry - float(r["line"]))
        worst = min(worst, to_line) if worst else to_line
    best = max(best, sg * (float(r["exit_price"]) - entry))
    worst = max(worst, sg * (entry - float(r["exit_price"])))
    res = result_of(r)
    why = (r.get("reason") or "")
    mistakes = []
    if delay_min > 30 and not restart:
        mistakes.append("late_entry")
    if (c - o).total_seconds() < 120 and res == "closed":
        mistakes.append("protection_failed")
    if "while_away" in why:
        mistakes.append("not_watching")
    if "rescored" in why:
        mistakes.append("sim_vs_lucid")
    pnl = float(r["pnl_usd"])
    kind = ("win" if pnl > 0 else ("execution" if mistakes and set(mistakes) & {"protection_failed", "late_entry"}
                                   and res == "closed" else "market"))
    return {"key": r["key"], "bot": r["bot"], "no": None, "attempt": r.get("attempt") or r.get("run"),
            "opened": r["opened_ts"], "closed": r["closed_ts"], "day": day, "lots": lots, "entry": entry,
            "direction": "short" if sg < 0 else "long",
            "exit": float(r["exit_price"]), "tp": r.get("tp"), "line": r.get("line"), "pnl": pnl, "result": res,
            "reason": why, "delay_min": delay_min, "restart": restart, "best_pts": best, "worst_pts": worst,
            "best_usd": best * pv * lots, "worst_usd": worst * pv * lots, "mistakes": mistakes, "kind": kind,
            "context": ctx.at(day), "note": r.get("note")}


def update(write_notes: bool = True) -> dict:
    ctx = Context()
    bars = _bars()
    m1 = _bars_1m()
    mem = json.loads(MEMORY.read_text(encoding="utf-8")) if MEMORY.exists() else {"trades": {}, "since": None}
    trades = load_trades()
    prev_of: dict[str, dict] = {}                 # the previous trade OF THE SAME BOT (restart detection)
    for i, r in enumerate(trades, start=1):
        a = analyse(r, prev_of.get(r["bot"]), bars, ctx, m1)
        a["no"] = i
        mem["trades"][a["key"]] = a
        prev_of[r["bot"]] = r
    mem["since"] = trades[0]["opened_ts"] if trades else None
    mem["updated"] = datetime.now(timezone.utc).isoformat()
    mem["lessons"] = lessons(mem, ctx)
    for t in mem["trades"].values():
        t["story"], t["verdict"] = story(t), verdict(t)
    mem["fixes"] = {k: {"name": v[0], "what": v[1], "fix": v[2]} for k, v in FIXES.items()}
    MEMORY.write_text(json.dumps(mem, indent=1, default=str), encoding="utf-8")
    if write_notes:
        write_ledger(mem)
        write_learning(mem)
    return mem


def lessons(mem: dict, ctx: Context) -> dict:
    tr = [t for t in mem["trades"].values() if t["kind"] != "execution"]
    out = {}
    for tag, name in TAGS.items():
        live = [t for t in tr if tag in (t["context"].get("tags") or [])]
        losses = sum(1 for t in live if t["pnl"] <= 0)
        h = ctx.history(tag)
        if len(live) < 5:
            verdict = "too early — fewer than 5 trades with this condition"
        elif np.isfinite(h["z"]) and h["z"] <= -2:
            verdict = "CONFIRMED by history — candidate rule for the prediction lab"
        elif np.isfinite(h["z"]) and abs(h["z"]) < 1:
            verdict = "rejected — history shows no real difference"
        else:
            verdict = "watching — history unclear"
        out[tag] = {"name": name, "live_trades": len(live), "live_losses": losses, "history": h, "verdict": verdict}
    return out


# --- notes ----------------------------------------------------------------------

def _t(ts: str, tz: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(tz)


def _usd(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}${abs(x):,.2f}"


def _ctx_line(cx: dict) -> str:
    if not cx:
        return "not available"
    parts = []
    if cx.get("move_19") is not None:
        parts.append(f"NQ {cx['move_19']:+.2%} between 16:00 and 19:00")
    if cx.get("prev_ret") is not None:
        parts.append(f"yesterday's session {cx['prev_ret']:+.2%}")
    if cx.get("vix") is not None:
        parts.append(f"VIX {cx['vix']:.1f}" + (f" ({cx['vix_chg']:+.1%} yesterday)" if cx.get("vix_chg") is not None else ""))
    if cx.get("tick") is not None:
        parts.append("yesterday " + ("buyers" if cx["tick"] > 0 else "sellers") + f" dominated (TICK {cx['tick']:+.0f})")
    if cx.get("vol_rank") is not None:
        parts.append(f"volatility {'high' if cx['vol_rank'] >= 2 / 3 else ('low' if cx['vol_rank'] < 1 / 3 else 'normal')}")
    if cx.get("event"):
        parts.append("**big release day**")
    return "; ".join(parts)


def summary(ts: list[dict]) -> str:
    """Per bot, in the terms that matter -- a trial trade's P&L lives inside its own $50,000 evaluation;
    on a real account a failed evaluation costs the reset fee, not the $2,000."""
    out = []
    tr = [t for t in ts if t["bot"] == "Trial"]
    if tr:
        p = sum(1 for t in tr if t["result"] == "target")
        f_ = sum(1 for t in tr if t["result"] == "loss limit")
        out.append(f"**Trial:** {len(tr)} trades — **{p} evaluation(s) passed, {f_} failed** "
                   f"(each trade's P&L is inside its own $50,000 paper evaluation; on a real Lucid account a "
                   f"failed evaluation costs the reset fee, about $120).")
    fu = [t for t in ts if t["bot"] == "Funded"]
    out.append(f"**Funded:** {len(fu)} trades, net {_usd(sum(t['pnl'] for t in fu))}." if fu else
               "**Funded:** no trades yet.")
    return "\n\n".join(out)


ICON = {"target": "✅", "loss limit": "❌", "16:00 close": "⏹️", "closed": "⚠️"}


def story(t: dict) -> list[str]:
    """A trade's story in plain words (markdown bullets without the dash) -- shared by the
    Trade ledger note and the dashboard's Brain page."""
    o_ny, c_ny = _t(t["opened"], "America/New_York"), _t(t["closed"], "America/New_York")
    o_pr, c_pr = _t(t["opened"], PRAGUE), _t(t["closed"], PRAGUE)
    held = c_ny - o_ny
    why = (f"evaluation race — attempt #{t['attempt']}" if t["bot"] == "Trial" else
           f"funded race — funded account #{t['attempt']}")
    if t["restart"]:
        why += ", started right after the previous attempt ended (owner's rule: start another one until it passes)"
    elif t["delay_min"] > 30:
        why += f", entered {t['delay_min'] / 60:.1f} h after the planned 19:00 New York start"
    else:
        why += ", the day's 19:00 New York entry"
    short = t.get("direction") == "short"
    sg = -1.0 if short else 1.0
    in_w, out_w = ("sold short", "bought back") if short else ("bought", "sold")
    tp_pts = sg * (t["tp"] - t["entry"]) if t["tp"] else None
    ln_pts = sg * (t["entry"] - t["line"]) if t["line"] else None
    return [
        f"**When:** {in_w} {o_ny:%a %H:%M} New York ({o_pr:%H:%M} Prague) → {out_w} {c_ny:%a %H:%M} New York "
        f"({c_pr:%H:%M} Prague) · held {int(held.total_seconds() // 3600)} h {int(held.total_seconds() % 3600 // 60)} min",
        f"**What:** {t['lots']:g} NQ {'short (betting on a fall)' if short else 'long'} at {t['entry']:,.2f}"
        + (f" · take-profit {t['tp']:,.2f} ({'−' if short else '+'}{tp_pts:,.2f} pts)" if tp_pts is not None else "")
        + (f" · fail line {t['line']:,.2f} ({'+' if short else '−'}{ln_pts:,.2f} pts)" if ln_pts is not None else "")
        + f" · {out_w} {t['exit']:,.2f}",
        f"**Why:** {why}.",
        f"**How it went:** best +{t['best_pts']:,.2f} pts ({_usd(t['best_usd'])}), worst −{t['worst_pts']:,.2f} pts "
        f"({_usd(-t['worst_usd'])})" + (f"; came within {max(0.0, ln_pts - t['worst_pts']):,.2f} pts of the fail line"
                                        if ln_pts is not None and t["result"] != "loss limit" else ""),
        f"**Market before entry:** {_ctx_line(t['context'])}",
        "**Execution:** " + ("OK" if not t["mistakes"] else
                             "; ".join(f"⚠️ {FIXES[m][0]}" for m in t["mistakes"] if m in FIXES)),
    ]


def verdict(t: dict) -> str:
    """Why the trade won or lost, in one sentence."""
    tags = t["context"].get("tags") or []
    cond = (" Conditions: " + ", ".join(TAGS[k] for k in tags) + ".") if tags else ""
    if t["pnl"] > 0:
        return (f"Win — the take-profit was reached (best +{t['best_pts']:,.2f} pts, worst dip "
                f"−{t['worst_pts']:,.2f} pts on the way).{cond}")
    if t["kind"] == "execution":
        return "Execution mistake — " + "; ".join(f"{FIXES[m][0]}: {FIXES[m][1]} Fix: {FIXES[m][2]}"
                                                  for m in t["mistakes"] if m in FIXES)
    short = t.get("direction") == "short"
    how = (f"the price {'rose' if short else 'fell'} to the fail line ({t['worst_pts']:,.2f} pts "
           f"{'above the short sale' if short else 'below the buy'}) after moving at most "
           f"{t['best_pts']:,.2f} pts its way" if t["result"] == "loss limit" else
           f"neither level was reached by the 16:00 close (best +{t['best_pts']:,.2f}, worst −{t['worst_pts']:,.2f} pts)")
    return f"Market loss — nothing went wrong in execution; {how}.{cond}"


def write_ledger(mem: dict) -> None:
    ts = sorted(mem["trades"].values(), key=lambda t: t["opened"], reverse=True)
    icon = ICON
    lines = ["# Trade ledger", "",
             f"*Written by the trade brain (`scripts/trade_brain.py`) after every trading day — "
             f"updated {datetime.now():%Y-%m-%d %H:%M}. Every trade of both bots, newest first. "
             f"Simulator (paper) trading.*", "",
             summary(ts), "",
             "| # | Day (New York) | Bot | Result | Trade P&L |", "|---|---|---|---|---|"]
    for t in ts:
        lines.append(f"| {t['no']} | {pd.Timestamp(t['day']):%a %d %b %Y} | {t['bot']} | {icon.get(t['result'], '')} "
                     f"{t['result']} | {_usd(t['pnl'])} |")
    lines += ["", "See [[Learning]] for what the losses taught.", ""]
    for t in ts:
        lines += [f"### #{t['no']} · {t['bot']} · {pd.Timestamp(t['day']):%a %d %b %Y} · {ICON.get(t['result'], '')} "
                  f"{t['result']} · {_usd(t['pnl'])}"] + [f"- {x}" for x in story(t)] + [f"- **Verdict:** {verdict(t)}", ""]
    LEDGER_NOTE.write_text("\n".join(lines), encoding="utf-8")


def write_learning(mem: dict) -> None:
    ts = sorted(mem["trades"].values(), key=lambda t: t["opened"], reverse=True)
    counts: dict[str, list] = {}
    for t in ts:
        for m in t["mistakes"]:
            counts.setdefault(m, []).append(t)
    lines = ["# Learning", "",
             f"*Written by the trade brain after every trading day — updated {datetime.now():%Y-%m-%d %H:%M}.*", "",
             "How the brain learns: every losing trade is sorted into an **execution mistake** (a real error — "
             "fixed for good and listed below) or a **market loss** (the race loses about 6 in 10 even when "
             "everything works). A market condition becomes a rule only when 5+ live trades **and** the race's "
             "history (2021-2026) agree — so one unlucky day never teaches it a superstition. Confirmed lessons go "
             "to the [[Prediction lab]] to prove themselves before they change any trading.", "",
             "## Never again — execution mistakes and their fixes", "",
             "| Mistake | Times | Last seen | What happened | Fix |", "|---|---|---|---|---|"]
    if counts:
        for m, tl in counts.items():
            name, what, fix = FIXES[m]
            lines.append(f"| {name} | {len(tl)} | {pd.Timestamp(tl[0]['day']):%d %b} (trade #{tl[0]['no']}) | {what} | {fix} |")
    else:
        lines.append("| none yet | | | | |")
    lines += ["", "## Market lessons being tested", "",
              "| Condition before the race | Live: losses / trades | History: race wins with it / without | Verdict |",
              "|---|---|---|---|"]
    for tag, l in mem["lessons"].items():
        h = l["history"]
        hist = (f"{h['with']:.0%} ({h['n_with']} days) / {h['without']:.0%} (z {h['z']:+.1f})"
                if h["n_with"] else "—")
        lines.append(f"| {l['name']} | {l['live_losses']} / {l['live_trades']} | {hist} | {l['verdict']} |")
    lines += ["", "## Loss analyses (newest first)", ""]
    for t in ts:
        if t["pnl"] > 0:
            continue
        ny = _t(t["opened"], "America/New_York")
        lines.append(f"### Trade #{t['no']} · {t['bot']} · {pd.Timestamp(t['day']):%a %d %b} · {t['result']} · "
                     f"{_usd(t['pnl'])}")
        lines.append(f"- **Verdict:** {verdict(t)}")
        lines.append(f"- **{'Sold short' if t.get('direction') == 'short' else 'Bought'}:** {ny:%H:%M} New York")
        lines.append("")
    LEARN_NOTE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    m = update()
    print(f"brain: {len(m['trades'])} trades remembered; notes written to {LEDGER_NOTE.parent}")
