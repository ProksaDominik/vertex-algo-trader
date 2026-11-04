"""Live strategy laboratory -- forward-test hundreds of configs on real data.

WHAT THIS IS
------------
Every config in the cohort is evaluated on live bars as they close. When one
signals, the entry, stop and target are recorded; the outcome is then scored
from SUBSEQUENT bars (did price reach the target or the stop first). No order
is sent in shadow mode, so hundreds of configs can run at once for nothing, and
the sample is genuinely out-of-sample -- it is scored on data that did not
exist when the backtests were run.

WHAT IT IS NOT
--------------
It is not a way to find a winner quickly, and the report deliberately refuses
to name one until the arithmetic allows it. Against a per-trade R standard
deviation near 1.0, detecting a +0.05R edge needs roughly 7,000 trades FOR THAT
CONFIG. Spread 2,000 trades/day across 500 configs and each gets 4/day, which
is seven years. Across 10 configs it is about a month. Breadth and speed trade
off directly, which is why this is two-tier: shadow is free and can be wide,
live fills are expensive and must be narrow.

The multiple-comparison trap is the real hazard. With 400 zero-edge configs,
about 20 will clear p<0.05 by chance alone, and the best will look excellent.
lab_report.py applies Benjamini-Hochberg and reports each config's own
detectable-effect floor, so "top of the leaderboard" is never mistaken for
"has an edge".

SAFETY
------
Shadow mode issues NO orders -- it only reads bars. It cannot touch the funded
or trial books. Live mode (--tier live) sends minimum-size orders under its own
magic block (8800+), never reusing the funded (9200s) or trial (9100s) ranges.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from metatraderbot.lab_registry import FUTURES_SYMBOLS, build, instantiate, Stat   # noqa: E402

load_dotenv(ROOT / ".env")
# VERTEX_BROKER=ninjatrader: forward-test Lucid's CME futures through
# NinjaTrader instead of MT5 CFDs. Separate files -- the configs' ids do not
# name a symbol, so the two symbol pools must never share one sample.
FUTURES = os.environ.get("VERTEX_BROKER", "mt5").strip().lower() == "ninjatrader"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / ("lab_futures.log" if FUTURES else "lab.log"),
                                  encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("lab")

STATE_PATH = ROOT / "data" / ("lab_futures_state.json" if FUTURES else "lab_state.json")
SIGNALS_PATH = ROOT / "data" / ("lab_futures_signals.jsonl" if FUTURES else "lab_signals.jsonl")
BAR_COUNT = 300
POLL_S = 30
MAX_HOLD_BARS = 96          # a shadow signal that never resolves is dropped
LAB_MAGIC_BASE = 8800       # distinct from funded (9200s) and trial (9100s)

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}

_STRAT_CACHE: dict = {}


def make_strategy(cfg, sym: str):
    """Instantiate a strategy for one (LabConfig, symbol).

    Cached per (cid, symbol): a config is pooled across 12 symbols, and several
    families hold per-symbol state (pip sizing, session ranges), so one shared
    instance across symbols would leak state between them.
    """
    key = (cfg.cid, sym)
    if key in _STRAT_CACHE:
        return _STRAT_CACHE[key]
    s = instantiate(cfg.family, cfg.params, cfg.tp_r, sym)
    _STRAT_CACHE[key] = s
    return s


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.warning("lab_state.json unreadable -- starting a fresh sample")
    return {"stats": {}, "started": datetime.now(timezone.utc).isoformat()}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(STATE_PATH)      # atomic: a crash mid-write must not truncate
                                 # the accumulated sample


AMBIGUOUS = "ambiguous"


def lucid_flat_due(bar_open: pd.Timestamp, tf: str) -> bool:
    """Futures mode: must a position be flat by the END of this bar? True
    when the bar closes inside Lucid's flatten window or after the cutoff --
    the bot is flat by then, so a shadow trade must be scored there too."""
    from metatraderbot import lucid_rules
    close = (bar_open + pd.Timedelta(seconds=TF_SECONDS.get(tf, 900))).to_pydatetime()
    return lucid_rules.session_state(close) in (lucid_rules.Session.FLATTEN,
                                                lucid_rules.Session.CLOSED)


def lucid_entry_ok(bar_open: pd.Timestamp, tf: str) -> bool:
    """Futures mode: could the bot open a trade when this bar closes?"""
    from metatraderbot import lucid_rules
    close = (bar_open + pd.Timedelta(seconds=TF_SECONDS.get(tf, 900))).to_pydatetime()
    return lucid_rules.session_state(close) == lucid_rules.Session.OPEN


def score_open(sig: dict, bars, fine=None, flat_due=None):
    """Resolve a pending signal from bars after its entry.

    Returns a float R, None (still open), or AMBIGUOUS (caller should retry
    with finer data, or drop the trade).

    Uses HIGH/LOW, not close: a stop is hit intrabar. Scoring on closes only
    would flatter every config by ignoring the excursions that actually take a
    trade out.

    THE AMBIGUOUS CASE MATTERS MORE THAN IT LOOKS. This originally scored a bar
    spanning BOTH stop and target as a loss, on the reasoning that it was the
    conservative choice. Measured on live 5m data that was badly wrong: stops
    sit only ~1.35x a single bar's range, 29% of trades resolve on the first
    bar, and 26% of THOSE span both levels. Forcing them all to -1R imposed a
    systematic bias near -0.078R per trade -- the same order as the edge this
    lab exists to detect. Months of running would have measured the scoring
    rule, not the strategies.

    So ambiguity is escalated rather than guessed: ``fine`` is a callable
    returning 1-minute bars for a window, which usually settles which level was
    touched first. Anything still ambiguous at 1m is dropped from the sample and
    counted, which is unbiased where a forced verdict is not.
    """
    # sig["ts"] is an ISO string, not a Timestamp: open signals are persisted to
    # JSON every poll so the sample survives a restart, and a Timestamp is not
    # serialisable. Parse back here rather than storing an unserialisable object.
    after = bars[bars.index > pd.Timestamp(sig["ts"])]
    if len(after) == 0:
        return None
    entry, sl, tp = sig["entry"], sig["sl"], sig["tp"]
    is_long = sig["dir"] == "long"
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    reward = abs(tp - entry) / risk

    for ts, b in after.iterrows():
        hi, lo = float(b["high"]), float(b["low"])
        if is_long:
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
        if hit_sl and hit_tp:
            if fine is None:
                return AMBIGUOUS
            fb = fine(ts)
            if fb is None or len(fb) == 0:
                return AMBIGUOUS
            for _, f in fb.iterrows():
                fhi, flo = float(f["high"]), float(f["low"])
                if is_long:
                    s_hit, t_hit = flo <= sl, fhi >= tp
                else:
                    s_hit, t_hit = fhi >= sl, flo <= tp
                if s_hit and t_hit:
                    return AMBIGUOUS      # still tied at 1m -- do not guess
                if s_hit:
                    return -1.0
                if t_hit:
                    return reward
            return AMBIGUOUS
        if hit_sl:
            return -1.0
        if hit_tp:
            return reward
        if flat_due is not None and flat_due(ts):
            # Lucid: flat before 4:45 PM ET. Scored where the bot would exit.
            close = float(b["close"])
            return ((close - entry) if is_long else (entry - close)) / risk
    if len(after) >= MAX_HOLD_BARS:
        last = float(after.iloc[-1]["close"])
        return ((last - entry) if is_long else (entry - last)) / risk
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=400)
    ap.add_argument("--tier", default="shadow", choices=["shadow", "live"])
    ap.add_argument("--report-every", type=int, default=20)
    args = ap.parse_args()

    if args.tier == "live":
        log.error("live tier is not enabled yet -- promote configs only after "
                  "shadow has produced a readable sample (see lab_report.py)")
        return 2

    cohort = build(symbols=FUTURES_SYMBOLS if FUTURES else None, max_configs=args.configs)
    log.warning("LAB starting: %d strategies, tier=%s, market=%s", len(cohort), args.tier,
                "Lucid futures (NinjaTrader)" if FUTURES else "MT5 CFDs")

    state = load_state()
    stats: dict[str, Stat] = {}
    for cid, d in state.get("stats", {}).items():
        s = Stat()
        s.__dict__.update(d)
        stats[cid] = s
    for c in cohort:
        stats.setdefault(c.cid, Stat())

    from metatraderbot import free_prices
    if FUTURES and free_prices.in_use():
        # paper mode (owner 2026-10-15): the internet's prices, 10 minutes delayed -- no NinjaTrader
        engine = free_prices.FreeFeed()
        while True:
            try:
                engine.connect()
                break
            except Exception as e:  # noqa: BLE001 -- no internet yet
                log.warning(f"internet prices not reachable yet: {e!r} -- retrying in 60s")
                time.sleep(60)
    elif FUTURES:
        # Bars only -- the lab never places an order, so no contract cap.
        from metatraderbot.nt_engine import BridgeError, NinjaTraderEngine
        engine = NinjaTraderEngine()
        # The desktop icon starts NinjaTrader and the bots together, so the
        # bridge may need a minute: wait for it rather than exit (an exited lab
        # stays dead until the next launch).
        while True:
            try:
                engine.connect()
                break
            except BridgeError as e:
                log.warning(f"NinjaTrader bridge not usable yet: {e} -- retrying in 60s")
                time.sleep(60)
    else:
        from metatraderbot.mt5_engine import MT5Engine
        engine = MT5Engine(symbol_map={"NAS100": "USTEC"})
        if not engine.connect():
            log.error("MT5 connect failed -- is the terminal running?")
            return 1

    # Strategies are symbol-pooled, so the work is (symbol x timeframe) bar
    # fetches, each feeding every strategy that runs on that timeframe. Bars are
    # fetched ONCE per pair per poll -- fetching per strategy would be 296x the
    # IPC traffic through a lock shared with two live trading bots.
    pairs = sorted({(s, c.timeframe) for c in cohort for s in c.symbols})
    by_tf: dict[str, list] = {}
    for c in cohort:
        by_tf.setdefault(c.timeframe, []).append(c)
    log.warning("cohort: %d strategies over %d symbol/timeframe pairs "
                "(%d strategy-symbol evaluations per poll)",
                len(cohort), len(pairs), sum(len(v) for v in by_tf.values()) * 1)

    last_bar: dict[tuple, object] = {}
    polls = 0
    sig_file = open(SIGNALS_PATH, "a", encoding="utf-8")

    while True:
        polls += 1
        t0 = time.time()
        new_signals = resolved = ambiguous = 0

        def _fine_for(symbol):
            """1-minute bars for disambiguating a single bar, fetched lazily.

            Only ~8% of trades need this, so it is a closure per symbol rather
            than a bulk fetch -- pulling 1m bars for 12 symbols every poll would
            multiply IPC traffic through a lock shared with two live bots for no
            benefit on the 92% that resolve cleanly.
            """
            cache = {}

            def _get(bar_ts):
                if bar_ts in cache:
                    return cache[bar_ts]
                try:
                    fb = engine.get_bars(symbol, "1m", 240)
                    lo = pd.Timestamp(bar_ts)
                    hi = lo + pd.Timedelta(seconds=TF_SECONDS.get(tf, 300))
                    fb = fb[(fb.index >= lo) & (fb.index < hi)]
                except Exception:
                    fb = None
                cache[bar_ts] = fb
                return fb
            return _get

        for sym, tf in pairs:
            try:
                if not engine.is_tradeable(sym):
                    continue
                bars = engine.get_bars(sym, tf, BAR_COUNT)
            except Exception as exc:
                log.debug("bars %s %s: %r", sym, tf, exc)
                continue
            if len(bars) < 60:
                continue
            closed = bars.iloc[:-1]          # drop the still-forming bar
            latest = closed.index[-1]
            cohort_tf = by_tf.get(tf, [])

            # resolve this symbol's pending signals for every strategy on this tf
            for c in cohort_tf:
                st = stats[c.cid]
                still = []
                for sig in st.open_signals:
                    if sig.get("sym") != sym:
                        still.append(sig)      # another symbol's -- not ours to score
                        continue
                    r = score_open(sig, closed, fine=_fine_for(sym),
                                   flat_due=(lambda ts, _tf=tf: lucid_flat_due(ts, _tf)) if FUTURES else None)
                    if r is None:
                        still.append(sig)
                    elif r is AMBIGUOUS:
                        # Still tied at 1m resolution. Dropped, not guessed --
                        # forcing a verdict here is what biased the first run.
                        ambiguous += 1
                        sig_file.write(json.dumps(
                            {"cid": c.cid, "sym": sym, "ts": sig["ts"],
                             "r": None, "dropped": "ambiguous",
                             "dir": sig["dir"], "resolved": latest.isoformat()}) + "\n")
                    else:
                        st.add(r, sig["dir"] == "long", latest.isoformat())
                        resolved += 1
                        sig_file.write(json.dumps(
                            {"cid": c.cid, "sym": sym, "ts": sig["ts"], "r": round(r, 4),
                             "dir": sig["dir"], "resolved": latest.isoformat()}) + "\n")
                st.open_signals = still

            if last_bar.get((sym, tf)) == latest:
                continue                      # no new closed bar -> no new signals
            last_bar[(sym, tf)] = latest
            if FUTURES and not lucid_entry_ok(latest, tf):
                continue                      # the bot could not open a trade now either

            for c in cohort_tf:
                st = stats[c.cid]
                # one open signal per (strategy, SYMBOL), not per strategy --
                # pooling exists to multiply the sample, and a single global
                # slot would throttle a 12-symbol strategy back to the rate of
                # a 1-symbol one, defeating the point.
                if any(s.get("sym") == sym for s in st.open_signals):
                    continue
                try:
                    specs = make_strategy(c, sym).on_bar(len(closed) - 1, closed)
                except Exception:
                    continue
                for spec in specs or []:
                    d = getattr(spec, "direction", None)
                    is_long = getattr(d, "name", str(d)).upper() == "LONG"
                    entry = float(getattr(spec, "entry_price", closed.iloc[-1]["close"]))
                    sl = float(getattr(spec, "stop_loss", 0) or 0)
                    if not sl or sl == entry:
                        continue
                    risk = abs(entry - sl)
                    tp = entry + (risk * c.tp_r if is_long else -risk * c.tp_r)
                    st.open_signals.append({"sym": sym, "ts": latest.isoformat(),
                                            "entry": entry, "sl": sl, "tp": tp,
                                            "dir": "long" if is_long else "short"})
                    new_signals += 1
                    break                     # first spec only

        sig_file.flush()
        state["stats"] = {k: {kk: vv for kk, vv in v.__dict__.items()}
                          for k, v in stats.items()}
        state["polls"] = polls
        state["updated"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        if polls % args.report_every == 0:
            tot = sum(s.n for s in stats.values())
            openn = sum(len(s.open_signals) for s in stats.values())
            ready = sum(1 for s in stats.values() if s.n >= 30)
            log.warning("poll %d: %d resolved trades total, %d open, %d/%d configs "
                        "past n=30, %d dropped ambiguous this poll  (%.1fs/poll)",
                        polls, tot, openn, ready, len(cohort), ambiguous,
                        time.time() - t0)

        time.sleep(max(1, POLL_S - (time.time() - t0)))


if __name__ == "__main__":
    raise SystemExit(main())
