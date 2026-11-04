"""Order flow (who traded aggressively: buyers or sellers) as a race predictor (2026-10-07).

Data: scripts/fetch_orderflow.py -> data/orderflow/NQ_<day>.parquet, per minute: volume
bought at the ask, sold at the bid, and the same for big trades (>= 10 contracts).
For the race of Lucid day d (in at 19:00 ET the evening before) only these are known:
  * the previous Lucid day: its regular session 09:30-16:00, its last hour, the hour after
    the close, the whole day, and the last 3 sessions together;
  * day d's own first hour, 18:00-19:00 ET, which ends when the race goes in.
"Delta" = (bought - sold) / all volume: > 0 buyers pushed, < 0 sellers pushed.

Scored like every earlier search (race 2 NQ +75.5 / -50 pts, rules by sign and by past-only
thirds), on the real NinjaTrader NQ bars. The history NinjaTrader keeps is short, so the
days with order flow are split in half by date: rank on the first half, judge on the second.

Usage:  .venv/Scripts/python.exe -I scripts/lucid_orderflow_tick_search.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_orderflow_search as of  # noqa: E402
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402

FLOW = ROOT / "data" / "orderflow"
ET = "America/New_York"
A19 = lp.S(19)


def day_features() -> pd.DataFrame:
    """One row per Lucid day (the file's day) with the deltas of its parts."""
    rows = []
    for f in sorted(FLOW.glob("NQ_*.parquet")):
        day = pd.Timestamp(f.stem[3:])
        df = pd.read_parquet(f)
        if df.empty:
            continue
        t = df.index.tz_convert(ET)
        mins = t.hour * 60 + t.minute
        on_day = np.asarray(t.date == day.date())

        def part(mask):
            x = df[mask]
            vol = float(x[["buy", "sell", "other"]].to_numpy().sum())
            if vol <= 0:
                return np.nan, np.nan, 0.0
            return (float(x["buy"].sum() - x["sell"].sum()) / vol,
                    float(x["big_buy"].sum() - x["big_sell"].sum()) / vol, vol)

        rth = on_day & (mins >= 570) & (mins < 960)
        d_rth, b_rth, v_rth = part(rth)
        d_last, b_last, _ = part(on_day & (mins >= 900) & (mins < 960))
        d_after, _, _ = part(on_day & (mins >= 960))
        d_all, b_all, _ = part(np.ones(len(df), bool))
        d_eve, b_eve, v_eve = part(~on_day & (mins >= 1080) & (mins < 1140))
        rows.append({"day": day, "rth": d_rth, "rth_big": b_rth, "rth_vol": v_rth, "last_hour": d_last,
                     "last_hour_big": b_last, "after_close": d_after, "all": d_all, "all_big": b_all,
                     "evening": d_eve, "evening_big": b_eve, "evening_vol": v_eve})
    return pd.DataFrame(rows).set_index("day").sort_index()


def prev_day(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """For each Lucid day d: the value of the last Lucid day BEFORE d (gaps of more than 4
    calendar days -- missing history -- give nothing)."""
    s = s.dropna()
    days = pd.DatetimeIndex(s.index).as_unit("ns")
    left = pd.DataFrame({"d": pd.DatetimeIndex(idx).as_unit("ns")})
    right = pd.DataFrame({"d": days, "src": days, "v": s.to_numpy()})
    m = pd.merge_asof(left, right, on="d", allow_exact_matches=False)
    gap = (m["d"] - m["src"]).dt.days
    return pd.Series(np.where(gap <= 4, m["v"], np.nan), index=idx)


def signals(feat: pd.DataFrame, idx: pd.DatetimeIndex, g: lp.Grid) -> dict:
    zs = lambda x: (x - x.rolling(40, min_periods=20).mean()) / x.rolling(40, min_periods=20).std()  # noqa: E731
    pv = lambda c: prev_day(feat[c], idx)  # noqa: E731
    same = lambda c: feat[c].reindex(idx)  # noqa: E731
    ret = g.daily.reindex(idx)["prev_rth_ret"]
    out = {
        "yesterday's session: delta": pv("rth"),
        "yesterday's session: big-trade delta": pv("rth_big"),
        "yesterday's session: delta vs past 40 days": prev_day(zs(feat["rth"]), idx),
        "yesterday's last hour: delta": pv("last_hour"),
        "yesterday's last hour: big-trade delta": pv("last_hour_big"),
        "yesterday after the close (16-17h): delta": pv("after_close"),
        "yesterday's whole day: delta": pv("all"),
        "yesterday's whole day: big-trade delta": pv("all_big"),
        "last 3 sessions: delta": prev_day(feat["rth"].rolling(3).mean(), idx),
        "18:00-19:00 before the race: delta": same("evening"),
        "18:00-19:00 before the race: big-trade delta": same("evening_big"),
        "yesterday: buyers vs price (delta minus price move)":
            prev_day(zs(feat["rth"]), idx) - (ret / ret.rolling(60, min_periods=30).std()),
        "yesterday's session volume vs past 40 days": prev_day(zs(feat["rth_vol"]), idx),
    }
    return {k: (A19, v, "cont") for k, v in out.items()}


def rules(x: pd.Series) -> dict[str, pd.Series]:
    """As lucid_signal_search.rules_for, but the thirds come from the previous 120 days with
    order flow (at least 40) -- the history is too short for the usual 252 / 120."""
    out = {}
    sgn = np.sign(x.fillna(0.0))
    out["side = sign (momentum)"] = sgn
    out["side = -sign (reversal)"] = -sgn
    v = x.dropna()
    q1 = v.rolling(120, min_periods=40).quantile(1 / 3).shift(1)
    q2 = v.rolling(120, min_periods=40).quantile(2 / 3).shift(1)
    t = pd.Series(np.where(v <= q1, 0, np.where(v <= q2, 1, 2)), index=v.index, dtype=float)
    t[q1.isna()] = np.nan
    t = t.reindex(x.index)
    for k, lab in ((0, "low"), (1, "middle"), (2, "high")):
        out[f"long only in {lab} third"] = pd.Series(np.where(t == k, 1, 0), index=x.index)
        out[f"short only in {lab} third"] = pd.Series(np.where(t == k, -1, 0), index=x.index)
    return out


def main() -> int:
    feat = day_features()
    if len(feat) < 60:
        print(f"only {len(feat)} days of order flow in {FLOW} -- run scripts/fetch_orderflow.py first")
        return 2
    g = lp.load_grid("NQ", path=of.NT["NQ"])
    idx = pd.to_datetime(g.dates)
    sigs = signals(feat, idx, g)
    lab = ss.race_labels(g, A19)
    have = idx >= feat.index.min()
    split = feat.index[len(feat) // 2]
    A, B = have & (idx < split), have & (idx >= split)
    always = pd.Series(1, index=idx)

    def win(side: pd.Series, mask) -> tuple[float, int]:
        s = side.reindex(idx).fillna(0).to_numpy()
        y = np.where(s > 0, lab[1], np.where(s < 0, lab[-1], np.nan))
        k = mask & (s != 0) & ~np.isnan(y)
        return (float(np.nanmean(y[k])) if k.any() else np.nan), int(k.sum())

    flow_days = pd.Series(1, index=idx).where(pd.Series(idx, index=idx).isin(feat.index), 0)
    ba, _ = win(flow_days, A)                                   # always-buy on every day with order flow
    bb, _ = win(flow_days, B)
    rows = []
    for name, (_, x, kind) in sigs.items():
        for rule, side in rules(x).items():
            pa, na = win(side, A)
            pb, nb = win(side, B)
            za = (pa - ba) / math.sqrt(ba * (1 - ba) / na) if na and 0 < ba < 1 else np.nan
            zb = (pb - bb) / math.sqrt(bb * (1 - bb) / nb) if nb and 0 < bb < 1 else np.nan
            rows.append({"signal": name, "rule": rule, "A win": pa, "A buy": ba, "A n": na, "A z": za,
                         "B win": pb, "B buy": bb, "B n": nb, "B z": zb})
    res = pd.DataFrame(rows)
    ok = res["A n"] >= 40
    res["A p"] = [ss.norm_sf(v) if np.isfinite(v) else 1.0 for v in res["A z"]]
    res["BH 10%"] = False
    if ok.any():
        res.loc[ok, "BH 10%"] = ss.bh_fdr(res.loc[ok, "A p"].to_numpy(), 0.10)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 52)
    print(f"Order flow: {len(feat)} days {feat.index.min():%Y-%m-%d} .. {feat.index.max():%Y-%m-%d}; "
          f"ranked on days before {split:%Y-%m-%d}, judged after. {int(ok.sum())} rules, "
          f"{int(res['BH 10%'].sum())} pass the false-discovery check.")
    f = {c: (lambda v: f"{v:.3f}") for c in ("A win", "A buy", "B win", "B buy")}
    f.update({c: (lambda v: f"{v:+.2f}") for c in ("A z", "B z")})
    cols = ["signal", "rule", "A win", "A buy", "A n", "A z", "B win", "B buy", "B n", "B z", "BH 10%"]
    print(res[ok].sort_values("A z", ascending=False).head(15)[cols].to_string(index=False, formatters=f))
    held = ok & (res["A z"] >= 2) & (res["B z"] >= 1.65)
    print(f"A z >= 2: {int((ok & (res['A z'] >= 2)).sum())}; also holding on the second half: {int(held.sum())}")
    res.to_csv(FLOW / "orderflow_search.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
