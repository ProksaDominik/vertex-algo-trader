"""Does Google's TimesFM (zero-shot time-series foundation model) predict the race? (2026-10-03)

Skill: claude-scientific-skills/timesfm-forecasting (TimesFM 2.5, 200M,
Apache-2.0 weights, pinned revision). Runs in its own environment
(C:\\Users\\Proksha\\.venv-timesfm) so torch never touches the bots' .venv.

Three steps:
  --export    (bot .venv)      NQ 15-min and daily closes + race outcomes -> data/futures/timesfm_in.npz
  --forecast  (timesfm venv)   for every Lucid day: from the 512 bars BEFORE 19:00 ET, forecast the
                               84 bars to 16:00 ET; and from 512 daily closes, the next close
                               -> data/futures/timesfm_out.npz
  --score     (bot .venv)      race on the model's side; win rate 2016-22 vs 2023-26 vs the 2026
                               slice (after the model's release -- the only certainly unseen data:
                               its pretraining corpus may contain NQ itself), then full replays.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "futures" / "timesfm_in.npz"
OUT_PATH = ROOT / "data" / "futures" / "timesfm_out.npz"
CTX, H15 = 512, 84          # 512 bars of context; 19:00 -> 16:00 ET = 84 fifteen-minute bars
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def export() -> None:
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import lucid_pass_research as lp
    import lucid_signal_search as ss
    g = lp.load_grid("NQ")
    a = lp.S(19)
    flat = pd.Series(g.C.ravel()).ffill().to_numpy()          # 18:00 .. 16:45 slots, day after day
    n = len(g.dates)
    ok = g.has[:, a] & g.has[:, lp.RTH_LAST] & (np.arange(n) * lp.K + a >= CTX)
    days = np.flatnonzero(ok)
    ctx15 = np.stack([flat[i * lp.K + a - CTX: i * lp.K + a] for i in days]).astype(np.float32)
    closes = g.daily["close"].ffill().to_numpy()
    ctxd = np.stack([closes[max(0, i - CTX): i] if i >= CTX else
                     np.concatenate([np.full(CTX - i, closes[0]), closes[:i]]) for i in days]).astype(np.float32)
    lab = ss.race_labels(g, a)
    np.savez_compressed(IN_PATH, dates=g.dates[days].astype("datetime64[D]").astype(str), days=days,
                        ctx15=ctx15, ctxd=np.nan_to_num(ctxd, nan=float(np.nanmean(closes))),
                        entry=g.O[days, a], y_long=lab[1][days], y_short=lab[-1][days])
    print(f"exported {len(days)} days -> {IN_PATH}")


def forecast(in_path: Path = IN_PATH, out_path: Path = OUT_PATH) -> None:
    import timesfm
    t0 = time.time()
    d = np.load(in_path)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", revision=REVISION,
                                                           torch_compile=False)
    model.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=128, per_core_batch_size=32,
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=False,
                                         fix_quantile_crossing=True))
    out = {}
    for key, h in (("ctx15", H15), ("ctxd", 1)):
        x = d[key]
        med, q20, q80 = [], [], []
        for s in range(0, len(x), 64):
            point, q = model.forecast(horizon=h, inputs=[r for r in x[s:s + 64]])
            med.append(q[:, -1, 5])
            q20.append(q[:, -1, 2])
            q80.append(q[:, -1, 8])
            if s % 640 == 0:
                print(f"  {key}: {s}/{len(x)} ({time.time() - t0:.0f}s)", flush=True)
        last = x[:, -1]
        out[f"{key}_med"] = np.concatenate(med) - last
        out[f"{key}_q20"] = np.concatenate(q20) - last
        out[f"{key}_q80"] = np.concatenate(q80) - last
    np.savez_compressed(out_path, **out)
    print(f"forecasts saved -> {out_path} in {time.time() - t0:.0f}s")


def score() -> None:
    import math
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import lucid_pass_research as lp
    import lucid_signal_search as ss
    d, f = np.load(IN_PATH), np.load(OUT_PATH)
    dates = pd.to_datetime(d["dates"])
    yrs = dates.year.to_numpy()
    yL, yS = d["y_long"], d["y_short"]

    def win(side, sel):
        y = np.where(side > 0, yL, np.where(side < 0, yS, np.nan))
        m = sel & (side != 0) & ~np.isnan(y)
        return (float(np.nanmean(y[m])) if m.any() else np.nan), int(m.sum())

    periods = {"2016-22": (yrs <= 2022), "2023-25": (yrs >= 2023) & (yrs <= 2025), "2026 (unseen)": yrs >= 2026}
    base = {p: win(np.ones(len(yrs)), m)[0] for p, m in periods.items()}
    rows = []
    for key, lab in (("ctx15", "15-min path to 16:00"), ("ctxd", "next daily close")):
        med, q20, q80 = f[f"{key}_med"], f[f"{key}_q20"], f[f"{key}_q80"]
        rules = {"side = forecast": np.sign(med), "side = against forecast": -np.sign(med),
                 "long only if forecast up": np.where(med > 0, 1, 0),
                 "long only if 80% band above 0": np.where(q20 > 0, 1, 0),
                 "short only if 80% band below 0": np.where(q80 < 0, -1, 0)}
        for rname, side in rules.items():
            r = {"model": lab, "rule": rname}
            for p, m in periods.items():
                w, n = win(side, m)
                r[f"{p} win"] = w
                r[f"{p} n"] = n
                r[f"{p} z"] = (w - base[p]) / math.sqrt(base[p] * (1 - base[p]) / n) if n else np.nan
            rows.append(r)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("baseline long race win rate: " + ", ".join(f"{p} {v:.3f}" for p, v in base.items()))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:.0f}"))
    print(f"\nforecast agreement with the actual 19:00->16:00 sign:")
    g = lp.load_grid("NQ")
    days = d["days"]
    actual = np.sign(g.C[days, lp.RTH_LAST] - d["entry"])
    for key in ("ctx15", "ctxd"):
        s = np.sign(f[f"{key}_med"])
        for p, m in periods.items():
            mm = m & (actual != 0) & (s != 0)
            print(f"  {key} {p}: {(s[mm] == actual[mm]).mean():.3f} of {mm.sum()} days")
    res.to_csv(ROOT / "data" / "futures" / "timesfm_race.csv", index=False)
    best = res.loc[res["2016-22 z"].idxmax()]
    key = "ctx15" if best["model"].startswith("15") else "ctxd"
    med, q20, q80 = f[f"{key}_med"], f[f"{key}_q20"], f[f"{key}_q80"]
    side_map = {"side = forecast": np.sign(med), "side = against forecast": -np.sign(med),
                "long only if forecast up": np.where(med > 0, 1, 0),
                "long only if 80% band above 0": np.where(q20 > 0, 1, 0),
                "short only if 80% band below 0": np.where(q80 < 0, -1, 0)}
    full = np.zeros(len(g.dates), int)
    full[days] = side_map[best["rule"]].astype(int)
    print(f"\nfull replays, best 2016-22 rule ({best['model']} | {best['rule']}) vs the deployed race:")
    rep = pd.DataFrame([{"rule": "deployed race", **ss.replays(g, lp.S(19), np.ones(len(g.dates), int))},
                        {"rule": "TimesFM best", **ss.replays(g, lp.S(19), full)}])
    print(rep.to_string(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 5 else f"{v:,.0f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("export", "forecast", "score"))
    ap.add_argument("--inp", type=Path, default=IN_PATH, help="forecast: contexts file (ctx15, ctxd)")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="forecast: where to write the forecasts")
    args = ap.parse_args()
    if args.step == "forecast":
        forecast(args.inp, args.out)
    else:
        {"export": export, "score": score}[args.step]()
