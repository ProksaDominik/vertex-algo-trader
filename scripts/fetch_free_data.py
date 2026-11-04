"""Free market data the race research had not used yet (2026-10-07, owner: "do all the free ones").

  * CFTC Traders in Financial Futures, E-mini Nasdaq-100 (code 209742), weekly since 2006:
    what dealers, asset managers (pension funds, mutual funds) and leveraged funds (hedge
    funds, CTAs) hold. Positions are as of Tuesday and published Friday 15:30 ET, so every
    row is stamped with its RELEASE day (Tuesday + 3) -- the research may use it only after.
  * Cboe option-based indices, daily: SKEW (price of crash protection), VVIX (volatility of
    VIX), VIX9D / VIX / VIX3M (the volatility curve: 9 days, 30 days, 3 months), VIX1D
    (today's expected move, from 2022), VXN (the Nasdaq-100 VIX).
  * Cboe put/call volume ratios, daily: total, equity and index options (history file to
    4 Oct 2019, then one small daily file per day; those also carry put/call open interest).

Raw downloads go to data/free/<source>/, the cleaned tables to data/free/*.parquet.
Rerunning only fetches what is missing.

Usage:  .venv/Scripts/python.exe -I scripts/fetch_free_data.py
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "free"
UA = {"User-Agent": "Mozilla/5.0 (research; vertex bot)"}


def get(url: str, timeout: int = 60) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except Exception:  # noqa: BLE001 -- network hiccup: retry
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


# --- CFTC ---------------------------------------------------------------------------

COT_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
COT_COLS = ["open_interest_all", "dealer_positions_long_all", "dealer_positions_short_all",
            "asset_mgr_positions_long", "asset_mgr_positions_short", "lev_money_positions_long",
            "lev_money_positions_short", "other_rept_positions_long", "other_rept_positions_short",
            "nonrept_positions_long_all", "nonrept_positions_short_all", "traders_tot_all"]


def fetch_cot(code: str = "209742") -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"$where": f"cftc_contract_market_code = '{code}'",
                                    "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": "1000",
                                    "$offset": str(offset)})
        batch = json.loads(get(f"{COT_URL}?{q}"))
        rows += batch
        offset += len(batch)
        if len(batch) < 1000:
            break
    raw = OUT / "cftc"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"tff_{code}.json").write_text(json.dumps(rows), encoding="utf-8")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in COT_COLS})
    out["as_of"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.normalize()
    out["released"] = out["as_of"] + pd.Timedelta(days=3)          # Friday 15:30 ET
    return out.set_index("released").sort_index()


# --- Cboe indices --------------------------------------------------------------------

CBOE_IDX = ("SKEW", "VVIX", "VIX9D", "VIX", "VIX3M", "VIX1D", "VXN")


def fetch_cboe_index(name: str) -> pd.Series:
    body = get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv")
    raw = OUT / "cboe_index"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{name}.csv").write_bytes(body)
    df = pd.read_csv(io.BytesIO(body))
    col = "CLOSE" if "CLOSE" in df.columns else name
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"), name=name)
    return s[s > 0].sort_index()


# --- Cboe put/call -------------------------------------------------------------------

PC_OLD = {"total": "totalpc", "equity": "equitypc", "index": "indexpc"}
PC_DAILY = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
PC_NEW_RATIO = {"total": "TOTAL PUT/CALL RATIO", "equity": "EQUITY PUT/CALL RATIO",
                "index": "INDEX PUT/CALL RATIO", "spx": "SPX + SPXW PUT/CALL RATIO"}
PC_NEW_OI = {"total": "SUM OF ALL PRODUCTS", "equity": "EQUITY OPTIONS", "index": "INDEX OPTIONS"}


def fetch_pc_old() -> pd.DataFrame:
    raw = OUT / "cboe_pc_history"
    raw.mkdir(parents=True, exist_ok=True)
    cols = {}
    for k, f in PC_OLD.items():
        body = get(f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{f}.csv")
        (raw / f"{f}.csv").write_bytes(body)
        lines = body.decode("utf-8", "replace").splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE"))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), skipinitialspace=True)
        df.columns = [c.strip().upper() for c in df.columns]
        d = pd.to_datetime(df["DATE"], format="mixed", errors="coerce")
        cols[f"pc_{k}"] = pd.Series(pd.to_numeric(df["P/C RATIO"], errors="coerce").to_numpy(), index=d)
    out = pd.DataFrame(cols)
    return out[out.index.notna()].sort_index()


def _daily_one(day: pd.Timestamp) -> dict | None:
    raw = OUT / "cboe_pc_daily"
    f = raw / f"{day:%Y-%m-%d}.json"
    if f.exists():
        body = f.read_bytes()
    else:
        try:
            body = get(PC_DAILY.format(d=f"{day:%Y-%m-%d}"), timeout=30)
        except urllib.error.HTTPError:
            return None                                   # holiday / not published
        f.write_bytes(body)
    try:
        d = json.loads(body)
    except ValueError:
        return None
    rat = {r.get("name"): r.get("value") for r in d.get("ratios") or []}
    row = {"date": day}
    for k, name in PC_NEW_RATIO.items():
        try:
            row[f"pc_{k}"] = float(rat.get(name))
        except (TypeError, ValueError):
            row[f"pc_{k}"] = float("nan")
    for k, sect in PC_NEW_OI.items():
        oi = next((r for r in d.get(sect) or [] if r.get("name") == "OPEN INTEREST"), None)
        row[f"oi_pc_{k}"] = (oi["put"] / oi["call"]) if oi and oi.get("call") else float("nan")
    return row


def fetch_pc_daily(start: str = "2019-10-07") -> pd.DataFrame:
    (OUT / "cboe_pc_daily").mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start, pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for r in ex.map(_daily_one, days) if r]
    return pd.DataFrame(rows).set_index("date").sort_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cot = fetch_cot()
    cot.to_parquet(OUT / "cot_nq_tff.parquet")
    print(f"CFTC NQ: {len(cot)} weeks, {cot['as_of'].min():%Y-%m-%d} .. {cot['as_of'].max():%Y-%m-%d}")
    idx = pd.DataFrame({n: fetch_cboe_index(n) for n in CBOE_IDX}).sort_index()
    idx.to_parquet(OUT / "cboe_indices.parquet")
    for n in CBOE_IDX:
        s = idx[n].dropna()
        print(f"Cboe {n}: {len(s)} days {s.index.min():%Y-%m-%d} .. {s.index.max():%Y-%m-%d}")
    old, new = fetch_pc_old(), fetch_pc_daily()
    pc = pd.concat([old[old.index < new.index.min()], new]).sort_index()
    pc.to_parquet(OUT / "cboe_putcall.parquet")
    print(f"Cboe put/call: {len(pc)} days {pc.index.min():%Y-%m-%d} .. {pc.index.max():%Y-%m-%d} "
          f"(daily files: {len(new)})")
    print(pc.tail(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
