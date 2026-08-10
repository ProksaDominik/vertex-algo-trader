"""What edge would 60% pass-in-10-days actually require?

Every tuning lever has now been swept: timeframes 1h-weekly, 10 strategy
families, targets 1R-8R, risk 0.5-10%, concurrency 2-30, time stops 1/2/3/5d,
session breakouts, stat-arb, volatility regimes, tight intraday targets,
parameter variants. Best 10-day result found: 32.3%.

Rather than tune further, this inverts the question. It takes the real trade
paths and SCALES every trade's R by a multiplier -- i.e. simulates having a
strategy k times better than the one that exists -- and finds the k needed to
hit 60% in 10 days. That converts "can we get there" into a measurable
distance.
"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
from canonical_strategy_test import RISK_USD, Windowed, load
from metatraderbot.backtest import Backtest
from metatraderbot.strategies_rsi import RSIReversionStrategy
from metatraderbot.strategies_bollinger import BollingerBounceStrategy

SYMS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURJPY",
        "GBPJPY","NZDJPY","AUDJPY","CHFJPY","GBPCHF","EURCHF","AUDCHF","EURAUD","AUDCAD"]
LEGS = [("RSI_1d", RSIReversionStrategy, {"rsi":{"period":14,"oversold":30,"overbought":70}}, 3.0, 1),
        ("Boll", BollingerBounceStrategy, {"bollinger":{"period":20,"std_dev":2.0}}, 4.0, None)]
DAILY_LIM, TOTAL_LIM, MIN_DAYS, TARGET = 0.05, 0.10, 4, 0.10

def _cell(job):
    li, sym = job
    _, cls, params, tp, mh = LEGS[li]
    try:
        b = load(sym,"1d")
        if b is None: return None
        b = b[b.index.dayofweek != 6]
        st = {"session":None,"exit":{"tp_r_multiple":tp,"sl_swing_lookback":10}}; st.update(params)
        cfg = {"symbol":sym,"capital":{"risk_usd_per_trade":RISK_USD,"initial_usd":10_000,
               "max_positions":2,"daily_stop_loss_usd":-1e9},"strategy":st}
        bt = Backtest(cfg,b,strategy=Windowed(cls(cfg))); bt.run()
        cl = b["close"]; out=[]
        for t in bt.closed:
            o=pd.Timestamp(t.entered_at).normalize(); c=pd.Timestamp(t.exited_at).normalize()
            rp=abs(t.entry_price-t.stop_loss)
            if rp<=0 or c<o: continue
            seg=cl.loc[o:c]; sg=1.0 if t.direction.name=="LONG" else -1.0
            path=(seg.values-t.entry_price)*sg/rp; path[-1]=t.pnl_usd/RISK_USD
            if mh is not None and len(path)>mh+1:
                cr=path[-1]-(seg.values[-1]-t.entry_price)*sg/rp
                path=path[:mh+1].copy(); path[-1]+=cr
            out.append({"open":o,"path":list(path),"final":float(path[-1])})
        return out
    except Exception: return None

def replay(by_day, days, i0, hz, risk, conc, per_day, k):
    eq, real, op, td = 1.0, 0.0, [], 0
    for j in range(hz):
        i=i0+j
        if i>=len(days): break
        d0=eq; opened=0
        for tr in by_day.get(days[i],[]):
            if len(op)>=conc or opened>=per_day: break
            op.append([tr,0]); opened+=1
        if opened: td+=1
        fl=0.0; still=[]
        for tr,age in op:
            age+=1
            if age<len(tr["path"]): fl+=tr["path"][age]*risk*tr.get("mult",1.0); still.append([tr,age])
            else: real+=tr["final"]*risk*tr.get("mult",1.0)
        op=still; eq=1.0+real+fl
        if eq<=d0-DAILY_LIM or eq<=1.0-TOTAL_LIM: return "fail"
        if eq>=1.0+TARGET and td>=MIN_DAYS: return "pass"
    return "unresolved"

if __name__=="__main__":
    jobs=[(i,s) for i in range(len(LEGS)) for s in SYMS]
    trades=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for c in ex.map(_cell,jobs,chunksize=8):
            if c: trades+=c
    by_day={}
    for t in trades: by_day.setdefault(t["open"],[]).append(t)
    days=sorted(by_day); starts=range(0,len(days)-1,4)
    print("="*88); print("HOW MUCH BETTER WOULD THE STRATEGY NEED TO BE?")
    print("k = multiplier on every trade's R (k=1 is the real strategy)")
    print("="*88)
    # PROPER TEST: raise the WIN RATE by converting a fraction of losers into
    # winners, keeping trade SIZES unchanged. That is what "a better strategy"
    # means -- more edge without more volatility. Scaling R (the previous test)
    # confounds edge with volatility and is why it showed no improvement.
    import random
    rng = random.Random(7)
    losers = [t for t in trades if t["final"] < 0]
    base_wr = 1 - len(losers)/len(trades)
    print(f"  actual win rate: {base_wr:.1%}, {len(trades)} trades")
    print(f"  {'flip%':>7}{'new win rate':>15}{'10-day pass':>14}{'30-day pass':>14}")
    for frac in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        for t in trades: t["mult"] = 1.0
        chosen = rng.sample(losers, int(len(losers)*frac))
        for t in chosen:
            t["mult"] = -1.0          # flip that loser into an equal-size winner
        wr = base_wr + (1-base_wr)*frac
        best10=best30=0.0
        for risk,conc,pd_ in ((0.02,10,5),(0.03,10,5),(0.015,15,8)):
            o10=[replay(by_day,days,i,10,risk,conc,pd_,1.0) for i in starts]
            o30=[replay(by_day,days,i,30,risk,conc,pd_,1.0) for i in starts]
            best10=max(best10,o10.count("pass")/len(o10))
            best30=max(best30,o30.count("pass")/len(o30))
        print(f"  {frac:>6.0%}{wr:>15.1%}{best10:>14.1%}{best30:>14.1%}")
    for k in ():
        best10=best30=0.0
        for risk,conc,pd_ in ((0.02,10,5),(0.03,10,5),(0.015,15,8)):
            o10=[replay(by_day,days,i,10,risk,conc,pd_,k) for i in starts]
            o30=[replay(by_day,days,i,30,risk,conc,pd_,k) for i in starts]
            best10=max(best10,o10.count("pass")/len(o10))
            best30=max(best30,o30.count("pass")/len(o30))
        print(f"  {k:>5.1f}{base_e*k:>+16.3f}{best10:>14.1%}{best30:>14.1%}")
