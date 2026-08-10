import numpy as np
rng=np.random.default_rng(21)

def sim(tpd, risk, W, RR, days=21, n=30000, init=10000.0, ratchet=True):
    """Explicit day loop so the 5% daily limit binds correctly."""
    eq=np.full(n,init); res=np.zeros(n,np.int8); tdone=np.zeros(n)
    for d in range(days):
        a=res==0
        if not a.any(): break
        ds=eq[a].copy()                      # daily reference at rollover
        for k in range(tpd):
            live=res==0
            if not live.any(): break
            e=eq[live]; p=(e-init)/init
            if ratchet:
                r=np.select([p>=0.08,p>=0.05,p>=0,p>=-0.03,p>=-0.05],
                            [0.0025,0.004,risk,0.004,0.0025],default=0.0015)
            else: r=np.full(e.shape,risk)
            rk=e*r
            e=e+np.where(rng.random(live.sum())<W,rk*RR,-rk)
            eq[live]=e
            idx=np.where(live)[0]
            dsl=ds[np.isin(np.where(res==0)[0],idx)] if False else None
            # daily ref aligned to currently-live subset
            # rebuild: track ds over full array instead
        # (handled below)
    return None

# cleaner: full-array daily reference
def sim2(tpd, risk, W, RR, days=21, n=30000, init=10000.0, ratchet=True):
    eq=np.full(n,init); ds=np.full(n,init); res=np.zeros(n,np.int8)
    for d in range(days):
        ds=np.where(res==0, eq, ds)                      # reset daily ref
        for k in range(tpd):
            a=res==0
            if not a.any(): return res
            e=eq[a]; p=(e-init)/init
            if ratchet:
                r=np.select([p>=0.08,p>=0.05,p>=0,p>=-0.03,p>=-0.05],
                            [0.0025,0.004,risk,0.004,0.0025],default=0.0015)
            else: r=np.full(e.shape,risk)
            rk=e*r
            e=e+np.where(rng.random(a.sum())<W,rk*RR,-rk)
            eq[a]=e
            idx=np.where(a)[0]
            res[idx[e<=ds[a]-500]]=2
            m=res[idx]==0
            res[idx[m & (e<=9000)]]=3
            m=res[idx]==0
            res[idx[m & (e>=11000)]]=1
    return res

def row(res):
    n=len(res)
    return (res==1).mean(),(res==2).mean(),(res==3).mean(),(res==0).mean()

for lbl,W,RR in [("MODEST EDGE  E=+0.09R (W=.42 RR=1.6)",0.42,1.6),
                 ("STRONG EDGE  E=+0.26R (W=.45 RR=1.8)",0.45,1.8)]:
    print(f"\n=== {lbl} — P(pass within 21 trading days) ===")
    print(f"{'':>12}" + "".join(f"{f'{r*100:.2f}%':>10}" for r in [0.005,0.0075,0.01,0.015,0.02]))
    for tpd in [1,2,3,5,8,12]:
        cells=[]
        for r in [0.005,0.0075,0.01,0.015,0.02]:
            p,d,t,u=row(sim2(tpd,r,W,RR))
            cells.append(f"{p*100:>9.1f}%")
        print(f"{tpd:>3} trd/day " + "".join(cells))

print("\n=== FAILURE BREAKDOWN, modest edge, 21 days ===")
print(f"{'tpd':>4}{'risk':>8}{'P(pass)':>9}{'P(fail_daily)':>15}{'P(fail_total)':>15}{'unresolved':>12}")
for tpd in [1,3,5,8,12]:
    for r in [0.005,0.01,0.02]:
        p,d,t,u=row(sim2(tpd,r,0.42,1.6))
        print(f"{tpd:>4}{r*100:>7.1f}%{p:>9.1%}{d:>15.1%}{t:>15.1%}{u:>12.1%}")

print("\n=== SAME CONFIGS, NO DEADLINE (unlimited time) — the comparison that matters ===")
print(f"{'tpd':>4}{'risk':>8}{'P(pass) 21d':>13}{'P(pass) unlimited':>19}{'P(fail) ever':>14}")
for tpd in [3,5,8]:
    for r in [0.005,0.01,0.02]:
        p21,_,_,_=row(sim2(tpd,r,0.42,1.6,days=21))
        pu,du,tu,uu=row(sim2(tpd,r,0.42,1.6,days=400))
        print(f"{tpd:>4}{r*100:>7.1f}%{p21:>13.1%}{pu:>19.1%}{du+tu:>14.1%}")
