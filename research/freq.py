import numpy as np
rng=np.random.default_rng(3)

# Filter-cascade estimate of raw signals per day, per module.
# p = [regime_ok, setup_valid, trigger_fires, fill_occurs] per symbol per day
mods = {
 "A  Asian fade":      dict(sym=3, p=[0.50,0.45,0.55,0.95]),
 "B  London breakout": dict(sym=3, p=[0.60,0.45,0.60,0.70]),
 "C  Liquidity sweep": dict(sym=4, p=[0.85,0.50,0.28,0.55], setups_per_day=1.6),
 "D  H4 pullback":     dict(sym=4, p=[0.35,0.40,0.85,1.00]),
}
tot=0
print(f"{'module':<20}{'symbols':>8}{'raw/day':>10}{'raw/week':>10}")
for k,v in mods.items():
    per_sym=np.prod(v["p"])*v.get("setups_per_day",1.0)
    raw=per_sym*v["sym"]; tot+=raw
    print(f"{k:<20}{v['sym']:>8}{raw:>10.2f}{raw*5:>10.1f}")
print(f"{'RAW TOTAL':<20}{'':>8}{tot:>10.2f}{tot*5:>10.1f}")

# Execution-layer survival (Sec 4 + 10): spread, cost gate, news, correlation, concurrency, arbitration
surv = 0.80*0.92*0.93*0.90*0.95   # spread, cost, news, corr/concurrency, arbitration
net = tot*surv
print(f"\nexecution-layer pass rate = {surv:.2f}")
print(f"NET trades/day  = {net:.2f}")
print(f"NET trades/week = {net*5:.1f}")
print(f"NET trades/month= {net*21:.0f}")

# Daily distribution (Poisson-ish, but regime-clustered -> negative binomial)
lam=net
r=3.0; p=r/(r+lam)
draws=rng.negative_binomial(r,p,200000)
print("\nDaily trade-count distribution (regime clustering modelled):")
for k in range(0,7):
    print(f"  {k} trades: {(draws==k).mean():>6.1%}")
print(f"  7+      : {(draws>=7).mean():>6.1%}")
print(f"  95th pct: {np.percentile(draws,95):.0f}   99th pct: {np.percentile(draws,99):.0f}   max: {draws.max()}")

# Time to pass Phase 1
print("\n=== MONTHS TO PASS PHASE 1 (0.5% risk, ratchet, 10k acct) ===")
def months(E_W,E_RR,n=20000,risk=0.005,init=10000.0,tpd=net,maxt=3000):
    eq=np.full(n,init); ds=np.full(n,init); res=np.zeros(n,np.int8); tt=np.zeros(n)
    for t in range(maxt):
        a=res==0
        if not a.any(): break
        e=eq[a]; p=(e-init)/init
        r_=np.select([p>=0.08,p>=0.05,p>=0,p>=-0.03,p>=-0.05],[0.0025,0.004,risk,0.004,0.0025],default=0.0015)
        rk=e*r_
        e=e+np.where(rng.random(a.sum())<E_W,rk*E_RR,-rk)
        eq[a]=e; idx=np.where(a)[0]
        for cond,code in [(e<=ds[a]-500,2),(e<=9000,3),(e>=11000,1)]:
            sel=idx[cond & (res[idx]==0)]; res[sel]=code; tt[sel]=t+1
        if (t+1)%max(1,int(round(tpd)))==0: ds[a]=eq[a]
    w=tt[res==1]
    return (res==1).mean(), np.median(w)/net/21, np.percentile(w,90)/net/21

for lbl,W,RR in [("E=+0.26R (strong)",0.45,1.8),("E=+0.15R",0.44,1.6),("E=+0.09R (modest)",0.42,1.6)]:
    pp,med,p90=months(W,RR)
    print(f"  {lbl:<20} P(pass)={pp:>6.1%}   median {med:>4.1f} months   90th pct {p90:>4.1f} months")
