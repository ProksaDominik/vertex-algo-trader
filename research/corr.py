import numpy as np
rng=np.random.default_rng(99)

def sim(tpd, risk, W, RR, rho=0.0, days=21, n=30000, init=10000.0, unlimited=False):
    """rho = within-day correlation between concurrent trades.
       Modelled via a shared daily latent factor driving outcomes."""
    if unlimited: days=400
    eq=np.full(n,init); ds=np.full(n,init); res=np.zeros(n,np.int8)
    thr=W
    for d in range(days):
        ds=np.where(res==0, eq, ds)
        # shared daily factor
        z_day=rng.standard_normal(n)
        for k in range(tpd):
            a=res==0
            if not a.any(): return res
            e=eq[a]; p=(e-init)/init
            r=np.select([p>=0.08,p>=0.05,p>=0,p>=-0.03,p>=-0.05],
                        [0.0025,0.004,risk,0.004,0.0025],default=0.0015)
            rk=e*r
            z=np.sqrt(rho)*z_day[a]+np.sqrt(1-rho)*rng.standard_normal(a.sum())
            # map latent to win/loss preserving marginal win rate W
            from scipy.stats import norm
            win = z < norm.ppf(thr)
            e=e+np.where(win,rk*RR,-rk)
            eq[a]=e
            idx=np.where(a)[0]
            res[idx[e<=ds[a]-500]]=2
            m=res[idx]==0; res[idx[m&(e<=9000)]]=3
            m=res[idx]==0; res[idx[m&(e>=11000)]]=1
    return res

def r_(res):
    return (res==1).mean(),(res==2).mean(),(res==3).mean(),(res==0).mean()

print("=== CORRELATION PENALTY (modest edge +0.09R, 0.5% risk, 21 days) ===")
print("rho = correlation between trades opened the same day")
print(f"{'tpd':>4}{'rho=0.0':>10}{'rho=0.2':>10}{'rho=0.4':>10}{'rho=0.6':>10}   <- P(pass in 21d)")
for tpd in [5,8,12]:
    cells=[]
    for rho in [0.0,0.2,0.4,0.6]:
        p,d,t,u=r_(sim(tpd,0.005,0.42,1.6,rho=rho))
        cells.append(f"{p*100:>9.1f}%")
    print(f"{tpd:>4}"+"".join(cells))

print(f"\n{'tpd':>4}{'rho=0.0':>10}{'rho=0.2':>10}{'rho=0.4':>10}{'rho=0.6':>10}   <- P(FAIL daily limit)")
for tpd in [5,8,12]:
    cells=[]
    for rho in [0.0,0.2,0.4,0.6]:
        p,d,t,u=r_(sim(tpd,0.005,0.42,1.6,rho=rho))
        cells.append(f"{d*100:>9.1f}%")
    print(f"{tpd:>4}"+"".join(cells))

print("\n=== SAME, at 1% risk (the tempting choice) — P(FAIL daily limit) ===")
print(f"{'tpd':>4}{'rho=0.0':>10}{'rho=0.2':>10}{'rho=0.4':>10}{'rho=0.6':>10}")
for tpd in [5,8,12]:
    cells=[]
    for rho in [0.0,0.2,0.4,0.6]:
        p,d,t,u=r_(sim(tpd,0.01,0.42,1.6,rho=rho))
        cells.append(f"{d*100:>9.1f}%")
    print(f"{tpd:>4}"+"".join(cells))

print("\n=== THE KEY COMPARISON: 8 trades/day, 0.5% risk, rho=0.3 ===")
res21=sim(8,0.005,0.42,1.6,rho=0.3,days=21)
resU =sim(8,0.005,0.42,1.6,rho=0.3,unlimited=True)
p,d,t,u=r_(res21); print(f"  within 21 days : pass {p:.1%}  fail {d+t:.1%}  still running {u:.1%}")
p,d,t,u=r_(resU);  print(f"  unlimited time : pass {p:.1%}  fail {d+t:.1%}  still running {u:.1%}")
