import numpy as np
rng=np.random.default_rng(7)

def sim(risk, W, RR, n=20000, tpd=4, init=10000.0, maxt=1500, ratchet=True, edge_sd=0.0):
    eq=np.full(n,init); ds=np.full(n,init); res=np.zeros(n,dtype=np.int8)  #0 unresolved 1 pass 2 daily 3 total
    w = np.full(n,W) if edge_sd==0 else np.clip(rng.normal(W,edge_sd,n),0.05,0.95)
    for t in range(maxt):
        live = res==0
        if not live.any(): break
        e=eq[live]
        if ratchet:
            p=(e-init)/init
            r=np.select([p>=0.08,p>=0.05,p>=0.0,p>=-0.03,p>=-0.05],
                        [0.0025,0.0040,risk,0.0040,0.0025],default=0.0015)
        else:
            r=np.full(e.shape,risk)
        rk=e*r
        win=rng.random(live.sum())<w[live]
        e=e+np.where(win,rk*RR,-rk)
        eq[live]=e
        idx=np.where(live)[0]
        res[idx[e<=ds[live]-500]]=2
        res[idx[(e<=9000)&(res[idx]==0)]]=3
        res[idx[(e>=11000)&(res[idx]==0)]]=1
        if (t+1)%tpd==0: ds[live]=eq[live]
    n_=float(n)
    return (res==1).sum()/n_,(res==2).sum()/n_,(res==3).sum()/n_,(res==0).sum()/n_

def show(title,rows):
    print(f"\n=== {title} ===")
    for r in rows: print(r)

rows=[f"{'risk':>6} {'P(pass)':>8} {'P(fail_daily)':>14} {'P(fail_total)':>14} {'unresolved':>11}"]
for r in [0.0025,0.005,0.01,0.02,0.03]:
    p,d,t,u=sim(r,0.42,1.6)
    rows.append(f"{r*100:>5.2f}% {p:>8.1%} {d:>14.1%} {t:>14.1%} {u:>11.1%}")
show("A. Modest real edge  W=.42 RR=1.6  E_net=+0.092R",rows)

rows=[]
for sd in [0.0,0.03,0.05,0.08,0.10]:
    p,d,t,u=sim(0.005,0.45,1.8,edge_sd=sd)
    rows.append(f"  sd(W)={sd:.2f}   P(pass)={p:>6.1%}   P(fail)={d+t:>6.1%}   unresolved={u:>5.1%}")
show("B. Parameter risk: backtest W=.45 RR=1.8, TRUE W ~ N(.45,sd), risk 0.5%",rows)

rows=[]
for lbl,W in [("E_net = 0.00R",0.3571),("E_net = -0.05R",0.34),("E_net = -0.15R",0.30)]:
    for r in [0.005,0.02]:
        p,d,t,u=sim(r,W,1.8)
        rows.append(f"  {lbl}  risk={r*100:>4.1f}%   P(pass)={p:>6.1%}   P(fail)={d+t:>6.1%}")
show("C. Zero / negative edge (RR=1.8)",rows)

rows=[f"{'risk':>6} {'P(pass)':>8} {'P(fail_daily)':>14} {'P(fail_total)':>14} {'unresolved':>11}"]
for r in [0.0025,0.005,0.0075,0.01,0.015,0.02,0.03]:
    p,d,t,u=sim(r,0.42,1.5)
    rows.append(f"{r*100:>5.2f}% {p:>8.1%} {d:>14.1%} {t:>14.1%} {u:>11.1%}")
show("D. Marginal edge W=.42 RR=1.5  E_net=+0.050R",rows)

rows=[]
for tpd in [2,4,8,15,25]:
    p,d,t,u=sim(0.01,0.42,1.5,tpd=tpd)
    rows.append(f"  {tpd:>2} trades/day  P(pass)={p:>6.1%}  P(fail_daily)={d:>6.1%}  P(fail_total)={t:>6.1%}")
show("E. Trade frequency vs daily-limit risk (marginal edge, 1% risk)",rows)

rows=[]
for r in [0.005,0.01,0.02]:
    p,d,t,u=sim(r,0.45,1.8,init=10000.0,maxt=1500)
    rows.append(f"  risk={r*100:>4.1f}%  P(pass)={p:>6.1%}  P(fail)={d+t:>6.1%}")
show("F. Strong edge W=.45 RR=1.8  E_net=+0.260R (for reference)",rows)
