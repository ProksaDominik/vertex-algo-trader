import numpy as np
rng=np.random.default_rng(11)

print("=== LOT-SIZE CONSTRAINT ON A $200 ACCOUNT ===")
print("Broker min lot 0.01 (micro). EURUSD 0.01 lot => $0.10 per pip.\n")
print(f"{'stop (pips)':>12} {'$ risk @0.01':>13} {'% of $200':>10} {'cost ratio*':>12} {'verdict':>22}")
for sp in [10,15,20,30,50,80,120]:
    d=sp*0.10; pct=d/200*100; cost=1.5/sp
    v = "OK" if (pct<=3 and cost<=0.08) else ("cost too high" if cost>0.08 else "risk too high")
    print(f"{sp:>12} {d:>12.2f} {pct:>9.2f}% {cost:>12.3f} {v:>22}")
print("*cost ratio = 1.5 pips all-in cost / stop distance, in R. Gate = 0.08 (Sec 1.1)\n")

def grow(risk, W, RR, months=12, tpm=20, n=20000, init=200.0, min_stake=None, ruin=40.0):
    eq=np.full(n,init); dead=np.zeros(n,bool)
    for t in range(months*tpm):
        a=~dead
        rk=eq[a]*risk
        if min_stake is not None: rk=np.maximum(rk,min_stake)
        win=rng.random(a.sum())<W
        eq[a]=eq[a]+np.where(win,rk*RR,-rk)
        dead |= eq<=ruin
        eq[dead]=ruin
    return eq,dead

def rep(label,risk,W,RR,min_stake=None):
    eq,dead=grow(risk,W,RR,min_stake=min_stake)
    alive=~dead
    print(f"{label}")
    print(f"   P(ruin<=$40 in 12mo) = {dead.mean():>6.1%}")
    print(f"   median final equity  = ${np.median(eq):>8.2f}")
    print(f"   mean final equity    = ${eq.mean():>8.2f}")
    print(f"   10th pct / 90th pct  = ${np.percentile(eq,10):>7.2f} / ${np.percentile(eq,90):>8.2f}")
    print(f"   P(final > $400)      = {(eq>400).mean():>6.1%}   P(final > $1000) = {(eq>1000).mean():>5.1%}\n")

print("=== PLAN 1: SCALED-DOWN (same system, forced 2.5%/trade by min lot) ===")
print("    Module D only (wide stops keep cost ratio low). ~20 trades/month.")
rep("  Modest edge W=.45 RR=1.8 (E=+0.26R):", 0.025, 0.45, 1.8)
rep("  Marginal edge W=.42 RR=1.6 (E=+0.09R):", 0.025, 0.42, 1.6)
rep("  No edge W=.357 RR=1.8 (E=0.00R):", 0.025, 0.3571, 1.8)

print("=== PLAN 2: AGGRESSIVE COMPOUNDING ===")
for r in [0.05,0.10,0.20]:
    print(f"-- risk {r*100:.0f}% per trade --")
    rep(f"  Modest edge W=.45 RR=1.8:", r, 0.45, 1.8)
    rep(f"  No edge W=.357 RR=1.8:", r, 0.3571, 1.8)

print("=== TIME TO REACH $10,000 FROM $200 (modest edge, 20 trades/mo) ===")
for r in [0.025,0.05,0.10]:
    eq,dead=grow(r,0.45,1.8,months=36)
    print(f"  risk {r*100:>4.1f}%: P(reach>$10k in 36mo)={(eq>10000).mean():>6.1%}  P(ruin)={dead.mean():>6.1%}  median=${np.median(eq):>9.2f}")
