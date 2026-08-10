# Strategy comparison — which is best (out-of-sample)

Validation window, 10-symbol basket, live 1R trail modelled, TP 2R.

| strategy | symbols | trades | net P&L | avg exp $/trade | avg PF | positive symbols |
|---|---|---|---|---|---|---|
| Donchian | 10 | 569 | +1270 | +4.11 | 1.14 | 5/10 |
| MACD | 10 | 454 | -2728 | -5.82 | 0.91 | 3/10 |
| BOS | 10 | 328 | -2707 | -7.86 | 0.91 | 3/10 |
| VWAPReversion | 10 | 912 | -14108 | -16.93 | 0.79 | 3/10 |
| SweepReclaim | 10 | 2039 | -53673 | -25.90 | 0.60 | 0/10 |
| TODReversion | 10 | 284 | -8284 | -29.66 | 0.66 | 2/10 |