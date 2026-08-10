# Walk-forward validation — Phase 5

70% train / 30% validation split per pair (last ~5.5 months of 30m
bars; validation = final ~7 weeks, strictly out-of-sample for the
original sweeps). Fills bar-close idealized; partial-take not
modeled; trail modeled at bar-close granularity (live trails
tighter). before = tp 0.333R / trail 0.15R; after = tp 3R / trail 1R.

## train window

| book | stack | slots | trades | net P&L | avg PF | positive slots |
|---|---|---|---|---|---|---|
| FTMO | before | 19 | 7105 | -60023 | 0.66 | 2/19 |
| FTMO | after | 19 | 2200 | -24024 | 0.85 | 3/19 |
| YOLO | before | 21 | 9050 | -20744 | 0.68 | 2/21 |
| YOLO | after | 21 | 4091 | -7538 | 0.88 | 6/21 |

## valid window

| book | stack | slots | trades | net P&L | avg PF | positive slots |
|---|---|---|---|---|---|---|
| FTMO | before | 19 | 3103 | -27670 | 0.63 | 1/19 |
| FTMO | after | 19 | 1000 | -7120 | 0.91 | 4/19 |
| YOLO | before | 21 | 3857 | -8741 | 0.68 | 1/21 |
| YOLO | after | 21 | 1742 | -4467 | 0.86 | 4/21 |

Per-slot rows: see walkforward_phase5.csv