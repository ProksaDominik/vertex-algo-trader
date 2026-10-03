---
type: experiment
date: {{date:YYYY-MM-DD}}
status: running   # running | done | abandoned
verdict:          # adopt | reject | inconclusive
tags: [experiment]
---

# {{title}}

## Hypothesis
What should be true, and why.

## Setup
Script, data window (in-sample / out-of-sample), symbols, parameters.
Command: `python scripts/...`

## Result
Key numbers (trades, expectancy in R, win rate, max drawdown) + output file.

## Verdict
Adopt / reject / inconclusive — and what sample size would make it decidable.
