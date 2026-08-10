"""Per-pair contract specifications for backtesting.

Approximated from IC Markets Raw Spread account defaults. Real broker specs
vary slightly; these are close enough for strategy selection. Phase 2 will
replace these with live MT5 symbol_info calls.

contract_mult: $ P&L per lot per $1 price move
spread_price:  half-spread bid/ask cost in PRICE units (native quote resolution)
commission_per_lot: $ round-turn commission per standard lot
"""
from __future__ import annotations

PAIR_SPECS: dict[str, dict[str, float]] = {
    "XAUUSD": {
        # 1 standard lot = 100 troy oz; $1 gold move = $100 per lot.
        "contract_mult": 100.0,
        "spread_price": 0.30,        # ~3 pips on Raw Spread
        "commission_per_lot": 3.50,
    },
    "EURUSD": {
        # 1 standard lot = 100k EUR; $1 EURUSD move = $100k per lot (huge, 10k pips).
        "contract_mult": 100_000.0,
        "spread_price": 0.00005,     # ~0.5 pip
        "commission_per_lot": 3.50,
    },
    "GBPUSD": {
        "contract_mult": 100_000.0,
        "spread_price": 0.00008,     # ~0.8 pip
        "commission_per_lot": 3.50,
    },
    "USDJPY": {
        # 1 standard lot = 100k USD; 1 JPY move at USDJPY=150 ~= $667 per lot.
        # 700 is a stable-ish approximation across a range of 140-160.
        "contract_mult": 700.0,
        "spread_price": 0.006,       # ~0.6 pip on Raw Spread
        "commission_per_lot": 3.50,
    },
    "US500": {
        # 1 CFD lot ~= 1 index unit on Dukascopy/typical CFD brokers.
        "contract_mult": 1.0,
        "spread_price": 0.50,        # ~0.5 point
        "commission_per_lot": 3.50,
    },

    # --- NON-FX, added 2026-08-05 -------------------------------------------
    # Contract multipliers are READ FROM THE LIVE MT5 TERMINAL
    # (symbol_info.trade_contract_size); spreads are ESTIMATES, since several
    # symbols returned a zero tick when queried outside their session.
    #
    # Treat every result computed with these as PROVISIONAL. Transaction costs
    # have been the binding constraint in this project all along (SL exits
    # realise -1.054R not -1.000R, and live fills slipped 2-12x the modelled
    # spread), so a wrong spread here would manufacture an edge that does not
    # exist. Verify against real fills before trading any of them.
    #
    # Broker ticker differs from the key for several: NAS100 -> "USTEC",
    # JP225 -> "JPN225", BTCUSD -> "BTC", ETHUSD -> "ETH". Those need
    # mt5.symbol_map entries in the config.
    "US30": {
        "contract_mult": 1.0,
        "spread_price": 1.20,        # measured live
        "commission_per_lot": 3.50,
    },
    "NAS100": {
        "contract_mult": 1.0,
        "spread_price": 1.50,        # ESTIMATE (zero tick when queried)
        "commission_per_lot": 3.50,
    },
    "DE40": {
        "contract_mult": 1.0,
        "spread_price": 0.50,        # measured live
        "commission_per_lot": 3.50,
    },
    "JP225": {
        "contract_mult": 100.0,      # live contract size
        "spread_price": 8.0,         # ESTIMATE, index points
        "commission_per_lot": 3.50,
    },
    "WTI": {
        "contract_mult": 1.0,
        "spread_price": 0.03,        # ESTIMATE, ~3 cents on crude
        "commission_per_lot": 3.50,
    },
    "BRENT": {                        # research only -- broker does NOT offer
        "contract_mult": 1.0,
        "spread_price": 0.03,
        "commission_per_lot": 3.50,
    },
    "COPPER": {                       # research only -- broker does NOT offer
        "contract_mult": 1.0,
        "spread_price": 0.0015,
        "commission_per_lot": 3.50,
    },
    "XPTUSD": {
        "contract_mult": 1.0,
        "spread_price": 2.50,        # ESTIMATE, platinum is wide
        "commission_per_lot": 3.50,
    },
    "BTCUSD": {
        "contract_mult": 1.0,
        "spread_price": 25.0,        # ESTIMATE; crypto spreads are very wide
        "commission_per_lot": 3.50,
    },
    "ETHUSD": {
        "contract_mult": 1.0,
        "spread_price": 2.00,        # ESTIMATE
        "commission_per_lot": 3.50,
    },
    "AUDUSD": {
        "contract_mult": 100_000.0,
        "spread_price": 0.00010,     # ~1.0 pip
        "commission_per_lot": 3.50,
    },
    "USDCAD": {
        # USD base at ~1.35 CAD/USD: 1 CAD move ≈ $74k per lot
        "contract_mult": 74_000.0,
        "spread_price": 0.00012,     # ~1.2 pip
        "commission_per_lot": 3.50,
    },
    "USDCHF": {
        # USD base at ~0.88 CHF/USD: 1 CHF move ≈ $114k per lot
        "contract_mult": 114_000.0,
        "spread_price": 0.00015,     # ~1.5 pip
        "commission_per_lot": 3.50,
    },
    "NZDUSD": {
        "contract_mult": 100_000.0,
        "spread_price": 0.00015,     # ~1.5 pip
        "commission_per_lot": 3.50,
    },
    # --- Added after the market-expansion sweep (backtested vs the existing
    # proven Donchian/MACD configs; approximations same style as above). ---
    "EURJPY": {"contract_mult": 700.0, "spread_price": 0.008, "commission_per_lot": 3.50},
    "GBPJPY": {"contract_mult": 700.0, "spread_price": 0.010, "commission_per_lot": 3.50},
    "NZDJPY": {"contract_mult": 700.0, "spread_price": 0.010, "commission_per_lot": 3.50},
    "AUDJPY": {"contract_mult": 700.0, "spread_price": 0.008, "commission_per_lot": 3.50},
    "CHFJPY": {"contract_mult": 700.0, "spread_price": 0.010, "commission_per_lot": 3.50},
    "GBPCHF": {"contract_mult": 114_000.0, "spread_price": 0.00018, "commission_per_lot": 3.50},
    "EURCHF": {"contract_mult": 114_000.0, "spread_price": 0.00015, "commission_per_lot": 3.50},
    "AUDCHF": {"contract_mult": 114_000.0, "spread_price": 0.00018, "commission_per_lot": 3.50},
    # AUD-quoted (EUR/AUD): convert via AUDUSD (~0.69) since AUD is USD-base-style quoted.
    "EURAUD": {"contract_mult": 69_000.0, "spread_price": 0.00018, "commission_per_lot": 3.50},
    # CAD-quoted, same conversion logic as USDCAD.
    "AUDCAD": {"contract_mult": 74_000.0, "spread_price": 0.00015, "commission_per_lot": 3.50},
    "US30": {"contract_mult": 1.0, "spread_price": 2.0, "commission_per_lot": 3.50},
    "DE40": {"contract_mult": 1.0, "spread_price": 1.0, "commission_per_lot": 3.50},
}


def get_specs(config_or_symbol) -> dict[str, float]:
    if isinstance(config_or_symbol, dict):
        symbol = config_or_symbol.get("symbol", "XAUUSD")
    else:
        symbol = str(config_or_symbol)
    specs = PAIR_SPECS.get(symbol)
    if specs is None:
        # The old silent fallback to XAUUSD specs (contract_mult=100) would
        # size an unknown FX pair ~1000x too large (AUDIT C5). Refuse loudly.
        raise ValueError(
            f"No contract specs for symbol {symbol!r}. Add it to "
            f"instruments.PAIR_SPECS (backtests) — live sizing uses broker "
            f"symbol_info and is unaffected once the symbol exists here."
        )
    return specs


def contract_mult(config_or_symbol) -> float:
    return get_specs(config_or_symbol)["contract_mult"]


def pip_size(symbol: str) -> float:
    """Standard pip size for strategies that think in pips (SL/TP buffers)
    rather than raw price distance. JPY pairs and gold quote at a coarser
    resolution than the other majors; indices use whole points."""
    s = symbol.upper()
    if s.endswith("JPY"):
        return 0.01
    if s == "XAUUSD":
        return 0.10
    if s in ("US30", "DE40", "US500"):
        return 1.0
    return 0.0001
