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
    return PAIR_SPECS.get(symbol, PAIR_SPECS["XAUUSD"])


def contract_mult(config_or_symbol) -> float:
    return get_specs(config_or_symbol)["contract_mult"]
