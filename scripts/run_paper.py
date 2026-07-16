"""
Phase 2 entrypoint: run a portfolio in live paper mode on MT5 demo.

Prereqs:
- MT5 desktop is RUNNING and logged into your IC Markets demo account
- AlgoTrading enabled (green button in MT5 toolbar)
- Python venv activated: `.venv\\Scripts\\Activate.ps1`

Usage:
    python scripts/run_paper.py                                  # default = portfolio_ftmo.yaml
    python scripts/run_paper.py --config config/portfolio_ftmo.yaml
    python scripts/run_paper.py --config config/portfolio_yolo.yaml

Ctrl+C to stop gracefully. Every trade + heartbeat is logged to the sqlite
ledger declared in the config's `ledger.path`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load MT5 credentials from .env if present (safe no-op when file missing).
load_dotenv(ROOT / ".env")

from metatraderbot.ledger import Ledger  # noqa: E402
from metatraderbot.mt5_engine import MT5Engine  # noqa: E402
from metatraderbot.portfolio_runner import PortfolioRunner  # noqa: E402


def setup_logging(log_suffix: str) -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"paper_{log_suffix}.log", encoding="utf-8"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a portfolio in MT5 paper mode.")
    parser.add_argument(
        "--config",
        default="config/portfolio_ftmo.yaml",
        help="Path to portfolio yaml (default: config/portfolio_ftmo.yaml)",
    )
    args = parser.parse_args()

    portfolio_path = (ROOT / args.config).resolve()
    if not portfolio_path.exists():
        # Try relative to cwd as fallback
        alt = Path(args.config).resolve()
        if alt.exists():
            portfolio_path = alt
        else:
            print(f"[FAIL] Config not found: {portfolio_path}")
            return 2

    log_suffix = portfolio_path.stem  # e.g. "portfolio_ftmo" -> "portfolio_ftmo"
    setup_logging(log_suffix)
    log = logging.getLogger("run_paper")

    with open(portfolio_path) as f:
        cfg = yaml.safe_load(f)

    log.info(f"Loaded portfolio: {portfolio_path.name}")

    log.info("=" * 60)
    log.info("MetaTraderBot — Phase 2 Paper Trading")
    log.info("=" * 60)
    log.info(f"Portfolio configs: {len(cfg['configs'])}")
    for c in cfg["configs"]:
        log.info(
            f"  {c['name']}: {c['symbol']} {c['strategy']} {c['timeframe']} "
            f"(magic={c['magic_number']})"
        )
    log.info(f"Risk per trade: ${cfg['capital']['risk_usd_per_trade']}")
    log.info(f"Daily stop: ${cfg['capital']['daily_stop_loss_usd']}")
    log.info(f"Max total DD: ${cfg['capital']['max_total_drawdown_usd']}")
    log.info("=" * 60)

    ledger_path = ROOT / cfg.get("ledger", {}).get("path", "data/paper_ledger.sqlite")
    ledger = Ledger(ledger_path)
    log.info(f"Ledger: {ledger_path}")

    # Shared start/stop control file — the dashboard writes to this path to
    # pause/resume new trade entries without touching already-open positions.
    control_path = ROOT / "data" / f"control_{log_suffix}.json"
    log.info(f"Control file: {control_path}")

    try:
        engine = MT5Engine(symbol_map=cfg.get("mt5", {}).get("symbol_map", {}))
    except RuntimeError as e:
        log.error(str(e))
        return 3

    runner = PortfolioRunner(cfg, engine, ledger, control_path=control_path)
    try:
        runner.initialize()
    except Exception as e:
        log.exception(f"Initialization failed: {e}")
        engine.disconnect()
        ledger.close()
        return 4

    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
