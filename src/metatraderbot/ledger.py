"""SQLite ledger for paper trading — every signal, order, close, and heartbeat.

Tables:
- signals:          strategy decisions (before order placement)
- orders:           MT5 order results (fill or rejection)
- closed_positions: reconciled trade outcomes (P&L, exit reason)
- heartbeats:       account state samples (~1/min)

Simple sqlite3 stdlib, no ORM. Read directly with `sqlite3 data/paper_ledger.sqlite`.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  config_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry_price REAL NOT NULL,
  stop_loss REAL NOT NULL,
  take_profit REAL NOT NULL,
  lots REAL NOT NULL,
  bar_index INTEGER
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  signal_id INTEGER,
  ticket INTEGER,
  config_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL,
  requested_price REAL,
  filled_price REAL,
  requested_lots REAL,
  filled_lots REAL,
  sl REAL,
  tp REAL,
  retcode INTEGER,
  comment TEXT,
  FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS closed_positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  closed_ts TEXT NOT NULL,
  opened_ts TEXT NOT NULL,
  ticket INTEGER,
  config_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry_price REAL,
  exit_price REAL,
  sl REAL,
  tp REAL,
  lots REAL,
  pnl_usd REAL,
  exit_reason TEXT
);

CREATE TABLE IF NOT EXISTS heartbeats (
  ts TEXT NOT NULL,
  equity REAL,
  balance REAL,
  open_positions INTEGER,
  daily_pnl REAL
);

CREATE INDEX IF NOT EXISTS idx_closed_by_day ON closed_positions(closed_ts);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def log_signal(self, *, config_name, symbol, timeframe, direction, entry_price,
                   sl, tp, lots, bar_index) -> int:
        cur = self.conn.execute(
            "INSERT INTO signals(ts, config_name, symbol, timeframe, direction, "
            "entry_price, stop_loss, take_profit, lots, bar_index) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_now(), config_name, symbol, timeframe, direction,
             entry_price, sl, tp, lots, bar_index),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_order(self, *, signal_id, config_name, symbol, direction, requested_price,
                  filled_price, requested_lots, filled_lots, sl, tp, ticket, retcode, comment):
        self.conn.execute(
            "INSERT INTO orders(ts, signal_id, ticket, config_name, symbol, direction, "
            "requested_price, filled_price, requested_lots, filled_lots, sl, tp, retcode, comment) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), signal_id, ticket, config_name, symbol, direction,
             requested_price, filled_price, requested_lots, filled_lots, sl, tp, retcode, comment),
        )
        self.conn.commit()

    def log_closed(self, *, opened_ts, ticket, config_name, symbol, direction,
                   entry_price, exit_price, sl, tp, lots, pnl_usd, exit_reason):
        self.conn.execute(
            "INSERT INTO closed_positions(closed_ts, opened_ts, ticket, config_name, symbol, "
            "direction, entry_price, exit_price, sl, tp, lots, pnl_usd, exit_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), opened_ts, ticket, config_name, symbol, direction,
             entry_price, exit_price, sl, tp, lots, pnl_usd, exit_reason),
        )
        self.conn.commit()

    def log_heartbeat(self, *, equity, balance, open_positions, daily_pnl):
        self.conn.execute(
            "INSERT INTO heartbeats(ts, equity, balance, open_positions, daily_pnl) VALUES (?,?,?,?,?)",
            (_now(), equity, balance, open_positions, daily_pnl),
        )
        self.conn.commit()

    def daily_pnl(self, date_iso: str) -> float:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) FROM closed_positions "
            "WHERE substr(closed_ts, 1, 10) = ?",
            (date_iso,),
        )
        return float(cur.fetchone()[0])

    def already_have_signal(self, config_name: str, bar_index: int) -> bool:
        """Idempotency: don't fire the same bar twice if we restart mid-cycle."""
        cur = self.conn.execute(
            "SELECT 1 FROM signals WHERE config_name = ? AND bar_index = ? LIMIT 1",
            (config_name, bar_index),
        )
        return cur.fetchone() is not None
