"""SQLite ledger for paper trading — every signal, order, close, and heartbeat.

Tables:
- signals:          strategy decisions (before order placement)
- orders:           MT5 order results (fill or rejection)
- closed_positions: reconciled trade outcomes (P&L, exit reason)
- heartbeats:       account state samples (~1/min)
- rejections:       every signal a gate/filter skipped, and why (CLAUDE.md
                     spec §13: "The rejections table is how filter bugs get
                     found" -- previously these only went to text logs)

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
  bar_index INTEGER,
  bar_ts TEXT
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

CREATE TABLE IF NOT EXISTS trade_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  config_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  direction TEXT,
  exit_reason TEXT,
  pnl_usd REAL,
  r_multiple REAL,
  entry_hour INTEGER,
  session TEXT,
  hold_minutes REAL,
  was_win INTEGER,
  why TEXT,
  mfe_r REAL,
  mae_r REAL
);

CREATE TABLE IF NOT EXISTS rejections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  config_name TEXT NOT NULL,
  symbol TEXT,
  gate TEXT NOT NULL,
  reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_closed_by_day ON closed_positions(closed_ts);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_rejections_gate ON rejections(gate);
CREATE INDEX IF NOT EXISTS idx_reviews_config ON trade_reviews(config_name);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        # WAL + busy_timeout: three processes (bot, dashboard, chart_api)
        # share these files; default rollback journal gives intermittent
        # "database is locked" under concurrent read/write.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """In-place migrations for DBs created by older schema versions.

        bar_ts (signals): the old idempotency key was (config_name,
        bar_index), but bar_index is a ROW POSITION in the rolling
        300-bar window — the newest closed bar is always 299, so after
        one signal a slot could never fire again (AUDIT C1). New key is
        (config_name, bar_ts). Old rows keep bar_ts NULL, which never
        matches a real timestamp, so previously-dead slots revive on the
        first new bar after upgrading."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(signals)")}
        if "bar_ts" not in cols:
            self.conn.execute("ALTER TABLE signals ADD COLUMN bar_ts TEXT")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def log_signal(self, *, config_name, symbol, timeframe, direction, entry_price,
                   sl, tp, lots, bar_index, bar_ts=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO signals(ts, config_name, symbol, timeframe, direction, "
            "entry_price, stop_loss, take_profit, lots, bar_index, bar_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), config_name, symbol, timeframe, direction,
             entry_price, sl, tp, lots, bar_index, bar_ts),
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
                   entry_price, exit_price, sl, tp, lots, pnl_usd, exit_reason, closed_ts=None):
        # closed_ts defaults to now (the normal live-reconciliation case: the
        # close was just observed). A caller backfilling a historical close
        # (e.g. a position whose owning strategy/slot no longer exists to
        # reconcile it live) can pass the real MT5 deal time instead, so
        # equity curves built from this column place it at the right point
        # in time rather than at the moment of the backfill.
        self.conn.execute(
            "INSERT INTO closed_positions(closed_ts, opened_ts, ticket, config_name, symbol, "
            "direction, entry_price, exit_price, sl, tp, lots, pnl_usd, exit_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (closed_ts or _now(), opened_ts, ticket, config_name, symbol, direction,
             entry_price, exit_price, sl, tp, lots, pnl_usd, exit_reason),
        )
        self.conn.commit()

    def log_heartbeat(self, *, equity, balance, open_positions, daily_pnl):
        self.conn.execute(
            "INSERT INTO heartbeats(ts, equity, balance, open_positions, daily_pnl) VALUES (?,?,?,?,?)",
            (_now(), equity, balance, open_positions, daily_pnl),
        )
        self.conn.commit()

    def log_rejection(self, *, config_name: str, symbol: str | None, gate: str, reason: str) -> None:
        """A signal a gate/filter skipped -- structured, so filter bugs (a
        gate always firing, a threshold set wrong) show up as a queryable
        pattern instead of needing a text-log grep (CLAUDE.md spec §13)."""
        self.conn.execute(
            "INSERT INTO rejections(ts, config_name, symbol, gate, reason) VALUES (?,?,?,?,?)",
            (_now(), config_name, symbol, gate, reason),
        )
        self.conn.commit()

    def total_realized(self) -> float:
        """All-time realized P&L for this book — the base number for the
        FTMO guard's book-scope math and percent-equity sizing."""
        cur = self.conn.execute("SELECT COALESCE(SUM(pnl_usd), 0) FROM closed_positions")
        return float(cur.fetchone()[0])

    def daily_pnl(self, date_iso: str) -> float:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) FROM closed_positions "
            "WHERE substr(closed_ts, 1, 10) = ?",
            (date_iso,),
        )
        return float(cur.fetchone()[0])

    def distinct_trading_days(self) -> int:
        """Number of distinct UTC dates on which an order was SENT.

        FTMO's "minimum 4 trading days" counts days you actually traded, so
        this keys on `orders` (an order was placed) rather than
        `closed_positions` (a trade finished) — a position opened on day 1 and
        closed on day 3 is one trading day, not two, and a still-open trade
        must count immediately.
        """
        cur = self.conn.execute(
            "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM orders"
        )
        return int(cur.fetchone()[0] or 0)

    def already_have_signal(self, config_name: str, bar_ts: str) -> bool:
        """Idempotency: don't fire on the same BAR twice.

        Keyed on the bar's actual timestamp. The previous key (bar_index)
        was the bar's position in the rolling window — always 299 for the
        newest bar — which permanently silenced a slot after its first
        signal (AUDIT C1, confirmed live)."""
        cur = self.conn.execute(
            "SELECT 1 FROM signals WHERE config_name = ? AND bar_ts = ? LIMIT 1",
            (config_name, bar_ts),
        )
        return cur.fetchone() is not None

    def log_review(self, review) -> None:
        """Persist a TradeReview (the per-trade post-mortem / memory)."""
        self.conn.execute(
            "INSERT INTO trade_reviews(ts, config_name, symbol, direction, exit_reason, "
            "pnl_usd, r_multiple, entry_hour, session, hold_minutes, was_win, why, mfe_r, mae_r) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), review.config_name, review.symbol, review.direction, review.exit_reason,
             review.pnl_usd, review.r_multiple, review.entry_hour, review.session,
             review.hold_minutes, 1 if review.was_win else 0, review.why,
             review.mfe_r, review.mae_r),
        )
        self.conn.commit()

    def all_reviews_for_learner(self) -> list[tuple]:
        """(config_name, session, r_multiple) rows to rebuild the SessionLearner."""
        cur = self.conn.execute(
            "SELECT config_name, session, r_multiple FROM trade_reviews "
            "WHERE session IS NOT NULL AND r_multiple IS NOT NULL"
        )
        return cur.fetchall()

    def recent_reviews(self, limit: int = 30) -> list[dict]:
        """Most-recent-first reviews for the dashboard 'why it happened' feed."""
        cur = self.conn.execute(
            "SELECT ts, config_name, symbol, direction, exit_reason, pnl_usd, "
            "r_multiple, session, was_win, why FROM trade_reviews ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def equity_curves_by_config(self):
        """(config_name, closed_ts, pnl_usd) for the per-strategy improving
        equity-curve chart — one climbing line per strategy."""
        cur = self.conn.execute(
            "SELECT config_name, closed_ts, pnl_usd FROM closed_positions "
            "WHERE pnl_usd IS NOT NULL ORDER BY closed_ts"
        )
        return cur.fetchall()

    def recent_closed_pnls(self, config_name: str, limit: int) -> list[float]:
        """Most-recent-first realized P&L for one slot — feeds the slot-health
        monitor so quarantine state survives a restart (rebuilt from the DB,
        never trusted to exist in memory)."""
        cur = self.conn.execute(
            "SELECT pnl_usd FROM closed_positions WHERE config_name = ? "
            "AND pnl_usd IS NOT NULL ORDER BY id DESC LIMIT ?",
            (config_name, limit),
        )
        return [float(r[0]) for r in cur.fetchall()]
