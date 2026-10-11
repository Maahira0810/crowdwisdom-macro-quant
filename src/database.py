"""SQLite persistence layer: store raw/processed tables and join them with SQL.

Datetimes are stored as ISO strings ('YYYY-MM-DD HH:MM:SS'); tz-aware values are
converted to UTC first. Daily keys (`trade_date`, `date`) are tz-naive midnights
in America/New_York market-day terms so they join cleanly.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.config import DB_PATH

_FMT = "%Y-%m-%d %H:%M:%S"


@contextmanager
def get_conn(db_path: Path | str | None = None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _serialise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            out[col] = out[col].dt.tz_convert("UTC").dt.strftime(_FMT)
        elif pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime(_FMT)
    return out


def write_table(df: pd.DataFrame, name: str, db_path=None, if_exists: str = "replace") -> int:
    with get_conn(db_path) as conn:
        _serialise(df).to_sql(name, conn, if_exists=if_exists, index=False)
    return len(df)


def read_table(name: str, db_path=None, parse_dates: list[str] | None = None,
               utc_cols: list[str] | None = None) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    for c in parse_dates or []:
        df[c] = pd.to_datetime(df[c])
    for c in utc_cols or []:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def query(sql: str, params: tuple = (), db_path=None) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def create_joined_view(db_path=None) -> None:
    """View joining every trade to the macro features of its (New York) trade date."""
    with get_conn(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(daily_macro)") if r[1] != "date"]
        select_macro = ", ".join(f"m.{c}" for c in cols)
        conn.execute("DROP VIEW IF EXISTS v_trades_macro")
        conn.execute(
            f"CREATE VIEW v_trades_macro AS SELECT t.*, {select_macro} "
            "FROM trades_enriched t LEFT JOIN daily_macro m ON t.trade_date = m.date"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades_enriched(trade_date)")


def store_all(tables: dict[str, pd.DataFrame], db_path=None) -> None:
    for name, df in tables.items():
        write_table(df, name, db_path)
    if {"trades_enriched", "daily_macro"} <= set(tables):
        create_joined_view(db_path)
