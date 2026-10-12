"""Helpers shared by the ingestion modules."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)

NEWS_COLUMNS = ["source", "query", "title", "url", "content", "published_at", "score"]

DEFAULT_NEWS_QUERIES = [
    "US CPI inflation report",
    "Federal Reserve FOMC interest rate outlook",
    "US jobs report nonfarm payrolls unemployment",
    "US economy outlook recession risk",
    "Treasury yields dollar stocks market reaction",
]


def save_raw_json(obj, prefix: str, raw_dir: Path | None = None) -> Path:
    raw_dir = Path(raw_dir or RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def normalize_news(records: list[dict], source: str) -> pd.DataFrame:
    """Map provider-specific result dicts onto one common news schema."""
    rows = []
    for r in records:
        rows.append({
            "source": source,
            "query": r.get("query"),
            "title": (r.get("title") or "").strip(),
            "url": r.get("url"),
            "content": (r.get("content") or r.get("text") or "").strip(),
            "published_at": r.get("published_date") or r.get("published_at"),
            "score": r.get("score"),
        })
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    before = len(df)
    df = df.dropna(subset=["published_at", "url"]).drop_duplicates(subset="url")
    if len(df) < before:
        log.info("%s: dropped %d rows (no date/url or duplicate)", source, before - len(df))
    return df.sort_values("published_at").reset_index(drop=True)
