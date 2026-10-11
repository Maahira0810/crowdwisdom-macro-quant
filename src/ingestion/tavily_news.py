"""Recent / relevant macro news via Tavily.

History is collected in consecutive time windows (default 6 x 30 days) so the sentiment
series covers the same ~180 days as the calendar. Requires tavily-python >= 0.5.1
(start_date / end_date support).

Run:  python -m src.ingestion.tavily_news --windows 6 --window-days 30 --max-results 10
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from src import config
from src.ingestion.common import DEFAULT_NEWS_QUERIES, normalize_news, save_raw_json

log = logging.getLogger(__name__)


def fetch_news(queries: list[str] | None = None, windows: int = 6, window_days: int = 30,
               max_results: int = 10, api_key: str | None = None,
               sleep_s: float = 0.2) -> pd.DataFrame:
    from tavily import TavilyClient

    api_key = api_key or config.TAVILY_API_KEY
    if not api_key:
        raise RuntimeError("Set TAVILY_API_KEY in .env")
    client = TavilyClient(api_key=api_key)
    queries = queries or DEFAULT_NEWS_QUERIES

    today = pd.Timestamp.now(tz="UTC").normalize()
    raw_records: list[dict] = []
    for w in range(windows):
        end = today - pd.Timedelta(days=w * window_days)
        start = end - pd.Timedelta(days=window_days)
        for q in queries:
            try:
                resp = client.search(
                    query=q, topic="news", search_depth="basic", max_results=max_results,
                    start_date=start.strftime("%Y-%m-%d"),
                    end_date=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                )
            except Exception as exc:  # network / quota errors shouldn't kill the whole run
                log.warning("Tavily failed for %r [%s..%s]: %s", q, start.date(), end.date(), exc)
                continue
            for r in resp.get("results", []):
                raw_records.append({**r, "query": q})
            time.sleep(sleep_s)

    save_raw_json(raw_records, "tavily_raw")
    return normalize_news(raw_records, source="tavily")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch macro news with Tavily")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--max-results", type=int, default=10)
    args = ap.parse_args()
    config.ensure_dirs()
    df = fetch_news(windows=args.windows, window_days=args.window_days, max_results=args.max_results)
    out = config.PROCESSED_DIR / "news_tavily.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} articles -> {out}")


if __name__ == "__main__":
    main()
