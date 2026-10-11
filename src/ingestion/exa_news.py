"""Second news source via Exa (neural search) - complements Tavily.

Run:  python -m src.ingestion.exa_news --windows 6 --window-days 30 --max-results 10
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
    from exa_py import Exa

    api_key = api_key or config.EXA_API_KEY
    if not api_key:
        raise RuntimeError("Set EXA_API_KEY in .env")
    exa = Exa(api_key=api_key)
    queries = queries or DEFAULT_NEWS_QUERIES

    today = pd.Timestamp.now(tz="UTC").normalize()
    records: list[dict] = []
    for w in range(windows):
        end = today - pd.Timedelta(days=w * window_days)
        start = end - pd.Timedelta(days=window_days)
        for q in queries:
            try:
                res = exa.search_and_contents(
                    q, num_results=max_results, category="news",
                    start_published_date=start.strftime("%Y-%m-%dT00:00:00.000Z"),
                    end_published_date=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z"),
                    text={"max_characters": 1500},
                )
            except Exception as exc:
                log.warning("Exa failed for %r [%s..%s]: %s", q, start.date(), end.date(), exc)
                continue
            for r in res.results:
                records.append({
                    "query": q, "title": getattr(r, "title", None), "url": getattr(r, "url", None),
                    "text": getattr(r, "text", None),
                    "published_date": getattr(r, "published_date", None),
                    "score": getattr(r, "score", None),
                })
            time.sleep(sleep_s)

    save_raw_json(records, "exa_raw")
    return normalize_news(records, source="exa")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch macro news with Exa")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--max-results", type=int, default=10)
    args = ap.parse_args()
    config.ensure_dirs()
    df = fetch_news(windows=args.windows, window_days=args.window_days, max_results=args.max_results)
    out = config.PROCESSED_DIR / "news_exa.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} articles -> {out}")


if __name__ == "__main__":
    main()
