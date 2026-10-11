"""Normalize the supplied combined Investing.com/Apify CSV without an API call."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.ingestion.apify_calendar import normalize_calendar

RAW = ROOT / "data" / "raw" / "apify_investing_calendar_combined_raw.csv"
OUT = ROOT / "data" / "processed" / "events.csv"


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing {RAW}")
    config.ensure_dirs()
    raw = pd.read_csv(RAW)
    events = normalize_calendar(
        raw.to_dict("records"),
        source_tz=config.APIFY_SOURCE_TZ,
        us_only=True,
        keep_unclassified=False,
    )
    events.to_csv(OUT, index=False)
    print(f"Normalized {len(raw)} raw rows into {len(events)} classified events: {OUT}")
    if not events.empty:
        print(f"UTC coverage: {events['event_time_utc'].min()} -> {events['event_time_utc'].max()}")
        print(events.groupby("event_type").size().to_string())


if __name__ == "__main__":
    main()
