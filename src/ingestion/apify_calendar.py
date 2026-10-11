"""Macro-economic calendar ingestion via Apify.

Scrapes CPI / FOMC / employment (+ PCE, GDP) releases for the last N days (and a few
upcoming days), then normalises them to a fixed schema with UTC *and* New-York (EST/EDT)
timestamps, actual / forecast / previous values and the surprise (actual - forecast).

Actor schemas differ. `normalize_calendar` therefore accepts several common field names
(see FIELD_CANDIDATES). If your actor uses other names, add them there - that is the only
place you should need to touch. Set APIFY_CALENDAR_ACTOR_ID in .env (the actor you use) and,
if needed, APIFY_CALENDAR_INPUT_JSON with that actor's input schema.

Run:  python -m src.ingestion.apify_calendar --days 180
"""
from __future__ import annotations

import argparse
import json
import logging
import re

import numpy as np
import pandas as pd

from src import config
from src.ingestion.common import save_raw_json

log = logging.getLogger(__name__)

FIELD_CANDIDATES: dict[str, list[str]] = {
    "event": ["event", "eventName", "event_name", "name", "title", "indicator"],
    "actual": ["actual", "actualValue", "actual_value"],
    "forecast": ["forecast", "forecastValue", "consensus", "estimate", "expected"],
    "previous": ["previous", "previousValue", "prev", "prior"],
    "timestamp": ["datetime", "dateTime", "timestamp", "date_time", "time_utc", "releaseTime"],
    "date": ["date", "day", "releaseDate"],
    "time": ["time", "releaseTime"],
    "country": ["country", "countryCode", "currency", "region"],
    "importance": ["importance", "impact", "volatility", "stars"],
}

US_ALIASES = {"us", "usa", "united states", "usd", "united states of america"}

CALENDAR_COLUMNS = [
    "event_time_utc", "event_time_est", "event_name", "event_type", "country", "importance",
    "actual", "forecast", "previous", "surprise", "surprise_pct",
]


# ---------------------------------------------------------------- parsing helpers
def parse_number(x) -> float:
    """'3.2%' -> 3.2, '225K' -> 225000, '-1.2M' -> -1.2e6, '' / '-' -> NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "").lstrip("$€£<>~≈ ").strip()
    if s in {"", "-", "—", "N/A", "n/a", "nan", "None"}:
        return np.nan
    m = re.fullmatch(r"(-?\d*\.?\d+)\s*([%KMBT]?)", s, flags=re.I)
    if not m:
        return np.nan
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2).upper(), 1.0)
    return float(m.group(1)) * mult


def classify_event(name: str) -> str | None:
    n = str(name).lower()
    for etype, kws in config.EVENT_KEYWORDS.items():
        if any(k in n for k in kws):
            return etype
    return None


def _first_present(df: pd.DataFrame, keys: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def _col(df: pd.DataFrame, key: str) -> pd.Series:
    c = _first_present(df, FIELD_CANDIDATES[key])
    return df[c] if c else pd.Series(np.nan, index=df.index, dtype="object")


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    """Parse mixed ISO (YYYY-MM-DD) and Investing.com day-first date strings safely.

    pandas' `format="mixed", dayfirst=True` can misread ISO dates such as
    2026-09-11 as 2026-11-09. Detect ISO-formatted values and parse those
    month-first/ISO, while retaining day-first parsing for strings like 11/09/2026.
    """
    def parse_one(value):
        if pd.isna(value):
            return pd.NaT
        value = str(value).strip()
        if not value or value.lower() in {"nan", "nat", "none"}:
            return pd.NaT
        is_iso = bool(re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", value))
        return pd.to_datetime(value, errors="coerce", format="mixed", dayfirst=not is_iso)

    return pd.Series((parse_one(v) for v in values), index=values.index)


def _to_utc(df: pd.DataFrame, source_tz: str) -> pd.Series:
    """Build a UTC timestamp from either a datetime column or date + time columns."""
    ts_col = _first_present(df, FIELD_CANDIDATES["timestamp"])
    if ts_col is not None and df[ts_col].notna().any():
        s = df[ts_col].astype(str)
        if s.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").any():      # carries its own offset
            return pd.to_datetime(s, errors="coerce", utc=True)
        parsed = _parse_datetime_series(s)
    else:
        date, time = _col(df, "date").astype(str), _col(df, "time").astype(str)
        combined = _parse_datetime_series(date + " " + time)
        parsed = combined.fillna(_parse_datetime_series(date))  # all-day events / missing time
    return (parsed.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC"))


# ---------------------------------------------------------------- normalisation
def normalize_calendar(items: list[dict], source_tz: str = "UTC",
                       us_only: bool = True, keep_unclassified: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    out = pd.DataFrame(index=df.index)
    out["event_time_utc"] = _to_utc(df, source_tz)
    out["event_name"] = _col(df, "event").astype(str).str.strip()
    out["country"] = _col(df, "country")
    out["importance"] = _col(df, "importance")
    out["event_type"] = out["event_name"].map(classify_event)
    for k in ("actual", "forecast", "previous"):
        out[k] = _col(df, k).map(parse_number)

    out = out.dropna(subset=["event_time_utc"])
    if us_only and out["country"].notna().any():
        out = out[out["country"].astype(str).str.lower().str.strip().isin(US_ALIASES)]
    if not keep_unclassified:
        out = out.dropna(subset=["event_type"])

    out["event_time_est"] = out["event_time_utc"].dt.tz_convert(config.TZ_EST)
    out["surprise"] = out["actual"] - out["forecast"]
    denom = out["forecast"].abs().replace(0, np.nan)
    out["surprise_pct"] = out["surprise"] / denom * 100
    out = (out.drop_duplicates(subset=["event_time_utc", "event_name"])
              .sort_values("event_time_utc").reset_index(drop=True))
    return out[CALENDAR_COLUMNS]


# ---------------------------------------------------------------- Apify call
def _build_run_input(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    if config.APIFY_CALENDAR_INPUT_JSON:
        raw = config.APIFY_CALENDAR_INPUT_JSON.replace("{start_date}", s).replace("{end_date}", e)
        return json.loads(raw)
    # Generic default - adjust to your actor's input schema via APIFY_CALENDAR_INPUT_JSON.
    return {"startDate": s, "endDate": e, "countries": ["US"]}


def fetch_calendar(days: int = config.LOOKBACK_DAYS, future_days: int = config.FUTURE_DAYS,
                   actor_id: str | None = None, run_input: dict | None = None,
                   token: str | None = None, source_tz: str | None = None) -> pd.DataFrame:
    from apify_client import ApifyClient  # imported lazily so tests don't need the package

    token = token or config.APIFY_TOKEN
    actor_id = actor_id or config.APIFY_CALENDAR_ACTOR_ID
    if not token or not actor_id:
        raise RuntimeError("Set APIFY_TOKEN and APIFY_CALENDAR_ACTOR_ID in .env")

    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start, end = today - pd.Timedelta(days=days), today + pd.Timedelta(days=future_days)
    run_input = run_input or _build_run_input(start, end)

    client = ApifyClient(token)
    log.info("Running Apify actor %s with input %s", actor_id, run_input)
    run = client.actor(actor_id).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    log.info("Actor returned %d raw items", len(items))
    path = save_raw_json({"actor_id": actor_id, "run_input": run_input, "items": items},
                         "apify_calendar_raw")
    log.info("Raw payload saved to %s", path)

    return normalize_calendar(items, source_tz=source_tz or config.APIFY_SOURCE_TZ)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Scrape macro calendar with Apify")
    ap.add_argument("--days", type=int, default=config.LOOKBACK_DAYS)
    ap.add_argument("--future-days", type=int, default=config.FUTURE_DAYS)
    args = ap.parse_args()
    config.ensure_dirs()
    df = fetch_calendar(args.days, args.future_days)
    out = config.PROCESSED_DIR / "events.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} events -> {out}")
    print(df.groupby("event_type").size())


if __name__ == "__main__":
    main()
