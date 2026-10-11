import numpy as np
import pandas as pd

from src.features.macro_features import score_text
from src.ingestion.apify_calendar import classify_event, normalize_calendar, parse_number


def test_parse_number():
    assert parse_number("3.2%") == 3.2
    assert parse_number("225K") == 225_000
    assert parse_number("-1.2M") == -1.2e6
    assert parse_number("$1,200") == 1200
    assert np.isnan(parse_number("")) and np.isnan(parse_number("-")) and np.isnan(parse_number(None))


def test_classify_event():
    assert classify_event("Core CPI m/m") == "CPI"
    assert classify_event("FOMC Interest Rate Decision") == "FOMC"
    assert classify_event("Nonfarm Payrolls") == "EMPLOYMENT"
    assert classify_event("Retail Sales") is None


def test_normalize_calendar_timezone_edt_and_est():
    items = [
        {"date": "2026-09-11", "time": "08:30", "event": "CPI m/m", "country": "US",
         "actual": "0.4%", "forecast": "0.3%", "previous": "0.2%"},   # EDT: UTC-4
        {"date": "2026-01-14", "time": "08:30", "event": "CPI m/m", "country": "US",
         "actual": "0.2%", "forecast": "0.3%", "previous": "0.2%"},   # EST: UTC-5
        {"date": "2026-09-11", "time": "08:30", "event": "UK CPI", "country": "UK",
         "actual": "0.4%", "forecast": "0.3%", "previous": "0.2%"},   # filtered: not US
    ]
    df = normalize_calendar(items, source_tz="America/New_York")
    assert len(df) == 2
    assert df.loc[1, "event_time_utc"] == pd.Timestamp("2026-09-11 12:30", tz="UTC")
    assert df.loc[0, "event_time_utc"] == pd.Timestamp("2026-01-14 13:30", tz="UTC")
    assert abs(df.loc[1, "surprise"] - 0.1) < 1e-9
    assert df.loc[1, "event_time_est"].hour == 8


def test_normalize_calendar_offset_timestamps_and_missing_actual():
    items = [{"datetime": "2026-10-12T12:30:00Z", "name": "CPI", "country": "United States",
              "actual": "", "forecast": "0.3%"}]
    df = normalize_calendar(items)
    assert df.loc[0, "event_time_utc"] == pd.Timestamp("2026-10-12 12:30", tz="UTC")
    assert np.isnan(df.loc[0, "actual"]) and np.isnan(df.loc[0, "surprise"])


def test_score_text_signs():
    hd, rs = score_text("Hot CPI fuels rate hike bets; stocks slump on recession fears")
    assert hd > 0 and rs < 0
    hd, rs = score_text("Cooling inflation boosts rate cut hopes as equities rally")
    assert hd < 0 and rs > 0


def test_normalize_calendar_investing_com_day_first_dates():
    items = [{
        "date": "07/09/2026", "time": "18:00", "event": "CPI m/m",
        "currency": "USD", "importance": "high", "actual": "3.2%",
        "forecast": "3.0%", "previous": "3.1%",
    }]
    df = normalize_calendar(items, source_tz="UTC")
    assert len(df) == 1
    assert df.loc[0, "event_time_utc"] == pd.Timestamp("2026-09-07 18:00", tz="UTC")
    assert abs(df.loc[0, "surprise"] - 0.2) < 1e-9
