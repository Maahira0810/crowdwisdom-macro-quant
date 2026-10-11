"""Time-based features (daily panel level and trade level)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_daily_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Calendar features for a daily panel. All are known in advance (no leakage)."""
    out = df.copy()
    d = out[date_col]
    out["dow"] = d.dt.dayofweek
    out["day_of_month"] = d.dt.day
    out["week_of_month"] = (d.dt.day - 1) // 7 + 1
    out["month"] = d.dt.month
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_month_start"] = (d.dt.day <= 3).astype(int)
    out["is_quarter_end_month"] = d.dt.month.isin([3, 6, 9, 12]).astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 5)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 5)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    return out


def _session(hour_est: pd.Series) -> pd.Series:
    bins = [-1, 3, 9, 16, 24]   # NY clock: overnight / pre-market+open / regular / after-hours
    labels = ["asia_overnight", "london_premarket", "us_regular", "after_hours"]
    return pd.cut(hour_est, bins=bins, labels=labels).astype(str)


def add_trade_time_features(trades: pd.DataFrame, ts_col: str = "entry_time") -> pd.DataFrame:
    """Trade-level time features. Assumes `ts_col` is tz-aware UTC."""
    out = trades.copy()
    ts_est = out[ts_col].dt.tz_convert(config.TZ_EST)
    out["trade_date"] = ts_est.dt.tz_localize(None).dt.normalize()      # NY market day
    out["hour_est"] = ts_est.dt.hour
    out["minute_of_day_est"] = ts_est.dt.hour * 60 + ts_est.dt.minute
    out["entry_dow"] = ts_est.dt.dayofweek
    out["session"] = _session(out["hour_est"])
    if "exit_time" in out.columns:
        out["holding_minutes"] = (out["exit_time"] - out[ts_col]).dt.total_seconds() / 60
    return out


def add_event_proximity(trades: pd.DataFrame, events: pd.DataFrame,
                        ts_col: str = "entry_time") -> pd.DataFrame:
    """Hours to the next / since the previous macro release at each trade's entry."""
    out = trades.copy()
    if events.empty:
        out["hours_to_next_event"] = np.nan
        out["hours_since_last_event"] = np.nan
        out["next_event_type"] = None
        return out

    ev = events[["event_time_utc", "event_type"]].copy()
    ev["event_time_utc"] = ev["event_time_utc"].astype("datetime64[ns, UTC]")
    ev = ev.sort_values("event_time_utc")
    left = out[[ts_col]].copy()
    left["_ts"] = left[ts_col].astype("datetime64[ns, UTC]")
    left["_row"] = np.arange(len(left))
    left = left.sort_values("_ts")

    fwd = pd.merge_asof(left, ev.rename(columns={"event_time_utc": "_next", "event_type": "next_event_type"}),
                        left_on="_ts", right_on="_next", direction="forward")
    bwd = pd.merge_asof(left, ev.rename(columns={"event_time_utc": "_prev", "event_type": "_prev_type"}),
                        left_on="_ts", right_on="_prev", direction="backward")
    fwd = fwd.sort_values("_row").reset_index(drop=True)
    bwd = bwd.sort_values("_row").reset_index(drop=True)

    out = out.reset_index(drop=True)
    out["hours_to_next_event"] = ((fwd["_next"] - fwd["_ts"]).dt.total_seconds() / 3600).clip(upper=24 * 30)
    out["hours_since_last_event"] = ((bwd["_ts"] - bwd["_prev"]).dt.total_seconds() / 3600).clip(upper=24 * 30)
    out["next_event_type"] = fwd["next_event_type"]
    return out
