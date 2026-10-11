"""Macro features: event surprises, news sentiment, calendar proximity.

LEAKAGE POLICY
  * Anything *realised* (actual values, news sentiment) enters the model with a 1-day lag.
  * Anything *scheduled* (is today a CPI day, days until next release) is known in advance
    and enters unlagged.
  * Columns prefixed `viz_` are unlagged, for charts only, and are never model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

MAX_DAYS = 30

# Deliberately transparent keyword lexicon. Swap `score_text` for FinBERT / an LLM
# classifier if you want a stronger signal - the rest of the pipeline is unchanged.
HAWKISH = ["hawkish", "rate hike", "hike", "hot inflation", "hotter", "sticky", "tightening",
           "higher for longer", "inflation surge", "accelerat", "strong jobs", "beats expectations",
           "overheating", "tighter", "inflation stays", "inflation remains elevated"]
DOVISH = ["dovish", "rate cut", "cut rates", "easing", "cooling", "softer", "disinflation",
          "slowdown", "weak jobs", "misses expectations", "pause", "lower inflation",
          "layoffs", "pivot", "unemployment rises"]
POSITIVE = ["rally", "surge", "gain", "rise", "record high", "optimism", "growth", "strong",
            "beat", "rebound", "upbeat", "resilient"]
NEGATIVE = ["selloff", "sell-off", "plunge", "fall", "drop", "slump", "fear", "recession",
            "weak", "miss", "downturn", "crash", "volatility", "uncertainty", "losses"]


def _count(text: str, words: list[str]) -> int:
    return sum(text.count(w) for w in words)


def score_text(text: str) -> tuple[float, float]:
    """Return (hawkish_dovish, risk_sentiment), each in [-1, 1]. + = hawkish / risk-on."""
    t = (text or "").lower()
    h, d = _count(t, HAWKISH), _count(t, DOVISH)
    p, n = _count(t, POSITIVE), _count(t, NEGATIVE)
    hd = (h - d) / (h + d) if (h + d) else 0.0
    rs = (p - n) / (p + n) if (p + n) else 0.0
    return hd, rs


def score_news(news: pd.DataFrame) -> pd.DataFrame:
    out = news.copy()
    if out.empty:
        out["hawk_dove"], out["risk_sent"] = [], []
        out["date"] = pd.to_datetime([])
        return out
    text = (out["title"].fillna("") + ". " + out["content"].fillna(""))
    scores = text.map(score_text)
    out["hawk_dove"] = [s[0] for s in scores]
    out["risk_sent"] = [s[1] for s in scores]
    out["date"] = out["published_at"].dt.tz_convert(config.TZ_EST).dt.tz_localize(None).dt.normalize()
    return out


# ------------------------------------------------------------------ events
def add_surprise_z(events: pd.DataFrame) -> pd.DataFrame:
    """Standardise surprises per event type using only *past* releases (expanding std)."""
    ev = events.sort_values("event_time_utc").copy()
    ev["event_date"] = ev["event_time_est"].dt.tz_localize(None).dt.normalize()

    def _z(s: pd.Series) -> pd.Series:
        sd = s.expanding(min_periods=3).std().shift(1)
        return (s / sd).replace([np.inf, -np.inf], np.nan)

    ev["surprise_z"] = ev.groupby("event_type")["surprise"].transform(_z).clip(-3, 3)
    return ev


def build_daily_macro(events: pd.DataFrame, news: pd.DataFrame,
                      start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One row per calendar day with leakage-safe macro features (see module docstring)."""
    idx = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D", name="date")
    d = pd.DataFrame(index=idx)

    # --- scheduled (calendar-known) ------------------------------------------------
    ev = add_surprise_z(events) if len(events) else events.assign(event_date=pd.NaT, surprise_z=np.nan)
    for t in config.EVENT_TYPES:
        cnt = ev[ev["event_type"] == t].groupby("event_date").size() if len(ev) else pd.Series(dtype=float)
        d[f"is_{t.lower()}_day"] = (cnt.reindex(idx).fillna(0) > 0).astype(int)
    d["n_events"] = ev.groupby("event_date").size().reindex(idx).fillna(0).astype(int) if len(ev) else 0
    d["is_event_day"] = (d["n_events"] > 0).astype(int)

    ev_dates = pd.DatetimeIndex(sorted(ev["event_date"].dropna().unique())) if len(ev) else pd.DatetimeIndex([])
    to_next, since_last = [], []
    for day in idx:
        fut, past = ev_dates[ev_dates >= day], ev_dates[ev_dates <= day]
        to_next.append(min((fut[0] - day).days, MAX_DAYS) if len(fut) else MAX_DAYS)
        since_last.append(min((day - past[-1]).days, MAX_DAYS) if len(past) else MAX_DAYS)
    d["days_to_next_event"], d["days_since_last_event"] = to_next, since_last

    # --- realised surprises (lag 1) ----------------------------------------------
    z_day = (ev.groupby("event_date")["surprise_z"].sum(min_count=1).reindex(idx)
             if len(ev) else pd.Series(np.nan, index=idx))
    z0 = z_day.fillna(0.0)
    d["hawkish_surprise_lag1"] = z0.shift(1)
    d["surprise_ewm"] = z0.ewm(halflife=5).mean().shift(1)
    for t in ("CPI", "EMPLOYMENT"):
        last = (ev[ev["event_type"] == t].groupby("event_date")["surprise_z"].last().reindex(idx).ffill()
                if len(ev) else pd.Series(np.nan, index=idx))
        d[f"{t.lower()}_surprise_last"] = last.shift(1).fillna(0.0)

    # --- news sentiment (lag 1) --------------------------------------------------
    sc = score_news(news)
    if len(sc):
        daily = sc.groupby("date").agg(hawk_dove=("hawk_dove", "mean"), risk_sent=("risk_sent", "mean"),
                                       n_articles=("title", "size")).reindex(idx)
    else:
        daily = pd.DataFrame({"hawk_dove": np.nan, "risk_sent": np.nan, "n_articles": 0}, index=idx)
    daily["n_articles"] = daily["n_articles"].fillna(0)
    hd_ewm = daily["hawk_dove"].ewm(halflife=3, min_periods=1).mean()
    rs_ewm = daily["risk_sent"].ewm(halflife=3, min_periods=1).mean()
    d["hawk_dove_ewm"] = hd_ewm.shift(1)
    d["risk_sent_ewm"] = rs_ewm.shift(1)
    d["sent_change_5d"] = (rs_ewm - rs_ewm.shift(5)).shift(1)
    d["news_volume_7d"] = daily["n_articles"].rolling(7, min_periods=1).sum().shift(1)

    # --- unlagged, chart-only ----------------------------------------------------
    d["viz_hawk_dove"] = hd_ewm
    d["viz_risk_sentiment"] = rs_ewm
    d["viz_surprise_ewm"] = z0.ewm(halflife=5).mean()
    d["viz_surprise_cum"] = z0.cumsum()

    return d.reset_index()
