"""Synthetic demo data (calendar, news, trade logs) so the whole pipeline runs without API keys.

The calendar is generated as *raw actor-style records* and pushed through the real
`normalize_calendar`, so the demo also exercises the parsing / timezone code.
Trade P&L has planted regime effects (event days, hawkish tilt) - results on this data only
prove the plumbing works; they say NOTHING about real-world performance.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import config
from src.ingestion.apify_calendar import normalize_calendar
from src.ingestion.common import NEWS_COLUMNS

LOOKBACKS, STOPS, TAKES = [10, 20, 50], [0.5, 1.0, 2.0], [1.0, 2.0, 3.0]


def _bday(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.offsets.BDay(0)


def make_calendar_raw(today: pd.Timestamp, days: int, future_days: int, rng) -> list[dict]:
    start, stop = today - pd.Timedelta(days=days), today + pd.Timedelta(days=future_days)
    items: list[dict] = []

    def add(day, time, name, forecast, actual, previous, fmt):
        if not (start <= day <= stop):
            return
        fut = day > today
        items.append({"date": day.strftime("%Y-%m-%d"), "time": time, "event": name, "country": "US",
                      "importance": "High", "forecast": fmt(forecast), "previous": fmt(previous),
                      "actual": "" if fut else fmt(actual)})

    pct = lambda x: f"{x:.1f}%"
    kjobs = lambda x: f"{x:.0f}K"
    for ms in pd.date_range(start - pd.offsets.MonthBegin(1), stop, freq="MS"):
        cpi_day = _bday(ms + pd.Timedelta(days=11))
        f = rng.choice([0.2, 0.3, 0.4])
        a = f + rng.choice([-0.1, 0.0, 0.1], p=[0.3, 0.3, 0.4])
        add(cpi_day, "08:30", "CPI m/m", f, a, f, pct)
        add(cpi_day, "08:30", "Core CPI m/m", f, a + rng.choice([-0.1, 0.0, 0.1]), f, pct)
        nfp_day = ms + pd.Timedelta(days=int((4 - ms.dayofweek) % 7))
        f = rng.normal(180, 20)
        add(nfp_day, "08:30", "Nonfarm Payrolls", f, f + rng.normal(0, 60), f, kjobs)
        u = rng.choice([3.8, 3.9, 4.0])
        add(nfp_day, "08:30", "Unemployment Rate", u, u + rng.choice([-0.1, 0.0, 0.1]), u, pct)
    rate, d = 5.25, start + pd.Timedelta(days=10)
    while d <= stop:
        day = _bday(d)
        surprise = rng.choice([0.0, 0.25, -0.25], p=[0.8, 0.1, 0.1])
        add(day, "14:00", "FOMC Interest Rate Decision", rate, rate + surprise, rate, lambda x: f"{x:.2f}%")
        d += pd.Timedelta(days=42)
    return items


def _event_signal(events: pd.DataFrame) -> pd.Series:
    """Per NY date: sign of realised surprise summed over events (+ = hawkish/hot)."""
    ev = events.dropna(subset=["surprise"]).copy()
    ev["event_date"] = ev["event_time_est"].dt.tz_localize(None).dt.normalize()
    return np.sign(ev["surprise"]).groupby(ev["event_date"]).sum()


def _latents(idx: pd.DatetimeIndex, sig: pd.Series, rng) -> pd.DataFrame:
    hawk, risk = np.zeros(len(idx)), np.zeros(len(idx))
    s = sig.reindex(idx).fillna(0.0).to_numpy()
    for i in range(1, len(idx)):
        hawk[i] = 0.92 * hawk[i - 1] + 0.30 * s[i - 1] + rng.normal(0, 0.12)
        risk[i] = 0.90 * risk[i - 1] - 0.15 * hawk[i - 1] + rng.normal(0, 0.15)
    return pd.DataFrame({"hawk": hawk, "risk": risk}, index=idx)


H_POS = ["Fed signals higher for longer as inflation stays sticky", "Hot CPI print fuels rate hike bets",
         "Strong jobs data raise tightening fears"]
H_NEG = ["Cooling inflation boosts rate cut hopes", "Softer jobs data points to dovish Fed pivot",
         "Disinflation trend supports easing outlook"]
H_NEU = ["Fed officials keep policy outlook data dependent", "Markets await next inflation reading"]
R_POS = ["Stocks rally on resilient growth and upbeat earnings", "Equities gain as optimism builds"]
R_NEG = ["Selloff deepens amid recession fears and rising volatility",
         "Markets slump as uncertainty weighs on growth"]
R_NEU = ["Markets trade in a narrow range"]


def _pick(rng, v, pos, neg, neu, thr=0.25):
    return rng.choice(pos if v > thr else neg if v < -thr else neu)


def make_news(lat: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for day, r in lat.iterrows():
        for k in range(int(rng.integers(1, 5)) if day.dayofweek < 5 else 1):
            title = f"{_pick(rng, r.hawk, H_POS, H_NEG, H_NEU)}; {_pick(rng, r.risk, R_POS, R_NEG, R_NEU)}"
            ts = day + pd.Timedelta(hours=int(rng.integers(12, 22)), minutes=int(rng.integers(0, 60)))
            rows.append({"source": "demo", "query": "demo", "title": title, "url": f"https://example.com/demo/{len(rows)}",
                         "content": "Demo article body. " + title, "published_at": ts.tz_localize("UTC"),
                         "score": float(rng.uniform(0.3, 1.0))})
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def make_trades(events: pd.DataFrame, lat: pd.DataFrame, today: pd.Timestamp, days: int, rng) -> pd.DataFrame:
    perms = list(itertools.product(LOOKBACKS, STOPS, TAKES))
    base = {p: rng.normal(0.0, 0.05) for p in perms}
    ev_dates = set(pd.to_datetime(events.dropna(subset=["actual"])["event_time_est"]
                                  .dt.tz_localize(None).dt.normalize()))
    rows = []
    for day in pd.bdate_range(today - pd.Timedelta(days=days), today):
        is_event = day in ev_dates
        hawk = lat["hawk"].get(day, 0.0)
        for lb, sl, tp in perms:
            mu = (base[(lb, sl, tp)]
                  + (0.12 if (is_event and sl == 0.5) else 0.0)
                  - 0.08 * hawk * (tp - 2.0)
                  + (0.05 if (lb == 50 and not is_event) else 0.0))
            for _ in range(int(rng.poisson(2.0))):
                entry = day + pd.Timedelta(hours=int(rng.integers(14, 20)), minutes=int(rng.integers(0, 60)))
                rows.append({"permutation_id": f"L{lb}_SL{sl}_TP{tp}", "lookback": lb, "stop_loss": sl,
                             "take_profit": tp, "entry_time": entry.tz_localize("UTC"),
                             "exit_time": (entry + pd.Timedelta(minutes=int(rng.integers(5, 120)))).tz_localize("UTC"),
                             "pnl": float(rng.normal(mu, 0.6))})
    df = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    df.insert(0, "trade_id", np.arange(1, len(df) + 1))
    return df


def make_demo_data(days: int = config.LOOKBACK_DAYS, future_days: int = config.FUTURE_DAYS,
                   seed: int = 7, today: pd.Timestamp | None = None):
    rng = np.random.default_rng(seed)
    today = (today or pd.Timestamp.now(tz="UTC").normalize().tz_localize(None))
    raw = make_calendar_raw(today, days, future_days, rng)
    events = normalize_calendar(raw, source_tz=config.TZ_EST)
    idx = pd.date_range(today - pd.Timedelta(days=days + 30), today + pd.Timedelta(days=future_days), freq="D")
    lat = _latents(idx, _event_signal(events), rng)
    news = make_news(lat.loc[: today], rng)
    trades = make_trades(events, lat, today, days, rng)
    return events, news, trades


def make_proxy_trades(days: int = config.LOOKBACK_DAYS, seed: int = 19,
                      today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Create neutral synthetic trade outcomes for pipeline testing only.

    Unlike make_trades(), this deliberately has NO planted relationship between
    strategy parameters, calendar features, news sentiment, and P&L. Returns are
    independent zero-mean noise. It cannot support claims about real strategy skill.
    """
    rng = np.random.default_rng(seed)
    today = today or pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    dates = pd.bdate_range(today - pd.Timedelta(days=days), today)
    perms = list(itertools.product(LOOKBACKS, STOPS, TAKES))
    rows: list[dict] = []
    trade_id = 1
    for day in dates:
        for lb, sl, tp in perms:
            entry = (day + pd.Timedelta(hours=14)).tz_localize("UTC")
            exit_time = entry + pd.Timedelta(minutes=30)
            rows.append({
                "trade_id": trade_id,
                "permutation_id": f"L{lb}_SL{sl}_TP{tp}",
                "lookback": lb, "stop_loss": sl, "take_profit": tp,
                "entry_time": entry, "exit_time": exit_time,
                # Synthetic P&L percentage; zero-mean noise, no planted signal.
                "pnl": float(rng.normal(loc=0.0, scale=0.6)),
                "data_provenance": "SYNTHETIC_PROXY_NOT_REAL_TRADES",
            })
            trade_id += 1
    return pd.DataFrame(rows)
