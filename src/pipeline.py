"""Orchestration helpers used by main.py and the notebook."""
from __future__ import annotations

import logging

import pandas as pd

from src import config, database
from src.features.macro_features import build_daily_macro
from src.features.time_features import add_event_proximity, add_trade_time_features
from src.modeling.metrics import regression_metrics, summarize_returns
from src.modeling.train import build_panel, get_feature_cols, make_model, recommend_next, train_final
from src.modeling.walk_forward import run_walk_forward

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------- loading
def _load_events(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
    df["event_time_est"] = df["event_time_utc"].dt.tz_convert(config.TZ_EST)
    return df


def _load_news(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["title"])
    df["published_at"] = pd.to_datetime(
    df["published_at"],
    format="mixed",
    utc=True,
    errors="coerce",
    )
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def _load_trades(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("entry_time", "exit_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def load_inputs(demo: bool = False, refresh: bool = False, seed: int = 7):
    """Return (events, news, trades, data_mode). Live mode reads CSVs written by the ingestion
    scripts (or re-scrapes with refresh=True) plus data/raw/trade_logs.csv."""
    config.ensure_dirs()
    if demo:
        from src.synthetic import make_demo_data
        events, news, trades = make_demo_data(seed=seed)
        for name, df in (("events", events), ("news", news), ("trade_logs", trades)):
            df.to_csv(config.DEMO_DIR / f"{name}.csv", index=False)
        return events, news, trades, "demo"

    ev_path = config.PROCESSED_DIR / "events.csv"
    if refresh or not ev_path.exists():
        from src.ingestion.apify_calendar import fetch_calendar
        fetch_calendar().to_csv(ev_path, index=False)
    news_paths = [config.PROCESSED_DIR / n for n in ("news_tavily.csv", "news_exa.csv")]
    if refresh or not any(p.exists() for p in news_paths):
        from src.ingestion import exa_news, tavily_news
        for mod, p in zip((tavily_news, exa_news), news_paths):
            try:
                mod.fetch_news().to_csv(p, index=False)
            except Exception as exc:
                log.warning("%s skipped: %s", mod.__name__, exc)
    news_paths = [p for p in news_paths if p.exists()]
    if not config.TRADE_LOGS_PATH.exists():
        raise FileNotFoundError(
            f"Trade log not found at {config.TRADE_LOGS_PATH}. Required columns: trade_id, entry_time, "
            "exit_time, pnl, plus strategy parameter columns (default lookback, stop_loss, take_profit). "
            "Or run with --demo.")
    return _load_events(ev_path), _load_news(news_paths), _load_trades(config.TRADE_LOGS_PATH), "live"


def load_proxy_inputs(seed: int = 19):
    """Load real collected calendar/news files and pair them with synthetic P&L.

    This mode is for pipeline validation only; it must not be presented as a
    historical trading backtest or evidence of profitability.
    """
    config.ensure_dirs()
    ev_path = config.PROCESSED_DIR / "events.csv"
    if not ev_path.exists():
        raise FileNotFoundError(f"Calendar file missing: {ev_path}. Run scripts/normalize_calendar_export.py first.")
    news_paths = [config.PROCESSED_DIR / n for n in ("news_tavily.csv", "news_exa.csv") if (config.PROCESSED_DIR / n).exists()]
    if not news_paths:
        raise FileNotFoundError("No news CSVs found. Run the Tavily and/or Exa ingestion modules first.")
    events = _load_events(ev_path)
    news = _load_news(news_paths)
    from src.synthetic import make_proxy_trades
    trades = make_proxy_trades(days=config.LOOKBACK_DAYS, seed=seed)
    proxy_dir = config.PROCESSED_DIR / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(proxy_dir / "trade_logs_proxy_synthetic.csv", index=False)
    return events, news, trades, "proxy"


# ----------------------------------------------------------------------- features
def infer_param_cols(trades: pd.DataFrame) -> list[str]:
    cols = [c for c in config.DEFAULT_PARAM_COLS if c in trades.columns]
    if cols:
        return cols
    skip = {"trade_id", "permutation_id", "entry_time", "exit_time", "pnl", "trade_date"}
    return [c for c in trades.columns if c not in skip and pd.api.types.is_numeric_dtype(trades[c])]


def build_features(events: pd.DataFrame, news: pd.DataFrame, trades: pd.DataFrame) -> dict:
    param_cols = infer_param_cols(trades)
    enriched = add_event_proximity(add_trade_time_features(trades), events)
    start = min(enriched["trade_date"].min(), events["event_time_est"].dt.tz_localize(None).min().normalize())
    end = enriched["trade_date"].max() + pd.Timedelta(days=config.FUTURE_DAYS)
    daily = build_daily_macro(events, news, start, end)
    panel = build_panel(enriched, daily, param_cols)
    return {"trades_enriched": enriched, "daily_macro": daily, "panel": panel,
            "feature_cols": get_feature_cols(panel), "param_cols": param_cols}


def store(events, news, feats, db_path) -> None:
    database.store_all({"events": events, "news": news, "trades_enriched": feats["trades_enriched"],
                        "daily_macro": feats["daily_macro"]}, db_path)


# ----------------------------------------------------------------------- modelling
def run_model(feats: dict, model_kind: str = "hgb", top_k: int = config.TOP_K,
              min_train_days: int = config.MIN_TRAIN_DAYS, test_days: int = config.TEST_DAYS):
    wf = run_walk_forward(feats["panel"], feats["feature_cols"], lambda: make_model(model_kind),
                          top_k=top_k, min_train_days=min_train_days, test_days=test_days)
    summary = summarize_returns(wf.returns)
    reg = regression_metrics(wf.predictions)
    final = train_final(feats["panel"], feats["feature_cols"], model_kind)
    recs = recommend_next(final, feats["panel"], feats["feature_cols"], feats["param_cols"], top_k)
    return wf, summary, reg, final, recs
