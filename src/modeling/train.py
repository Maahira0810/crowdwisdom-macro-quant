"""Panel construction + model factory + final fit.

Prediction task
  One row per (trading day, strategy permutation). Target `y` = that permutation's P&L on that
  day (percent of its allocated capital; 0 if it did not trade). Features use information that
  is available *before* the day: lagged permutation performance, scheduled calendar events,
  lagged macro surprises / sentiment, and calendar time features.

Run:  python -m src.modeling.train      (fit on everything, write recommendations for next day)
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features.time_features import add_daily_time_features

log = logging.getLogger(__name__)

NON_FEATURE_COLS = {"date", "permutation_id", "y", "pnl_sum", "n_trades", "wins", "is_future"}


def make_permutation_id(trades: pd.DataFrame, param_cols: list[str]) -> pd.Series:
    return trades[param_cols].astype(str).agg("|".join, axis=1) if param_cols else trades["permutation_id"]


def build_panel(trades: pd.DataFrame, daily_macro: pd.DataFrame, param_cols: list[str],
                horizon: bool = True) -> pd.DataFrame:
    """Build the (date x permutation) panel. If `horizon`, append the next business day with y=NaN
    so the fitted model can rank permutations for the *future* period."""
    t = trades.copy()
    if "permutation_id" not in t.columns:
        t["permutation_id"] = make_permutation_id(t, param_cols)
    t["win"] = (t["pnl"] > 0).astype(int)

    daily = (t.groupby(["trade_date", "permutation_id"])
               .agg(pnl_sum=("pnl", "sum"), n_trades=("pnl", "size"), wins=("win", "sum"))
               .reset_index().rename(columns={"trade_date": "date"}))
    params = t.drop_duplicates("permutation_id").set_index("permutation_id")[param_cols]

    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    last = dates.max()
    future = last + pd.offsets.BDay(1)
    all_dates = dates.append(pd.DatetimeIndex([future])) if horizon else dates
    grid = pd.MultiIndex.from_product([all_dates, params.index], names=["date", "permutation_id"]).to_frame(index=False)
    grid = grid.merge(daily, on=["date", "permutation_id"], how="left")
    grid["is_future"] = (grid["date"] > last).astype(int)
    for c in ("pnl_sum", "n_trades", "wins"):
        grid[c] = grid[c].fillna(0.0)
    grid.loc[grid["is_future"] == 1, ["pnl_sum", "n_trades", "wins"]] = np.nan   # unknown
    grid = grid.sort_values(["permutation_id", "date"]).reset_index(drop=True)

    g = grid.groupby("permutation_id", sort=False)
    grid["ret_lag1"] = g["pnl_sum"].shift(1)
    for w in (5, 20):
        roll = lambda s, w=w: s.shift(1).rolling(w, min_periods=3)
        grid[f"mean_pnl_{w}"] = g["pnl_sum"].transform(lambda s: roll(s).mean())
        grid[f"vol_pnl_{w}"] = g["pnl_sum"].transform(lambda s: roll(s).std())
        wins = g["wins"].transform(lambda s: roll(s).sum())
        n = g["n_trades"].transform(lambda s: roll(s).sum())
        grid[f"win_rate_{w}"] = wins / n.replace(0, np.nan)
    grid["n_trades_5"] = g["n_trades"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    grid["rank_mean_pnl_20"] = grid.groupby("date")["mean_pnl_20"].rank(pct=True)
    grid["mkt_mean_pnl_5"] = grid.groupby("date")["mean_pnl_5"].transform("mean")

    grid = grid.merge(params.reset_index(), on="permutation_id", how="left")
    grid = grid.merge(daily_macro, on="date", how="left")
    grid = add_daily_time_features(grid, "date")
    grid["y"] = grid["pnl_sum"]
    return grid


def get_feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns
            if c not in NON_FEATURE_COLS and not c.startswith("viz_")
            and pd.api.types.is_numeric_dtype(panel[c])]


def make_model(kind: str = "hgb"):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "hgb":
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             random_state=config.RANDOM_STATE)
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(f"unknown model kind: {kind}")


def train_final(panel: pd.DataFrame, feature_cols: list[str], kind: str = "hgb", save_path=None):
    hist = panel[panel["is_future"] == 0]
    model = make_model(kind).fit(hist[feature_cols], hist["y"])
    if save_path:
        joblib.dump({"model": model, "features": feature_cols}, save_path)
    return model


def recommend_next(model, panel: pd.DataFrame, feature_cols: list[str], param_cols: list[str],
                   top_k: int = config.TOP_K) -> pd.DataFrame:
    """Rank permutations for the next (future) trading day by predicted P&L."""
    fut = panel[panel["is_future"] == 1].copy()
    if fut.empty:
        raise ValueError("panel has no future rows; build it with horizon=True")
    fut["pred_pnl_%"] = model.predict(fut[feature_cols])
    cols = ["date", "permutation_id"] + param_cols + ["pred_pnl_%"]
    return fut.sort_values("pred_pnl_%", ascending=False)[cols].head(top_k).reset_index(drop=True)


def main() -> None:
    from src.pipeline import load_inputs, build_features
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    events, news, trades, _ = load_inputs(demo=False)
    feats = build_features(events, news, trades)
    model = train_final(feats["panel"], feats["feature_cols"], save_path=config.PROCESSED_DIR / "model.joblib")
    print(recommend_next(model, feats["panel"], feats["feature_cols"], feats["param_cols"]))


if __name__ == "__main__":
    main()
