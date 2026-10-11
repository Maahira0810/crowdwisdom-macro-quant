"""Performance and forecast-quality metrics.

Convention: daily strategy returns are expressed in percent of capital (a pnl of 0.4 means
+0.4%). Equity is therefore 100 + cumulative pnl, and drawdowns are fractions of peak equity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR as ANN


def sharpe_ratio(r: pd.Series, periods: int = ANN) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if len(r) > 1 and sd > 0 else np.nan


def sortino_ratio(r: pd.Series, periods: int = ANN, mar: float = 0.0) -> float:
    r = pd.Series(r).dropna()
    downside = np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2)) if len(r) else np.nan
    return float((r.mean() - mar) / downside * np.sqrt(periods)) if downside and downside > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    """Most negative peak-to-trough change of equity (e.g. -0.12 = -12%)."""
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    equity = 100.0 + r.cumsum()
    equity = pd.concat([pd.Series([100.0]), equity.reset_index(drop=True)], ignore_index=True)
    return float((equity / equity.cummax() - 1.0).min())


def summarize_returns(returns: pd.DataFrame, periods: int = ANN) -> pd.DataFrame:
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "n_days": len(r),
            "total_pnl_%": r.sum(),
            "mean_daily_%": r.mean(),
            "ann_vol_%": r.std(ddof=1) * np.sqrt(periods) if len(r) > 1 else np.nan,
            "sharpe": sharpe_ratio(r, periods),
            "sortino": sortino_ratio(r, periods),
            "max_drawdown": max_drawdown(r),
            "hit_rate": (r > 0).mean() if len(r) else np.nan,
        }
    return pd.DataFrame(rows).T


def regression_metrics(preds: pd.DataFrame) -> dict:
    """Forecast quality on out-of-sample rows: RMSE/MAE/R2 + daily cross-sectional rank IC."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    p = preds.dropna(subset=["y", "y_pred"])
    ic = (p.groupby("date")[["y", "y_pred"]]
           .apply(lambda g: g["y"].corr(g["y_pred"], method="spearman") if g["y_pred"].nunique() > 1 else np.nan)
           .dropna())
    return {
        "n_predictions": int(len(p)),
        "rmse": float(np.sqrt(mean_squared_error(p["y"], p["y_pred"]))),
        "mae": float(mean_absolute_error(p["y"], p["y_pred"])),
        "r2": float(r2_score(p["y"], p["y_pred"])),
        "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
        "ic_ir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan,
        "ic_positive_days": float((ic > 0).mean()) if len(ic) else np.nan,
        "direction_accuracy": float((np.sign(p["y"]) == np.sign(p["y_pred"])).mean()),
    }
