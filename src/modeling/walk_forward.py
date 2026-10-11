"""Expanding-window walk-forward validation (strictly out-of-sample).

For each fold the model is fitted on days [0, i - embargo) and predicts days [i, i + test_days);
then the window rolls forward. At every test day three portfolios are formed:
  * model_topk     - equal-weight the k permutations with the highest *predicted* P&L
  * trailing_topk  - baseline: the k permutations with the best trailing 20-day mean P&L
  * equal_weight   - baseline: all permutations
  * model_bottomk  - diagnostic: the k *lowest* predictions (should do worse than top-k)
Realised P&L of the selected permutations is the strategy return - no re-fitting on test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from src import config


def walk_forward_splits(dates, min_train_days: int = config.MIN_TRAIN_DAYS,
                        test_days: int = config.TEST_DAYS,
                        embargo_days: int = config.EMBARGO_DAYS) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    dates = np.sort(np.asarray(pd.unique(pd.Series(dates))))
    i = min_train_days
    while i < len(dates):
        train = dates[: max(i - embargo_days, 0)]
        test = dates[i: i + test_days]
        if len(train) and len(test):
            yield train, test
        i += test_days


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    returns: pd.DataFrame
    n_folds: int
    last_model: object = None
    last_X: pd.DataFrame = None
    last_y: pd.Series = None
    config: dict = field(default_factory=dict)


def _pick(g: pd.DataFrame, col: str, k: int, ascending: bool = False) -> float:
    return g.sort_values(col, ascending=ascending, na_position="last").head(k)["y"].mean()


def run_walk_forward(panel: pd.DataFrame, feature_cols: list[str], model_factory: Callable,
                     top_k: int = config.TOP_K, min_train_days: int = config.MIN_TRAIN_DAYS,
                     test_days: int = config.TEST_DAYS,
                     embargo_days: int = config.EMBARGO_DAYS) -> WalkForwardResult:
    hist = panel[panel["is_future"] == 0]
    dates = np.sort(hist["date"].unique())
    if len(dates) <= min_train_days:
        raise ValueError(f"Need more than {min_train_days} trading days, have {len(dates)}")

    preds, tail_X, tail_y, last_model, n_folds = [], [], [], None, 0
    for fold, (tr_d, te_d) in enumerate(walk_forward_splits(dates, min_train_days, test_days, embargo_days)):
        tr, te = hist[hist["date"].isin(tr_d)], hist[hist["date"].isin(te_d)]
        model = model_factory().fit(tr[feature_cols], tr["y"])
        p = te[["date", "permutation_id", "y", "mean_pnl_20"]].copy()
        p["y_pred"] = model.predict(te[feature_cols])
        p["fold"] = fold
        preds.append(p)
        tail_X.append(te[feature_cols]); tail_y.append(te["y"])
        last_model, n_folds = model, fold + 1
    predictions = pd.concat(preds, ignore_index=True)

    rows = {}
    for date, g in predictions.groupby("date"):
        rows[date] = {
            "model_topk": _pick(g, "y_pred", top_k),
            "trailing_topk": _pick(g, "mean_pnl_20", top_k),
            "equal_weight": g["y"].mean(),
            "model_bottomk": _pick(g, "y_pred", top_k, ascending=True),
        }
    returns = pd.DataFrame(rows).T.sort_index()
    returns.index.name = "date"

    return WalkForwardResult(
        predictions=predictions, returns=returns, n_folds=n_folds, last_model=last_model,
        last_X=pd.concat(tail_X[-4:]), last_y=pd.concat(tail_y[-4:]),
        config={"top_k": top_k, "min_train_days": min_train_days, "test_days": test_days,
                "embargo_days": embargo_days, "n_features": len(feature_cols)},
    )
