import numpy as np
import pandas as pd

from src.features.macro_features import build_daily_macro
from src.features.time_features import add_event_proximity, add_trade_time_features
from src.modeling.train import build_panel, get_feature_cols, make_model
from src.modeling.walk_forward import run_walk_forward, walk_forward_splits
from src.synthetic import make_demo_data

TODAY = pd.Timestamp("2026-06-30")
PARAMS = ["lookback", "stop_loss", "take_profit"]


def _build(trades, events, news):
    enriched = add_event_proximity(add_trade_time_features(trades), events)
    daily = build_daily_macro(events, news, enriched["trade_date"].min() - pd.Timedelta(days=30),
                              enriched["trade_date"].max() + pd.Timedelta(days=14))
    return build_panel(enriched, daily, PARAMS)


def test_splits_never_overlap_and_respect_embargo():
    dates = pd.bdate_range("2026-01-01", periods=100)
    n = 0
    for tr, te in walk_forward_splits(dates, min_train_days=40, test_days=5, embargo_days=1):
        assert tr.max() < te.min()
        assert len(set(tr) & set(te)) == 0
        gap = np.searchsorted(dates.values, te.min()) - np.searchsorted(dates.values, tr.max())
        assert gap >= 2          # embargo=1 => at least one skipped day between train and test
        n += 1
    assert n > 5


def test_features_at_date_d_do_not_depend_on_pnl_at_or_after_d():
    events, news, trades = make_demo_data(days=90, seed=3, today=TODAY)
    base = _build(trades, events, news)
    cutoff = base.loc[base["is_future"] == 0, "date"].sort_values().unique()[-10]

    perturbed = trades.copy()
    mask = add_trade_time_features(perturbed)["trade_date"] >= cutoff
    perturbed.loc[mask, "pnl"] += 100.0                       # huge shock from `cutoff` onward
    alt = _build(perturbed, events, news)

    feat = get_feature_cols(base)
    b = base[base["date"] <= cutoff].sort_values(["date", "permutation_id"])[feat].reset_index(drop=True)
    a = alt[alt["date"] <= cutoff].sort_values(["date", "permutation_id"])[feat].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a)                       # features up to and including cutoff unchanged


def test_walk_forward_runs_and_returns_aligned_oos_only():
    events, news, trades = make_demo_data(days=120, seed=5, today=TODAY)
    panel = _build(trades, events, news)
    feat = get_feature_cols(panel)
    res = run_walk_forward(panel, feat, lambda: make_model("ridge"), top_k=3, min_train_days=30, test_days=5)
    hist_dates = np.sort(panel.loc[panel["is_future"] == 0, "date"].unique())
    assert res.returns.index.min() >= hist_dates[30]          # nothing predicted inside the first train window
    assert res.predictions["y_pred"].notna().all()
    assert {"model_topk", "trailing_topk", "equal_weight"} <= set(res.returns.columns)
