import numpy as np
import pandas as pd

from src.modeling.metrics import max_drawdown, sharpe_ratio, sortino_ratio, summarize_returns


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.1] * 10)))


def test_sharpe_known_value():
    r = pd.Series([1.0, -1.0] * 50)
    r.iloc[0] = 2.0  # mean slightly positive
    assert sharpe_ratio(r) > 0


def test_max_drawdown_known():
    # equity: 100 -> 110 -> 99 -> 104.5 ; peak 110, trough 99 => -10%
    r = pd.Series([10.0, -11.0, 5.5])
    assert abs(max_drawdown(r) - (-0.10)) < 1e-9


def test_max_drawdown_no_loss_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.0, 1.0])) == 0.0


def test_sortino_ignores_upside_volatility():
    a = pd.Series([1.0, 3.0, 1.0, 3.0, -1.0, 1.0])
    b = pd.Series([1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
    assert sortino_ratio(a) > sortino_ratio(b)


def test_summary_columns():
    s = summarize_returns(pd.DataFrame({"x": np.random.default_rng(0).normal(0.1, 1, 100)}))
    assert {"sharpe", "sortino", "max_drawdown"} <= set(s.columns)
