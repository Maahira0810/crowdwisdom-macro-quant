import pandas as pd

from src.synthetic import make_proxy_trades


def test_proxy_trades_are_explicitly_labelled_and_have_required_columns():
    trades = make_proxy_trades(days=20, seed=5, today=pd.Timestamp("2026-10-09"))
    required = {"trade_id", "permutation_id", "lookback", "stop_loss", "take_profit",
                "entry_time", "exit_time", "pnl", "data_provenance"}
    assert required.issubset(trades.columns)
    assert trades["data_provenance"].eq("SYNTHETIC_PROXY_NOT_REAL_TRADES").all()
    assert trades["permutation_id"].nunique() == 27
    assert trades["trade_id"].is_unique
    assert isinstance(trades["entry_time"].dtype, pd.DatetimeTZDtype)
