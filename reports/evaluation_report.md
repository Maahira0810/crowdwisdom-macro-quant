# Evaluation Report - Walk-Forward Strategy Permutation Model

> **PROXY EVALUATION — SYNTHETIC P&L, NOT REAL TRADES.** Calendar events and news are loaded from the collected source exports, but every trading outcome is randomly generated with zero-mean noise and no planted predictive relationship. Sharpe, Sortino, drawdown, forecast scores and rankings are software-test outputs only; they are not evidence of profitability.


## 1. Setup
- Model: `hgb`  |  Features: 41  |  Permutations: 27  |  Trades: 3483
- Target: next-day P&L (% of allocated capital) of each (day, permutation) row
- Validation: expanding-window walk-forward, min train 40 days, test block 5 days,
  embargo 1 day, **18 folds**, out-of-sample span 2026-06-09 -> 2026-10-09 (89 days)
- Selection rule: each day hold the top-5 permutations by predicted P&L, equal weight
- Leakage controls: realised data lagged 1 day; only scheduled calendar info unlagged; model never refit on test rows
- No transaction costs, slippage or capacity limits are modelled.

## 2. Strategy performance (out-of-sample)
|  | n_days | total_pnl_% | mean_daily_% | ann_vol_% | sharpe | sortino | max_drawdown | hit_rate |
|---|---|---|---|---|---|---|---|---|
| model_topk | 89.000 | 0.872 | 0.010 | 3.606 | 0.684 | 0.946 | -0.015 | 0.483 |
| trailing_topk | 89.000 | -1.407 | -0.016 | 3.966 | -1.004 | -1.414 | -0.038 | 0.438 |
| equal_weight | 89.000 | 0.503 | 0.006 | 1.883 | 0.757 | 1.088 | -0.007 | 0.483 |
| model_bottomk | 89.000 | 0.024 | 0.000 | 4.478 | 0.015 | 0.023 | -0.025 | 0.449 |

Sharpe / Sortino annualised with sqrt(252); max drawdown is relative to peak equity (start = 100).
**Any apparent ranking/performance differences are artifacts of synthetic outcomes and must not be interpreted as evidence of predictive skill.**

![equity](figures/walk_forward_equity.png)

## 3. Forecast quality (out-of-sample)
|  | value |
|---|---|
| n_predictions | 2403.0000 |
| rmse | 0.6158 |
| mae | 0.4930 |
| r2 | -0.0435 |
| mean_rank_ic | 0.0107 |
| ic_ir | 0.0578 |
| ic_positive_days | 0.5056 |
| direction_accuracy | 0.5031 |

`mean_rank_ic` is the average daily cross-sectional Spearman correlation between predicted and realised P&L; it is the
metric that matters for a ranking/selection use-case. R2 is typically near zero for noisy P&L and should not be over-read.

## 4. Feature importance (permutation, last 4 folds)
|  | importance | std |
|---|---|---|
| ret_lag1 | 0.01187 | 0.00155 |
| vol_pnl_5 | 0.01043 | 0.00199 |
| vol_pnl_20 | 0.00846 | 0.00111 |
| mean_pnl_5 | 0.00829 | 0.00163 |
| hawk_dove_ewm | 0.00805 | 0.00132 |
| day_of_month | 0.00802 | 0.00211 |
| mean_pnl_20 | 0.00601 | 0.00158 |
| mkt_mean_pnl_5 | 0.00480 | 0.00107 |
| dow | 0.00431 | 0.00072 |
| stop_loss | 0.00396 | 0.00130 |

## 5. Synthetic proxy rankings (not trading recommendations)
| permutation_id | date | lookback | stop_loss | take_profit | pred_pnl_% |
|---|---|---|---|---|---|
| L20_SL1.0_TP3.0 | 2026-10-12 00:00:00 | 20 | 1.000 | 3.000 | 0.054 |
| L20_SL2.0_TP1.0 | 2026-10-12 00:00:00 | 20 | 2.000 | 1.000 | 0.044 |
| L50_SL2.0_TP1.0 | 2026-10-12 00:00:00 | 50 | 2.000 | 1.000 | 0.040 |
| L20_SL1.0_TP2.0 | 2026-10-12 00:00:00 | 20 | 1.000 | 2.000 | 0.036 |
| L50_SL1.0_TP3.0 | 2026-10-12 00:00:00 | 50 | 1.000 | 3.000 | 0.021 |

## 6. Macro visuals
![sentiment](figures/macro_sentiment_timeline.png)
![surprise](figures/macro_surprise_matrix.png)
![regime](figures/permutation_regime_matrix.png)

## 7. Caveats
- Few macro releases fit in 180 days (about 6 CPI, 6 jobs, 4 FOMC); event-specific conclusions have wide error bars.
- Sentiment uses a keyword lexicon; upgrade to a finance-tuned classifier for production.
- Top-k selection from many permutations carries multiple-testing risk; confirm on a longer, untouched hold-out.
