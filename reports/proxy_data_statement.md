# Proxy-data statement

This project does not contain genuine historical trading logs. The proxy evaluation generates synthetic daily P&L values solely to exercise the feature pipeline, model fitting, walk-forward splitting, metric calculations, and chart/report generation. The synthetic P&L is independent zero-mean random noise; no real trades, fills, transaction costs, slippage, or executable prices are represented.

The calendar and news inputs are the collected Investing.com/Apify, Tavily, and Exa exports present locally at run time. Their availability and quality do not make the synthetic P&L real. The current supplied calendar data covers approximately one month, so it does not yet meet the 180-day calendar requirement.

Any Sharpe ratio, Sortino ratio, maximum drawdown, RMSE, rank correlation, feature importance, or recommended permutation emitted in proxy mode is a software-validation artifact only. It must not be represented as a genuine backtest, a forecast of live profitability, or investment advice.
