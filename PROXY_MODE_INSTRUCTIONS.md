# Run an honest proxy evaluation

You do not have historical trading logs. Do not invent a `data/raw/trade_logs.csv` and do not describe proxy metrics as a historical backtest.

## 1. Apply the patch

Extract `crowdwisdom-macro-quant-proxy-mode-patch.zip` directly into your existing `crowdwisdom-macro-quant` project root and allow it to overwrite matching code/documentation files. This patch intentionally does not include or replace your local API output CSVs, `.env`, or generated data.

## 2. Check that the local inputs exist

- `data/processed/events.csv`
- `data/processed/news_tavily.csv` and/or `data/processed/news_exa.csv`

The proxy command uses these existing files and does not call Apify, Tavily, or Exa.

## 3. Run tests

```powershell
pytest
```

Expected: `16 passed`.

## 4. Run the proxy evaluation

```powershell
python main.py --proxy
```

This creates synthetic zero-mean random P&L only for exercising the modeling pipeline. It does not represent actual fills, prices, transactions, or strategy performance. The generated proxy trade file is written to `data/processed/proxy/trade_logs_proxy_synthetic.csv`; the proxy database is `data/database/proxy.db`; recommendations are stored in `data/processed/proxy/recommendations.csv`.

The evaluation report and macro outlook are regenerated in `reports/`. The evaluation report carries a prominent synthetic-proxy warning. Its rankings are not trading recommendations.

## 5. Disclose remaining gaps

The currently supplied calendar covers approximately one month, not the requested 180 days. Do not claim that requirement is complete until you have fetched, merged, and verified the missing calendar history. The proxy run is a pipeline demonstration, not proof of predictive skill or profitability.
