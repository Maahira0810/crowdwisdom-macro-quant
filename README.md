# CrowdWisdom Macro Quant

A Python pipeline that (1) scrapes a macro-economic calendar and macro news, (2) stores everything in a
SQLite database and joins it to historical trading logs, (3) engineers leakage-safe macro + time features,
(4) trains a model that forecasts the P&L of every strategy permutation, validates it with **walk-forward**
out-of-sample testing, and (5) produces macro-sentiment charts plus an evaluation report
(Sharpe, Sortino, Max Drawdown).

```
 Apify calendar ─┐                                   ┌─> walk-forward model ─> evaluation_report.md
 Tavily news ────┼─> SQLite (events/news/trades) ─> features ─┤
 Exa news ───────┤        + SQL view v_trades_macro  └─> macro_matrix charts ─> macro_outlook.md
 trade_logs.csv ─┘
```

## Quick start (no API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --demo          # synthetic data, runs end to end in ~10 s
pytest                         # tests include date parsing and look-ahead-leakage checks
```

> **Demo data is synthetic** with planted effects. It proves the plumbing works; it is not evidence of edge.

## Supplied calendar data (current status)

The repository includes the two supplied Apify/Investing.com calendar exports combined in
`data/raw/apify_investing_calendar_combined_raw.csv`, plus normalized macro events in
`data/processed/events.csv`. The current sample has 200 raw rows and 57 classified US CPI/PCE/FOMC/jobs/GDP
events after normalization. It covers about one month, not the requested 180 days.

The source dates are day-first (`DD/MM/YYYY`); the normalizer handles this format. Confirm the Actor's
`timeZone` input before relying on timestamps. To re-normalize the included raw export and rebuild the calendar-only outlook and surprise chart, run:

```bash
python scripts/normalize_calendar_export.py
python scripts/build_calendar_snapshot.py
```

Tavily/Exa news exports and a genuine historical trade log are not included. Consequently,
`reports/evaluation_report.md` contains **synthetic demo metrics only**, not actual strategy performance.
Do not submit those metrics as real results. A live model evaluation is not possible until the trade log is supplied.

## Live run

1. `cp .env.example .env` and fill in `APIFY_TOKEN`, `APIFY_CALENDAR_ACTOR_ID`, `TAVILY_API_KEY`, `EXA_API_KEY`.
2. Put your trade log at `data/raw/trade_logs.csv` (schema below).
3. Ingest (each step writes raw JSON to `data/raw/` and a clean CSV to `data/processed/`):
   ```bash
   python -m src.ingestion.apify_calendar --days 180
   python -m src.ingestion.tavily_news    --windows 6 --window-days 30
   python -m src.ingestion.exa_news       --windows 6 --window-days 30
   ```
4. `python main.py` - features -> database -> walk-forward -> charts -> reports. Add `--refresh` to re-scrape first.

### Trade-log schema (`data/raw/trade_logs.csv`)
| column | notes |
|---|---|
| `trade_id` | unique id |
| `entry_time`, `exit_time` | ISO timestamps; naive values are treated as UTC |
| `pnl` | per-trade P&L in **percent of allocated capital** |
| strategy parameters | default `lookback, stop_loss, take_profit`; any other numeric columns are auto-detected. A `permutation_id` column is optional - otherwise it is built from the parameters. |

### Apify actor notes
Actors differ in output schema. `src/ingestion/apify_calendar.py::FIELD_CANDIDATES` lists the field names
understood (event/actual/forecast/previous/date/time/country...). If your actor uses others, add them there.
Set `APIFY_SOURCE_TZ` if the actor returns naive local times, and `APIFY_CALENDAR_INPUT_JSON` for that actor's
input schema (`{start_date}` / `{end_date}` are substituted). **Check one scraped sample against the actor's
site** to confirm the timezone assumption - a wrong timezone silently shifts features by hours.

## How the model works

* **Unit of prediction:** one row per (trading day, permutation); target = that permutation's P&L that day.
* **Features:** lagged permutation stats (mean P&L, vol, win-rate over 5/20d, cross-sectional rank), permutation
  parameters, scheduled-event flags (CPI/FOMC/jobs/PCE/GDP day, days to next / since last release), lagged
  surprise z-scores, lagged news tone (hawkish/dovish, risk-on/off), calendar features (dow, month-end, sin/cos).
* **Leakage rule:** realised data (actuals, news) is lagged 1 day; only calendar-*scheduled* info is unlagged.
  `tests/test_walk_forward_leakage.py` shocks future P&L and asserts earlier features do not change.
* **Validation:** expanding-window walk-forward (default: 40-day minimum train, 5-day test blocks, 1-day embargo).
  Each test day forms equal-weight portfolios of the top-k permutations by predicted P&L and compares them to
  trailing-performance and equal-weight baselines.
* **Metrics:** Sharpe, Sortino (annualised, sqrt 252), max drawdown, hit rate; plus RMSE/MAE/R2 and daily rank-IC.
* **Output for the "future period":** `data/processed/recommendations.csv` - top-k permutations for the next trading day.

## Layout

```
main.py                         one-command pipeline (--demo | --refresh | --model hgb|ridge | --top-k | ...)
src/config.py                   paths, constants, env credentials
src/database.py                 SQLite helpers + joined view v_trades_macro
src/pipeline.py                 load -> features -> store -> model orchestration
src/synthetic.py                demo data generator
src/reporting.py                writes the two markdown reports
src/ingestion/                  apify_calendar.py  tavily_news.py  exa_news.py  common.py
src/features/                   time_features.py   macro_features.py
src/modeling/                   train.py  walk_forward.py  metrics.py
src/visualization/macro_matrix.py   sentiment timeline, surprise matrix, permutation x regime matrix, equity curves
notebooks/01_exploration.ipynb  guided walkthrough
reports/                        macro_outlook.md, evaluation_report.md, calendar_data_audit.md, figures/
tests/                          metrics, parsing/timezones, leakage + walk-forward
```

Database tables: `events`, `news`, `trades_enriched`, `daily_macro`, and view `v_trades_macro`
(every trade joined to its trade-date macro features):
```sql
SELECT is_cpi_day, AVG(pnl), COUNT(*) FROM v_trades_macro GROUP BY 1;
```

## Known limitations
* No transaction costs, slippage or capacity modelling.
* News sentiment is a keyword lexicon (transparent baseline); swap `score_text` for FinBERT/LLM scoring.
* ~180 days contains few CPI/FOMC/jobs events, so event-specific effects are statistically thin.
* Picking top-k from many permutations has multiple-testing risk; confirm on a longer untouched hold-out.
* `macro_outlook.md` is auto-drafted from the data; add your own commentary before submitting.

## Submission checklist
- [ ] Repo link  - [ ] Apify actor ID + Tavily/Exa keys (send privately; never commit `.env`)  - [ ] Extend calendar coverage to 180 days  - [ ] Ingest Tavily/Exa news  - [ ] Supply real trade logs and regenerate `reports/evaluation_report.md` on live data
## Honest proxy evaluation (when no trade logs are available)

If you do not have genuine historical trades, do **not** create a fake `data/raw/trade_logs.csv`.
Instead, after collecting the real calendar and news CSVs, run:

```powershell
python main.py --proxy
```

This mode uses the collected `data/processed/events.csv`, `news_tavily.csv`, and/or `news_exa.csv`,
but creates a separate synthetic trade panel with zero-mean random P&L and no planted relationship to
features. The generated file is saved as `data/processed/proxy/trade_logs_proxy_synthetic.csv`; the
database is `data/database/proxy.db`; and recommendations are saved under `data/processed/proxy/`.
The evaluation report is explicitly labelled **PROXY EVALUATION — SYNTHETIC P&L, NOT REAL TRADES**.
Its Sharpe, Sortino, drawdown, forecast scores and strategy rankings only test the software pipeline and
must not be described as historical results or evidence that any strategy is profitable.

The proxy mode does not call APIs or scrape the calendar. It requires the already-generated calendar and
at least one news CSV. The current calendar export spans only about one month; fetch and verify the full
180-day range before claiming that assignment requirement is met.

