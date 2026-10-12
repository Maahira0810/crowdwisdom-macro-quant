"""Auto-generate reports/evaluation_report.md and reports/macro_outlook.md from pipeline outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features.macro_features import score_news

DEMO_BANNER = ("> **DEMO MODE - FULLY SYNTHETIC DATA.** Calendar, news and trading outcomes are generated. "
               "This validates software plumbing only and says nothing about real performance.\n\n")
PROXY_BANNER = ("> **PROXY EVALUATION — SYNTHETIC P&L, NOT REAL TRADES.** Calendar events and news are loaded "
                "from the collected source exports, but every trading outcome is randomly generated with zero-mean "
                "noise and no planted predictive relationship. Sharpe, Sortino, drawdown, forecast scores and "
                "rankings are software-test outputs only; they are not evidence of profitability.\n\n")


def df_to_md(df: pd.DataFrame, digits: int = 3) -> str:
    d = df.copy()
    cols = [d.index.name or ""] + [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in d.iterrows():
        vals = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) and pd.notna(v) else ("" if pd.isna(v) else str(v))
                for v in row.tolist()]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def feature_importance(model, X: pd.DataFrame, y: pd.Series, top: int = 10) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=8, random_state=config.RANDOM_STATE,
                               scoring="neg_mean_squared_error")
    return (pd.DataFrame({"importance": r.importances_mean, "std": r.importances_std}, index=X.columns)
            .sort_values("importance", ascending=False).head(top))


def write_evaluation_report(wf, summary: pd.DataFrame, reg: dict, recs: pd.DataFrame, n_trades: int,
                            n_perms: int, model_kind: str, demo: bool, path: Path | None = None, proxy: bool = False) -> Path:
    path = Path(path or config.REPORTS_DIR / "evaluation_report.md")
    r = wf.returns
    c = wf.config
    imp = feature_importance(wf.last_model, wf.last_X, wf.last_y)
    best = summary.loc["model_topk"]
    base = summary.loc[["trailing_topk", "equal_weight"]]
    verdict = ("Any apparent ranking/performance differences are artifacts of synthetic outcomes and must not be interpreted as evidence of predictive skill."
               if proxy else ("The model portfolio beats both baselines on Sharpe in this sample."
               if best["sharpe"] > base["sharpe"].max()
               else "The model portfolio does NOT beat the best baseline on Sharpe in this sample."))
    txt = ["# Evaluation Report - Walk-Forward Strategy Permutation Model\n"]
    if demo:
        txt.append(DEMO_BANNER)
    elif proxy:
        txt.append(PROXY_BANNER)
    txt.append(f"""## 1. Setup
- Model: `{model_kind}`  |  Features: {c['n_features']}  |  Permutations: {n_perms}  |  Trades: {n_trades}
- Target: next-day P&L (% of allocated capital) of each (day, permutation) row
- Validation: expanding-window walk-forward, min train {c['min_train_days']} days, test block {c['test_days']} days,
  embargo {c['embargo_days']} day, **{wf.n_folds} folds**, out-of-sample span {r.index.min().date()} -> {r.index.max().date()} ({len(r)} days)
- Selection rule: each day hold the top-{c['top_k']} permutations by predicted P&L, equal weight
- Leakage controls: realised data lagged 1 day; only scheduled calendar info unlagged; model never refit on test rows
- No transaction costs, slippage or capacity limits are modelled.

## 2. Strategy performance (out-of-sample)
{df_to_md(summary)}

Sharpe / Sortino annualised with sqrt({config.TRADING_DAYS_PER_YEAR}); max drawdown is relative to peak equity (start = 100).
**{verdict}**

![equity](figures/walk_forward_equity.png)

## 3. Forecast quality (out-of-sample)
{df_to_md(pd.DataFrame(reg, index=['value']).T, 4)}

`mean_rank_ic` is the average daily cross-sectional Spearman correlation between predicted and realised P&L; it is the
metric that matters for a ranking/selection use-case. R2 is typically near zero for noisy P&L and should not be over-read.

## 4. Feature importance (permutation, last 4 folds)
{df_to_md(imp, 5)}

## 5. {'Synthetic proxy rankings (not trading recommendations)' if proxy else 'Recommended permutations for the next trading day'}
{df_to_md(recs.set_index('permutation_id'))}

## 6. Macro visuals
![sentiment](figures/macro_sentiment_timeline.png)
![surprise](figures/macro_surprise_matrix.png)
![regime](figures/permutation_regime_matrix.png)

## 7. Caveats
- Few macro releases fit in 180 days (about 6 CPI, 6 jobs, 4 FOMC); event-specific conclusions have wide error bars.
- Sentiment uses a keyword lexicon; upgrade to a finance-tuned classifier for production.
- Top-k selection from many permutations carries multiple-testing risk; confirm on a longer, untouched hold-out.
""")
    path.write_text("\n".join(txt), encoding="utf-8")
    return path


def write_macro_outlook(events: pd.DataFrame, news: pd.DataFrame, daily: pd.DataFrame, demo: bool,
                        path: Path | None = None, proxy: bool = False, as_of: pd.Timestamp | None = None,
                        narrative: dict | None = None) -> Path:
    """Write reports/macro_outlook.md.

    `as_of` pins the report date (use the data's last date for reproducible output; defaults to today).
    `narrative` = {"text", "meta", "unverified"} adds the LLM-written section (see src/llm_summary.py).
    """
    path = Path(path or config.REPORTS_DIR / "macro_outlook.md")
    today = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now()).normalize()
    d = daily.set_index("date")
    past = d.loc[:today]
    ev = events.copy()
    ev["event_date"] = ev["event_time_est"].dt.tz_localize(None).dt.normalize()

    rel = ev[(ev["event_date"] <= today) & ev["actual"].notna()]
    last = (rel.sort_values("event_time_utc").groupby("event_type").tail(1).set_index("event_type")
            [["event_date", "event_name", "actual", "forecast", "surprise"]])
    last["event_date"] = last["event_date"].dt.strftime("%Y-%m-%d")
    last["event_name"] = last["event_name"].str.replace(r"\s+", " ", regex=True).str.strip()

    def lvl(col):
        now, prev = past[col].iloc[-1], past[col].iloc[-31] if len(past) > 31 else past[col].iloc[0]
        return now, now - prev
    rs, rs_chg = lvl("viz_risk_sentiment")
    hd, hd_chg = lvl("viz_hawk_dove")
    tone = "hawkish" if hd > 0.1 else "dovish" if hd < -0.1 else "balanced"
    risk = "risk-on" if rs > 0.1 else "risk-off" if rs < -0.1 else "mixed"

    up = ev[(ev["event_date"] > today) & (ev["event_date"] <= today + pd.Timedelta(days=14))]
    up_lines = [f"- {r.event_date:%a %b %d}: {r.event_name} (forecast {r.forecast:g})" if pd.notna(r.forecast)
                else f"- {r.event_date:%a %b %d}: {r.event_name}" for r in up.sort_values("event_time_utc").itertuples()]

    sc = score_news(news)
    sc["score"] = pd.to_numeric(sc["score"], errors="coerce").fillna(0.0)
    recent = (sc[sc["date"] >= today - pd.Timedelta(days=14)]
              .sort_values(["score", "published_at", "title"], ascending=[False, False, True]).head(5))
    heads = [f"- {r.title} ({r.source})" for r in recent.itertuples()]

    surprise_bias = past["viz_surprise_ewm"].iloc[-1]
    data_line = ("Recent data have surprised to the upside (hot / strong), supporting a higher-for-longer rate path."
                 if surprise_bias > 0.2 else
                 "Recent data have surprised to the downside (soft / cool), supporting rate-cut expectations."
                 if surprise_bias < -0.2 else "Recent data surprises have been mixed, with no clear directional bias.")

    txt = [f"# Macro Outlook - as of {today:%Y-%m-%d}\n"]
    if demo:
        txt.append(DEMO_BANNER)
    elif proxy:
        txt.append("> **DATA NOTE.** This outlook uses the collected calendar/news exports. Trading performance in the companion evaluation report uses synthetic proxy P&L, not actual trades. Calendar coverage is incomplete and should not be interpreted as a complete 180-day macro history.\n\n")

    if narrative:
        m = narrative.get("meta", {})
        txt.append("## Outlook (LLM-written)\n")
        txt.append(f"_Written by {m.get('generated_by', 'an LLM')} (model: {m.get('model', 'n/a')}, {m.get('generated_on', '')}) "
                   "from the structured context in `reports/llm_context.json`; it may use no other facts._\n")
        txt.append(narrative["text"].strip() + "\n")
        if narrative.get("unverified"):
            txt.append("_Automated check - figures in the text not found verbatim in the context "
                       f"(derived or rounded values show up here; verify): {', '.join(narrative['unverified'])}._\n")
        txt.append("## Signal snapshot (rule-based)\n")
    else:
        txt.append("## Headline view\n")
    txt.append(f"""News flow is **{tone}** on policy (tone index {hd:+.2f}, {hd_chg:+.2f} over 30d) and **{risk}** on markets
(sentiment {rs:+.2f}, {rs_chg:+.2f} over 30d). {data_line}

![Macro sentiment and data surprises over time](figures/macro_sentiment_timeline.png)

![Macro surprise matrix (event type by week)](figures/macro_surprise_matrix.png)

## Latest releases
{df_to_md(last, 2)}

## Next 14 days
{chr(10).join(up_lines) if up_lines else '- No scheduled releases in the calendar window.'}

## Notable recent headlines
{chr(10).join(heads) if heads else '- No recent articles.'}

## Implications for the strategy book
""")
    if proxy or demo:
        txt.append("- Strategy P&L in this run is synthetic, so no claim is made about which permutations the macro backdrop favours.\n"
                   "- The pipeline feeds scheduled-event flags, lagged surprise z-scores and lagged news tone to the model; "
                   "with real trade logs the evaluation report shows whether they add any out-of-sample value.\n")
    else:
        txt.append("- Event days (CPI / FOMC / jobs) can change which permutations work; the model includes scheduled-event flags\n"
                   "  and lagged surprise features so allocation tilts ahead of releases.\n"
                   f"- A {tone} tone with {risk} sentiment is fed to the model as lagged features, not hard-coded rules.\n")
    path.write_text("\n".join(txt), encoding="utf-8")
    return path
