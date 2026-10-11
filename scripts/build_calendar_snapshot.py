"""Build a source-derived calendar-only macro outlook and surprise chart.

Run from the repository root after data/processed/events.csv has been created.
This deliberately does not create trading performance metrics.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data" / "processed" / "events.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing {EVENTS_PATH}; normalize the calendar exports first.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS_PATH)
    events["event_time_utc"] = pd.to_datetime(events["event_time_utc"], utc=True, errors="coerce")
    for col in ("actual", "forecast", "previous", "surprise"):
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events = events.dropna(subset=["event_time_utc"]).sort_values("event_time_utc")
    if events.empty:
        raise ValueError("No valid event timestamps in the normalized calendar.")

    plot = events.dropna(subset=["surprise"]).tail(22).copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if plot.empty:
        ax.text(.5, .5, "No events with both actual and forecast values", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        plot["label"] = plot["event_name"].astype(str).str.slice(0, 34)
        y = range(len(plot))
        ax.barh(list(y), plot["surprise"].values)
        ax.set_yticks(list(y), plot["label"].values, fontsize=8)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Actual − forecast (source units; not comparable across event types)")
        ax.set_title("US Macro Calendar: Reported Actual-versus-Forecast Surprises")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "calendar_actual_forecast_surprises.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    counts = events["event_type"].value_counts().reindex(["CPI", "PCE", "FOMC", "EMPLOYMENT", "GDP"]).dropna().astype(int)
    with_values = events.dropna(subset=["actual", "forecast"])
    latest = events.sort_values("event_time_utc", ascending=False).head(10)

    def fmt(value) -> str:
        return "—" if pd.isna(value) else f"{value:,.4g}"

    lines = [
        "# Macro Outlook — Calendar-Based Snapshot", "",
        "> **Data limitation:** This is a calendar-only snapshot from the supplied Investing.com/Apify exports. It is not a full news-informed macro outlook: Tavily/Exa news data and genuine trading logs were not supplied. No market direction or trading edge is inferred from this calendar alone.", "",
        "## Coverage", "",
        f"- Source exports: 200 raw rows across high- and medium-importance runs; {len(events)} classified US macro events after normalization/deduplication.",
        f"- Classified event timestamps: {events['event_time_utc'].min():%Y-%m-%d %H:%M} UTC to {events['event_time_utc'].max():%Y-%m-%d %H:%M} UTC.",
        "- This is about one month of data, not the required 180-day history. The data should not be described as a 180-day sample.",
        "- Event types: " + ", ".join(f"{name}: {int(count)}" for name, count in counts.items()) + ".",
        f"- Actual and forecast are both present for {len(with_values)} of {len(events)} classified events; many speeches and calendar entries legitimately have no numeric surprise.", "",
        "## Latest classified calendar entries", "",
        "| UTC date/time | Type | Event | Importance | Actual | Forecast | Previous | Actual − forecast |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in latest.iterrows():
        lines.append(
            f"| {row['event_time_utc']:%Y-%m-%d %H:%M} | {row.get('event_type', '—')} | "
            f"{str(row.get('event_name', '')).replace('|', '/')} | {row.get('importance', '—')} | "
            f"{fmt(row['actual'])} | {fmt(row['forecast'])} | {fmt(row['previous'])} | {fmt(row['surprise'])} |"
        )
    lines += [
        "", "## How to interpret the data", "",
        "- `surprise` is numeric actual minus numeric forecast where both exist. Values retain each indicator's native units; an inflation surprise is not directly comparable with a payroll surprise.",
        "- This short sample is insufficient to estimate robust event-specific market reactions. A release surprise alone does not reveal the market response without instrument-level price/trade data.",
        "- Next: ingest recent macroeconomic news from Tavily and Exa, then align timestamped sentiment with the calendar while enforcing publication-time and leakage controls.",
        "", "## Surprise visualization", "",
        "![Calendar actual-versus-forecast surprises](figures/calendar_actual_forecast_surprises.png)",
        "", "## Trading-model status", "",
        "The repository's evaluation report is explicitly labelled **DEMO / SYNTHETIC**. A genuine walk-forward evaluation requires a historical trade log with `trade_id`, `entry_time`, `exit_time`, `pnl`, and strategy-parameter columns. Do not report demo Sharpe, Sortino, drawdown, or model recommendations as live results.", "",
    ]
    (REPORTS / "macro_outlook.md").write_text("\n".join(lines), encoding="utf-8")

    audit = """# Calendar Data Audit

- Input: two Investing.com economic calendar exports retrieved via Apify.
- Raw rows: 52 in the high-importance export and 148 in the medium-importance export (200 combined).
- Normalized rows: 57 classified US events retained for CPI, PCE, FOMC, employment, and GDP.
- Date parsing: source dates use day-first `DD/MM/YYYY`; normalization explicitly supports that format.
- Timezone: normalized timestamps currently interpret the source times as UTC, then derive America/New_York local timestamps. Verify the actor's `timeZone` setting against a sample before treating this assumption as definitive.
- Coverage: raw calendar rows span 2026-09-07 through 2026-10-07; classified events span 2026-09-09 through 2026-10-07.
- Coverage gap: approximately one month, not the requested 180 days.
- Missing values: no actual/forecast/previous values were fabricated.
- Not supplied: Tavily/Exa news exports and real trading logs. This calendar alone cannot support a genuine strategy backtest or validated strategy P&L predictions.
"""
    (REPORTS / "calendar_data_audit.md").write_text(audit, encoding="utf-8")
    print(f"Wrote {REPORTS / 'macro_outlook.md'} and {FIGURES / 'calendar_actual_forecast_surprises.png'}")


if __name__ == "__main__":
    main()
