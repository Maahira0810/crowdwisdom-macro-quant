# Calendar Data Audit

- Input: two Investing.com economic calendar exports retrieved via Apify.
- Raw rows: 52 in the high-importance export and 148 in the medium-importance export (200 combined).
- Normalized rows: 57 classified US events retained for CPI, PCE, FOMC, employment, and GDP.
- Date parsing: source dates use day-first `DD/MM/YYYY`; normalization explicitly supports that format.
- Timezone: normalized timestamps currently interpret the source times as UTC, then derive America/New_York local timestamps. Verify the actor's `timeZone` setting against a sample before treating this assumption as definitive.
- Coverage: raw calendar rows span 2026-09-07 through 2026-10-07; classified events span 2026-09-09 through 2026-10-07.
- Coverage gap: approximately one month, not the requested 180 days.
- Missing values: no actual/forecast/previous values were fabricated.
- Not supplied: Tavily/Exa news exports and real trading logs. This calendar alone cannot support a genuine strategy backtest or validated strategy P&L predictions.
