"""CrowdWisdom macro quant pipeline - one command end to end.

  python main.py --demo                  # fully synthetic data, no API keys needed
  python main.py --proxy                 # collected calendar/news + synthetic P&L (not a real backtest)
  python main.py                         # live: uses data/processed/*.csv + data/raw/trade_logs.csv
  python main.py --refresh               # re-scrape Apify / Tavily / Exa first
"""
from __future__ import annotations

import argparse
import logging

from src import config, pipeline, reporting
from src.visualization import macro_matrix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use fully synthetic data (no API keys)")
    ap.add_argument("--proxy", action="store_true", help="use real collected calendar/news plus synthetic proxy P&L (not a real backtest)")
    ap.add_argument("--refresh", action="store_true", help="re-run Apify/Tavily/Exa ingestion")
    ap.add_argument("--model", choices=["hgb", "ridge"], default="hgb")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--min-train-days", type=int, default=config.MIN_TRAIN_DAYS)
    ap.add_argument("--test-days", type=int, default=config.TEST_DAYS)
    ap.add_argument("--seed", type=int, default=7, help="demo-data seed")
    args = ap.parse_args()
    if args.demo and args.proxy:
        ap.error("Choose either --demo or --proxy, not both")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("main")

    if args.proxy:
        if args.refresh:
            ap.error("--proxy uses already-collected CSVs; do not combine it with --refresh")
        events, news, trades, mode = pipeline.load_proxy_inputs(args.seed)
    else:
        events, news, trades, mode = pipeline.load_inputs(args.demo, args.refresh, args.seed)
    demo = mode == "demo"
    proxy = mode == "proxy"
    log.info("Loaded %d events, %d articles, %d trades (%s)", len(events), len(news), len(trades), mode)

    feats = pipeline.build_features(events, news, trades)
    db_path = (config.DEMO_DB_PATH if demo else config.DB_DIR / "proxy.db" if proxy else config.DB_PATH)
    pipeline.store(events, news, feats, db_path)
    log.info("Stored tables + view v_trades_macro in %s", db_path)

    wf, summary, reg, final, recs = pipeline.run_model(feats, args.model, args.top_k,
                                                       args.min_train_days, args.test_days)
    figs = macro_matrix.make_all(feats["daily_macro"], events, feats["panel"], wf.returns)
    output_dir = config.DEMO_DIR if demo else (config.PROCESSED_DIR / "proxy" if proxy else config.PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(output_dir / "recommendations.csv", index=False)
    ev_path = reporting.write_evaluation_report(
        wf, summary, reg, recs, len(trades), trades["permutation_id"].nunique() if "permutation_id" in trades else
        feats["panel"]["permutation_id"].nunique(), args.model, demo, proxy=proxy)
    mo_path = reporting.write_macro_outlook(events, news, feats["daily_macro"], demo, proxy=proxy)

    print("\n=== Out-of-sample performance ===")
    print(summary.round(3).to_string())
    print("\n=== Forecast quality ===")
    print({k: round(v, 4) for k, v in reg.items()})
    print("\n=== Synthetic proxy rankings (NOT trading recommendations) ===" if proxy else "\n=== Next-day recommended permutations ===")
    print(recs.assign(date=recs["date"].dt.date).round(3).to_string(index=False))
    print(f"\nReports: {ev_path}, {mo_path}\nFigures: {[p.name for p in figs]}")


if __name__ == "__main__":
    main()
