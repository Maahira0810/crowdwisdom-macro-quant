"""Central configuration: paths, constants, credentials (read from .env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ---- paths -----------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_DIR = ROOT / "data" / "processed" / "demo"
DB_DIR = ROOT / "data" / "database"
DB_PATH = DB_DIR / "macro_quant.db"
DEMO_DB_PATH = DB_DIR / "demo.db"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TRADE_LOGS_PATH = RAW_DIR / "trade_logs.csv"

# ---- time ------------------------------------------------------------------
TZ_UTC = "UTC"
TZ_EST = "America/New_York"   # handles EST/EDT; US macro releases are anchored here
LOOKBACK_DAYS = 180
FUTURE_DAYS = 14              # also pull upcoming events so calendar-known features exist

# ---- credentials -----------------------------------------------------------
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
APIFY_CALENDAR_ACTOR_ID = os.getenv("APIFY_CALENDAR_ACTOR_ID", "")
APIFY_SOURCE_TZ = os.getenv("APIFY_SOURCE_TZ", "UTC")
APIFY_CALENDAR_INPUT_JSON = os.getenv("APIFY_CALENDAR_INPUT_JSON", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# ---- event taxonomy --------------------------------------------------------
# First matching type wins, so order matters (specific before generic).
EVENT_KEYWORDS: dict[str, list[str]] = {
    "FOMC": ["fomc", "fed interest rate", "federal funds", "fed funds", "interest rate decision",
             "fed chair", "powell", "fomc minutes", "fomc press conference"],
    "CPI": ["cpi", "consumer price"],
    "PCE": ["pce", "personal consumption expenditures price"],
    "EMPLOYMENT": ["nonfarm", "non-farm", "non farm", "payroll", "unemployment rate",
                   "jobless claims", "average hourly earnings", "adp employment", "jolts"],
    "GDP": ["gdp", "gross domestic product"],
}
EVENT_TYPES = list(EVENT_KEYWORDS.keys())

# ---- strategy permutation parameter columns in the trade log ---------------
DEFAULT_PARAM_COLS = ["lookback", "stop_loss", "take_profit"]

# ---- modelling defaults ----------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
TOP_K = 5
MIN_TRAIN_DAYS = 40
TEST_DAYS = 5
EMBARGO_DAYS = 1
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for p in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, DB_DIR, REPORTS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)
