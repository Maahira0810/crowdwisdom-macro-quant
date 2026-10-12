import json

import pandas as pd
import pytest

from src import llm_summary as L
from src.features.macro_features import build_daily_macro
from src.ingestion.apify_calendar import normalize_calendar
from src.ingestion.common import normalize_news


@pytest.fixture()
def data():
    events = normalize_calendar([
        {"date": "2026-09-11", "time": "08:30", "event": "CPI (YoY) (Aug)", "country": "US", "actual": "3.4%", "forecast": "3.4%", "previous": "3.4%"},
        {"date": "2026-10-02", "time": "08:30", "event": "Nonfarm Payrolls (Sep)", "country": "US", "actual": "29K", "forecast": "89K", "previous": "133K"},
    ], source_tz="America/New_York")
    news = normalize_news([
        {"title": "Hot CPI fuels rate hike bets", "url": "https://www.reuters.com/a", "content": "Stocks slump on recession fears",
         "published_date": "2026-10-05T12:00:00Z", "score": 0.9},
        {"title": "Cooling inflation boosts rate cut hopes", "url": "https://cnbc.com/b", "content": "equities rally",
         "published_date": "2026-10-06T12:00:00Z", "score": 0.5},
    ], "test")
    as_of = pd.Timestamp("2026-10-09")
    daily = build_daily_macro(events, news, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-20"))
    return events, news, daily, as_of


def test_context_is_deterministic_and_has_expected_facts(data):
    events, news, daily, as_of = data
    a, b = L.build_context(events, news, daily, as_of), L.build_context(events, news, daily, as_of)
    assert L.context_hash(a) == L.context_hash(b)
    assert a["as_of"] == "2026-10-09" and a["proxy_pnl"] is True
    assert {r["type"] for r in a["latest_releases"]} == {"CPI", "EMPLOYMENT"}
    assert a["recent_headlines"][0]["outlet"] == "reuters.com"       # highest score first, 'www.' stripped
    assert a["coverage"]["calendar_start"] == "2026-09-11"


def test_generate_with_stub_llm_and_prompt_contains_context(data):
    ctx = L.build_context(*data[:3], data[3])
    seen = {}

    def fake(system, user):
        seen["system"], seen["user"] = system, user
        return "**Headline view.** Payrolls rose 29000."
    text, meta = L.generate_outlook(ctx, complete=fake)
    assert text.startswith("**Headline view.**") and meta["context_sha256"] == L.context_hash(ctx)
    assert "untrusted" in seen["system"] and "Nonfarm Payrolls" in seen["user"]


def test_no_key_raises_clear_error(monkeypatch, data):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="No LLM key"):
        L.generate_outlook(L.build_context(*data[:3], data[3]))


def test_unverified_numbers_flags_invented_figures(data):
    ctx = L.build_context(*data[:3], data[3])
    ok = "CPI was 3.4% YoY and payrolls rose 29,000 versus 89,000 on 2 October 2026."
    bad = "Payrolls rose 29,000 and the 10-year yield hit 7.77%."
    assert L.unverified_numbers(ok, ctx) == []
    assert L.unverified_numbers(bad, ctx) == ["7.77"]


def test_cache_valid_then_stale_when_data_changes(tmp_path, data):
    ctx = L.build_context(*data[:3], data[3])
    p = tmp_path / "n.md"
    L.write_narrative(p, "hello", {"model": "m", "context_sha256": L.context_hash(ctx)})
    assert L.read_narrative(p, ctx)[2] == "ok"
    changed = json.loads(json.dumps(ctx)); changed["as_of"] = "2026-10-10"
    assert L.read_narrative(p, changed)[2] == "stale"
    assert L.read_narrative(tmp_path / "none.md", ctx)[2] == "missing"
