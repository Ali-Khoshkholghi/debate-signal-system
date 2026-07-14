"""Smoke tests proving the cross-repo dependency actually works: the five
evidence agents and the router's model-call functions import correctly from
the installed stock-signal-system package, and gather_evidence() can drive
them end to end.
"""
import inspect

import pytest

from evidence.tools import (
    call_model_with_routing,
    call_structured_model,
    ensure_run,
    gather_evidence,
)


def test_reused_agents_and_router_import_and_are_callable():
    import agents.classify_ticker_agent as classify_ticker_agent
    import agents.fundamentals_agent as fundamentals_agent
    import agents.news_agent as news_agent
    import agents.price_agent as price_agent
    import agents.risk_agent as risk_agent

    for fn in (
        classify_ticker_agent.classify_ticker_agent,
        price_agent.price_agent,
        fundamentals_agent.fundamentals_agent,
        risk_agent.risk_agent,
        news_agent.news_agent,
    ):
        assert callable(fn)
        assert list(inspect.signature(fn).parameters) == ["state"]

    assert callable(call_model_with_routing)
    assert callable(call_structured_model)


def test_ensure_run_creates_a_run_row(tmp_path, monkeypatch):
    import config
    import db.store as store

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_debate_signal.db"))

    ensure_run("test-run-1", "AAPL")

    conn = store.get_connection()
    try:
        row = conn.execute("SELECT ticker FROM runs WHERE run_id = ?", ("test-run-1",)).fetchone()
    finally:
        conn.close()
    assert row == ("AAPL",)


@pytest.mark.live
@pytest.mark.parametrize("ticker", ["AAPL"])
def test_gather_evidence_against_live_apis(tmp_path, monkeypatch, ticker):
    """Real end-to-end call through the installed package -- costs tokens,
    network-dependent. Skipped by default; run explicitly with:
        python -m pytest tests/ -m live
    """
    import config

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "live_test.db"))

    run_id = f"live-{ticker}"
    ensure_run(run_id, ticker)
    evidence = gather_evidence(ticker, run_id)

    assert evidence["price_result"]["status"] in {"ok", "error"}
    assert evidence["ticker_classification"] is not None
