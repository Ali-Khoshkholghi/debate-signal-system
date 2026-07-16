"""verify_claim(): checks a checkable claim's factual anchor against a
frozen evidence snapshot, never the interpretation built on top of it (see
judge/judge_agent.py's module docstring and design decision log).

Mocked tests build hand-written evidence dicts matching stock-signal-
system's real per-agent output shapes (confirmed by reading
agents/fundamentals_agent.py, price_agent.py, risk_agent.py, news_agent.py
directly -- not assumed) and stub call_structured_model for the anchor-
extraction call, mirroring tests/test_judge_agent.py's pattern. The paired
@pytest.mark.live test runs gather_evidence against a real ticker and
verifies a claim built from that real data, proving the anchor-extraction
model call actually works end to end.
"""
import pytest

import judge.judge_agent as judge_agent_mod
from judge.judge_agent import (
    FUNDAMENTALS_ETF_FIELDS,
    FUNDAMENTALS_EQUITY_FIELDS,
    verify_claim,
)
from judge.schemas import Claim, ClaimAnchor, SentimentAnchor
from tests.conftest import make_agent_result, make_structured_result


def _fundamentals_evidence(pe_ratio=23.4, quote_type="EQUITY", confidence=0.9) -> dict:
    expected_fields = FUNDAMENTALS_ETF_FIELDS if quote_type == "ETF" else FUNDAMENTALS_EQUITY_FIELDS
    output = {
        "ticker": "AAPL",
        "quote_type": quote_type,
        "expected_fields": expected_fields,
        "pe_ratio": pe_ratio if quote_type == "EQUITY" else None,
        "revenue_growth_yoy": 5.2,
        "eps": 6.1,
        "profit_margin": 25.3,
        "expense_ratio": None if quote_type == "EQUITY" else 0.03,
        "nav_price": None if quote_type == "EQUITY" else 450.0,
        "total_assets": None if quote_type == "EQUITY" else 400_000_000_000,
        "category": None if quote_type == "EQUITY" else "Large Blend",
        "dividend_yield": None if quote_type == "EQUITY" else 1.2,
        "confidence": confidence,
        "reasoning": "stub reasoning",
    }
    return {"fundamentals_result": make_agent_result("fundamentals_agent", output=output, guardrail={"passed": True, "reason": "ok", "checks": {}})}


def _price_evidence(current_price=210.0, sma_50=200.0) -> dict:
    output = {
        "ticker": "AAPL",
        "current_price": current_price,
        "change_pct": 1.1,
        "sma_20": 205.0,
        "sma_50": sma_50,
        "volatility_20d": 1.5,
        "trend": "up",
        "data_as_of": "2026-07-14",
        "confidence": 0.9,
        "reasoning": "stub reasoning",
    }
    return {"price_result": make_agent_result("price_agent", output=output, guardrail={"passed": True, "reason": "ok", "checks": {}})}


def _risk_evidence(volatility_20d=2.3, volatility_annualized=36.51, risk_level="high", confidence=0.9) -> dict:
    output = {
        "ticker": "AAPL",
        "volatility_20d": volatility_20d,
        "volatility_annualized": volatility_annualized,
        "risk_level": risk_level,
        "confidence": confidence,
        "upstream_price_ok": True,
        "reasoning": "stub reasoning",
    }
    return {"risk_result": make_agent_result("risk_agent", output=output, guardrail={"passed": True, "reason": "ok", "checks": {}})}


def _news_evidence(sentiment="positive", confidence=0.8) -> dict:
    output = {
        "ticker": "AAPL",
        "headlines": ["stub headline"],
        "headline_sentiments": [sentiment],
        "sentiment": sentiment,
        "agreement_ratio": 1.0,
        "source_diversity": 1.0,
        "confidence": confidence,
        "reasoning": "stub reasoning",
    }
    return {"news_result": make_agent_result("news_agent", output=output, guardrail={"passed": True, "reason": "ok", "checks": {}})}


def test_verify_claim_verified_numeric_fundamentals_claim(monkeypatch):
    """P/E claim of 23 against a real pe_ratio of 23.4 -- within the 10%
    relative tolerance, so the factual anchor is verified. Does NOT touch
    whether 'fairly valued' is a reasonable read of that number."""
    claim = Claim(claim_type="fundamentals", claim_text="The P/E ratio of 23 shows this is fairly valued.", checkable=True)
    evidence = _fundamentals_evidence(pe_ratio=23.4)
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="value", metric="pe_ratio", claimed_value="23")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.evidence_agent == "fundamentals_result"
    assert result.real_value == "23.4"


def test_verify_claim_contradicted_numeric_fundamentals_claim(monkeypatch):
    """Same claimed P/E of 23, but the real value (48.0) is far outside
    tolerance -- contradicted, regardless of whether 'fairly valued' would
    still be a reasonable characterization of 48.0."""
    claim = Claim(claim_type="fundamentals", claim_text="The P/E ratio of 23 shows this is fairly valued.", checkable=True)
    evidence = _fundamentals_evidence(pe_ratio=48.0)
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="value", metric="pe_ratio", claimed_value="23")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "contradicted"
    assert result.real_value == "48.0"


def test_verify_claim_relational_price_technical_claim(monkeypatch):
    """'Trading above the 50-day SMA' is a relationship between two
    metrics, not a single claimed value -- current_price=210 > sma_50=200
    holds, so verified."""
    claim = Claim(claim_type="price_technical", claim_text="The stock is trading above its 50-day SMA.", checkable=True)
    evidence = _price_evidence(current_price=210.0, sma_50=200.0)
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(
            data=ClaimAnchor(anchor_type="relational", metric_a="current_price", operator="above", metric_b="sma_50")
        ),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.evidence_agent == "price_result"


def test_verify_claim_risk_technical_volatility_20d_claim_matches_period_specific_field(monkeypatch):
    """Regression test for a real live-run bug: Bear claimed 'the volatility
    of 2.3 over 20 days'. risk_result's own volatility_20d (2.3) matches
    exactly, but until RISK_TECHNICAL_FIELDS listed volatility_20d as a
    valid anchor field, the anchor-extraction prompt only ever offered
    volatility_annualized as a volatility-shaped option, forcing a
    same-concept-wrong-period match against 36.51 and a false CONTRADICTED
    verdict.

    The stub below mimics that forced-choice behavior instead of hand-
    picking an answer: it can only return a metric that's actually present
    in the prompt's 'Valid metric names' list, exactly as a real model is
    constrained -- so this test is red before the allow-list fix (forced
    onto volatility_annualized) and green after (volatility_20d becomes a
    legal choice), without the test ever assuming which field is correct."""
    claim = Claim(claim_type="risk_technical", claim_text="the volatility of 2.3 over 20 days", checkable=True)
    evidence = _risk_evidence(volatility_20d=2.3, volatility_annualized=36.51)

    def _period_aware_stub(system_prompt, user_prompt, schema, extra_check=None):
        valid_line = next(line for line in system_prompt.splitlines() if line.startswith("Valid metric names"))
        valid_fields = [f.strip() for f in valid_line.split(":", 1)[1].rstrip(".").split(",")]
        metric = "volatility_20d" if "volatility_20d" in valid_fields else "volatility_annualized"
        return make_structured_result(data=ClaimAnchor(anchor_type="value", metric=metric, claimed_value="2.3"))

    monkeypatch.setattr(judge_agent_mod, "call_structured_model", _period_aware_stub)

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.real_value == "2.3"


def test_verify_claim_price_technical_sma20_claim_matches_period_specific_field(monkeypatch):
    """Defensive coverage for the milder version of the same risk class:
    sma_20 and sma_50 are both already in PRICE_TECHNICAL_FIELDS, so this
    isn't broken today (unlike the risk_technical case above, which was
    missing its field entirely). This locks in that a claim naming one
    period ('20-day') resolves to that period's field (sma_20=205.0), not
    the other period's field (sma_50=200.0) -- catching any future
    regression in the lookup, even though it doesn't exercise the model
    prompt itself (see test_extract_numeric_anchor_prompt_includes_
    period_matching_instruction below for that)."""
    claim = Claim(claim_type="price_technical", claim_text="the stock's 20-day moving average is around 205", checkable=True)
    evidence = _price_evidence(current_price=210.0, sma_50=200.0)  # sma_20 defaults to 205.0

    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="value", metric="sma_20", claimed_value="205")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.real_value == "205.0"


def test_extract_numeric_anchor_prompt_includes_period_matching_instruction(monkeypatch):
    """Locks in that the anchor-extraction system prompt actually carries
    explicit period-matching guidance (defense-in-depth for the
    volatility_20d/volatility_annualized and sma_20/sma_50 collision risk),
    not just that the allow-lists happen to be complete today."""
    captured = {}

    def _capture(system_prompt, user_prompt, schema, extra_check=None):
        captured["system_prompt"] = system_prompt
        return make_structured_result(data=ClaimAnchor(anchor_type="value", metric="volatility_annualized", claimed_value="36.51"))

    monkeypatch.setattr(judge_agent_mod, "call_structured_model", _capture)

    judge_agent_mod._extract_numeric_anchor(
        Claim(claim_type="risk_technical", claim_text="volatility is 36.51 annualized", checkable=True),
        judge_agent_mod.RISK_TECHNICAL_FIELDS,
    )

    assert "period" in captured["system_prompt"].lower()


def test_verify_claim_risk_technical_categorical_risk_level_claim_is_verified(monkeypatch):
    """Supplementary mocked test locking in the downstream comparison side:
    once anchor extraction correctly returns a categorical value anchor for
    risk_level (post-fix behavior for adjectival phrasing like 'medium-risk
    name'), verify_claim must resolve it via _values_match's CATEGORICAL_FIELDS
    string-comparison branch and score it verified against the real
    risk_level -- this was already correct once you got a 'value' anchor;
    the bug lived entirely in the extraction step, not here."""
    claim = Claim(claim_type="risk_technical", claim_text="This is a medium-risk name given its volatility profile.", checkable=True)
    evidence = _risk_evidence(risk_level="medium")
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="value", metric="risk_level", claimed_value="medium")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.real_value == "medium"


def test_verify_claim_etf_fundamentals_mismatch_is_unverifiable_not_contradicted(monkeypatch):
    """A P/E-ratio claim checked against an ETF's real output: ETFs don't
    report pe_ratio at all, so this is unverifiable (asset-type mismatch),
    explicitly distinct from contradicted."""
    claim = Claim(claim_type="fundamentals", claim_text="The P/E ratio of 23 shows this is fairly valued.", checkable=True)
    evidence = _fundamentals_evidence(quote_type="ETF")
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="value", metric="pe_ratio", claimed_value="23")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "unverifiable"
    assert "not applicable to this asset type" in result.reason


def test_verify_claim_no_matching_evidence_field_is_unverifiable(monkeypatch):
    """'Incredible momentum' has no corresponding field anywhere in
    price_agent's output -- the anchor extraction call signals no_match,
    and verify_claim must not silently drop or contradict this claim."""
    claim = Claim(claim_type="price_technical", claim_text="This stock has incredible momentum.", checkable=True)
    evidence = _price_evidence()
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="no_match")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "unverifiable"
    assert result.reason == "no evidence field exists for this claim's metric"


def test_verify_claim_non_checkable_claim_short_circuits_without_calling_anything(monkeypatch):
    """checkable=False claims must never reach routing or call the model --
    verify_claim should short-circuit immediately."""
    claim = Claim(claim_type="rhetorical", claim_text="Investors should be excited about this stock.", checkable=False)

    def _fail_if_called(*a, **k):
        raise AssertionError("verify_claim must not call the model for a non-checkable claim")

    monkeypatch.setattr(judge_agent_mod, "call_structured_model", _fail_if_called)

    result = verify_claim(claim, evidence={})

    assert result.verdict == "unverifiable"
    assert result.reason == "claim marked non-checkable"


def test_verify_claim_low_confidence_news_is_unverifiable_not_contradicted(monkeypatch):
    """News confidence below CONFIDENCE_THRESHOLD (0.6) must resolve to
    unverifiable with its own distinct reason, not be treated as
    contradicting evidence -- thin evidence isn't grounds to contradict."""
    claim = Claim(claim_type="news_sentiment", claim_text="News sentiment is very positive on this stock.", checkable=True)
    evidence = _news_evidence(sentiment="positive", confidence=0.4)

    def _fail_if_called(*a, **k):
        raise AssertionError("verify_claim must gate on low confidence before calling the anchor-extraction model")

    monkeypatch.setattr(judge_agent_mod, "call_structured_model", _fail_if_called)

    result = verify_claim(claim, evidence)

    assert result.verdict == "unverifiable"
    assert "confidence" in result.reason
    assert "0.4" in result.reason


def test_verify_claim_verified_news_sentiment_claim(monkeypatch):
    claim = Claim(claim_type="news_sentiment", claim_text="News sentiment is very positive on this stock.", checkable=True)
    evidence = _news_evidence(sentiment="positive", confidence=0.9)
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=SentimentAnchor(claimed_sentiment="positive")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.evidence_agent == "news_result"


@pytest.mark.live
def test_verify_claim_against_live_evidence_for_a_real_ticker(tmp_path, monkeypatch):
    """Real end-to-end run: gather real evidence for a real ticker, then
    verify a claim built from that same real data. Since the claim asserts
    the exact real pe_ratio, this should verify -- proving the live
    anchor-extraction call actually parses the number back out correctly,
    not just the schema plumbing. Skipped by default; run explicitly with:
        python -m pytest tests/ -m live
    """
    import config as p1_config
    from evidence.tools import ensure_run, gather_evidence

    monkeypatch.setattr(p1_config, "DB_PATH", str(tmp_path / "live_verify_test.db"))

    ticker = "AAPL"
    run_id = f"live-verify-{ticker}"
    ensure_run(run_id, ticker)
    evidence = gather_evidence(ticker, run_id)

    fundamentals_output = (evidence.get("fundamentals_result") or {}).get("output") or {}
    pe_ratio = fundamentals_output.get("pe_ratio")
    if pe_ratio is None:
        pytest.skip("live fundamentals data had no pe_ratio to build a claim from")

    claim = Claim(claim_type="fundamentals", claim_text=f"The P/E ratio is approximately {pe_ratio}.", checkable=True)

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.reason


@pytest.mark.live
def test_verify_claim_risk_technical_volatility_20d_against_live_evidence(tmp_path, monkeypatch):
    """Real end-to-end regression check for the live-run bug: builds a claim
    phrased exactly like the real Bear claim ('the volatility of X over 20
    days') from a real risk_result's own volatility_20d, then verifies it
    through the real anchor-extraction model call. Proves the fix holds
    against a live model, not just the constrained stub above. Skipped by
    default; run explicitly with:
        python -m pytest tests/ -m live
    """
    import config as p1_config
    from evidence.tools import ensure_run, gather_evidence

    monkeypatch.setattr(p1_config, "DB_PATH", str(tmp_path / "live_verify_risk_test.db"))

    ticker = "AAPL"
    run_id = f"live-verify-risk-{ticker}"
    ensure_run(run_id, ticker)
    evidence = gather_evidence(ticker, run_id)

    risk_output = (evidence.get("risk_result") or {}).get("output") or {}
    volatility_20d = risk_output.get("volatility_20d")
    if volatility_20d is None:
        pytest.skip("live risk data had no volatility_20d to build a claim from")

    claim = Claim(claim_type="risk_technical", claim_text=f"the volatility of {volatility_20d} over 20 days", checkable=True)

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.real_value == str(volatility_20d)


@pytest.mark.live
def test_verify_claim_risk_technical_categorical_risk_level_against_live_evidence(tmp_path, monkeypatch):
    """Real end-to-end regression check for the categorical-phrasing bug:
    builds a claim using the exact adjectival phrasing that reproduced the
    bug ('a {risk_level}-risk name'), from a real risk_result's own
    risk_level, then verifies it through the real anchor-extraction model
    call. Proves the fix holds against a live model, not just the mocked
    comparison-side test above. Skipped by default; run explicitly with:
        python -m pytest tests/ -m live
    """
    import config as p1_config
    from evidence.tools import ensure_run, gather_evidence

    monkeypatch.setattr(p1_config, "DB_PATH", str(tmp_path / "live_verify_risk_level_test.db"))

    ticker = "AAPL"
    run_id = f"live-verify-risk-level-{ticker}"
    ensure_run(run_id, ticker)
    evidence = gather_evidence(ticker, run_id)

    risk_output = (evidence.get("risk_result") or {}).get("output") or {}
    risk_level = risk_output.get("risk_level")
    if risk_level is None:
        pytest.skip("live risk data had no risk_level to build a claim from")

    claim = Claim(claim_type="risk_technical", claim_text=f"This is a {risk_level}-risk name given its volatility profile.", checkable=True)

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.real_value == risk_level
