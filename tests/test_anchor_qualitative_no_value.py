"""Regression test for a bug found in a live debate run: a Bull round-2
claim -- "AAPL's strong fundamentals, including a high P/E ratio and
significant revenue growth" -- named a real metric (P/E ratio) but never
asserted an actual number about it, only the qualitative word "high".
_extract_numeric_anchor nonetheless returned anchor_type="value",
metric="pe_ratio", claimed_value="high", which _values_match() then
compared against the real pe_ratio (38.16) and scored "contradicted" --
a single parsing artifact that flipped the whole debate's final ruling
from Bull-favorable to Bear winning.

Root cause: the anchor schema's "no_match" branch is defined narrowly as
"the field doesn't exist at all" (see judge/schemas.py), so it was never a
truthful option once a real metric name ("P/E ratio") was mentioned -- the
model was left inventing a placeholder claimed_value to satisfy the
"value" shape's requirements. Fixed by adding a fourth anchor_type,
"named_but_no_value", for exactly this case: metric recognized, no real
number/category stated.

The @pytest.mark.live tests call the REAL _extract_numeric_anchor (real
Gemini call, no mocking) with the exact real claim_text from the live run,
plus a few more qualitative-descriptor claims, to prove the fix
generalizes rather than patching this one wording. Skipped by default;
run explicitly with:
    python -m pytest tests/test_anchor_qualitative_no_value.py -m live -s

The non-live tests below mock call_structured_model (same pattern as
tests/test_verify_claim.py) to deterministically check verify_claim's
handling of the new anchor_type, in particular that its reason string is
distinct from "no_match"'s -- these two unverifiable causes are scored
identically today but must stay separable if this data is ever analyzed
later.
"""
import pytest

import judge.judge_agent as judge_agent_mod
from judge.judge_agent import (
    FUNDAMENTALS_ALL_FIELDS,
    RISK_TECHNICAL_FIELDS,
    _extract_numeric_anchor,
    verify_claim,
)
from judge.schemas import Claim, ClaimAnchor
from tests.conftest import make_agent_result, make_structured_result


@pytest.mark.live
def test_extract_numeric_anchor_high_pe_ratio_is_not_a_fabricated_value():
    """The exact real claim_text from the live run that caused the false
    'contradicted' verdict. Must NOT come back as anchor_type='value' with
    a non-numeric placeholder like 'high' -- that's the bug. Must instead
    recognize pe_ratio as a real, named metric with no actual number
    stated."""
    claim = Claim(
        claim_type="fundamentals",
        claim_text="AAPL's strong fundamentals, including a high P/E ratio and significant revenue growth",
        checkable=True,
    )

    anchor = _extract_numeric_anchor(claim, FUNDAMENTALS_ALL_FIELDS)

    assert anchor is not None
    assert anchor.anchor_type != "value", (
        f"regression: model fabricated a claimed_value ({anchor.claimed_value!r}) for "
        "vague qualitative language instead of recognizing no real number was stated"
    )
    assert anchor.anchor_type == "named_but_no_value"
    assert anchor.metric == "pe_ratio"
    assert anchor.claimed_value is None


@pytest.mark.live
def test_extract_numeric_anchor_significant_revenue_growth_is_not_a_fabricated_value():
    claim = Claim(claim_type="fundamentals", claim_text="AAPL shows significant revenue growth", checkable=True)

    anchor = _extract_numeric_anchor(claim, FUNDAMENTALS_ALL_FIELDS)

    assert anchor is not None
    assert anchor.anchor_type != "value", f"fabricated claimed_value: {anchor.claimed_value!r}"
    assert anchor.anchor_type == "named_but_no_value"
    assert anchor.metric == "revenue_growth_yoy"


@pytest.mark.live
def test_extract_numeric_anchor_strong_profit_margins_is_not_a_fabricated_value():
    claim = Claim(claim_type="fundamentals", claim_text="AAPL has strong profit margins", checkable=True)

    anchor = _extract_numeric_anchor(claim, FUNDAMENTALS_ALL_FIELDS)

    assert anchor is not None
    assert anchor.anchor_type != "value", f"fabricated claimed_value: {anchor.claimed_value!r}"
    assert anchor.anchor_type == "named_but_no_value"
    assert anchor.metric == "profit_margin"


@pytest.mark.live
def test_extract_numeric_anchor_substantial_volatility_is_not_a_fabricated_value():
    """No period is stated ('substantial volatility', not 'over 20 days' or
    'annualized'), so either volatility field is a legitimate match --
    RISK_TECHNICAL_FIELDS carries both volatility_20d and
    volatility_annualized (see judge_agent.py's field-inventory fix), and
    this test only pins down that no number gets fabricated, not which of
    the two equally-valid volatility fields the model names."""
    claim = Claim(claim_type="risk_technical", claim_text="This stock shows substantial volatility", checkable=True)

    anchor = _extract_numeric_anchor(claim, RISK_TECHNICAL_FIELDS)

    assert anchor is not None
    assert anchor.anchor_type != "value", f"fabricated claimed_value: {anchor.claimed_value!r}"
    assert anchor.anchor_type == "named_but_no_value"
    assert anchor.metric in ("volatility_20d", "volatility_annualized")


@pytest.mark.live
def test_extract_numeric_anchor_medium_risk_name_is_recognized_as_categorical_value():
    """Regression test for a related bug found in review: risk_level is a
    CATEGORICAL field (a small fixed set of real values -- low/medium/high,
    see judge_agent.py's CATEGORICAL_FIELDS) but shared the same shape-4
    'vague qualitative language' guidance written for open-ended numeric
    fields like pe_ratio. Natural adjectival phrasing ('medium-risk name')
    was misclassified as named_but_no_value even though 'medium' names a
    real, exact category value, not a vague descriptor -- the mirror-image
    of this file's original bug (there, vague language was wrongly treated
    as a real value; here, a real categorical value is wrongly treated as
    vague language). Confirmed live and reproducible before the fix; the
    literal phrase 'a medium risk level' already worked, so this uses the
    exact adjectival phrasing that actually reproduced the bug."""
    claim = Claim(
        claim_type="risk_technical",
        claim_text="This is a medium-risk name given its volatility profile.",
        checkable=True,
    )

    anchor = _extract_numeric_anchor(claim, RISK_TECHNICAL_FIELDS)

    assert anchor is not None
    assert anchor.anchor_type == "value", (
        f"regression: a real categorical value was misclassified as vague qualitative "
        f"language (anchor_type={anchor.anchor_type!r})"
    )
    assert anchor.metric == "risk_level"
    assert anchor.claimed_value is not None and anchor.claimed_value.strip().lower() == "medium"


def _fundamentals_evidence(pe_ratio=38.16) -> dict:
    output = {
        "ticker": "AAPL",
        "quote_type": "EQUITY",
        "expected_fields": ["pe_ratio", "revenue_growth_yoy", "eps", "profit_margin"],
        "pe_ratio": pe_ratio,
        "revenue_growth_yoy": 5.2,
        "eps": 6.1,
        "profit_margin": 25.3,
        "confidence": 0.9,
        "reasoning": "stub reasoning",
    }
    return {"fundamentals_result": make_agent_result("fundamentals_agent", output=output, guardrail={"passed": True, "reason": "ok", "checks": {}})}


def test_verify_claim_named_but_no_value_is_unverifiable_not_contradicted(monkeypatch):
    """Mocked version of the exact live-run bug: pretend the anchor
    extraction correctly returns named_but_no_value/pe_ratio (post-fix
    behavior) and confirm verify_claim scores it unverifiable, never
    contradicted -- regardless of the real pe_ratio (38.16) on file."""
    claim = Claim(
        claim_type="fundamentals",
        claim_text="AAPL's strong fundamentals, including a high P/E ratio and significant revenue growth",
        checkable=True,
    )
    evidence = _fundamentals_evidence(pe_ratio=38.16)
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="named_but_no_value", metric="pe_ratio")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "unverifiable"
    assert result.claimed_value is None
    assert result.reason == "metric recognized, no number stated"


def test_named_but_no_value_reason_is_distinct_from_no_match_reason(monkeypatch):
    """The two 'nothing to check' causes -- 'no field exists at all'
    (no_match) vs. 'field exists but no number was stated'
    (named_but_no_value) -- must keep distinguishable reason strings, even
    though both score as unverifiable today, so this data stays separable
    if it's ever analyzed later."""
    claim = Claim(claim_type="fundamentals", claim_text="doesn't matter, model output is mocked", checkable=True)
    evidence = _fundamentals_evidence()

    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="no_match")),
    )
    no_match_result = verify_claim(claim, evidence)

    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=ClaimAnchor(anchor_type="named_but_no_value", metric="pe_ratio")),
    )
    named_but_no_value_result = verify_claim(claim, evidence)

    assert no_match_result.verdict == named_but_no_value_result.verdict == "unverifiable"
    assert no_match_result.reason != named_but_no_value_result.reason
    assert no_match_result.reason == "no evidence field exists for this claim's metric"
    assert named_but_no_value_result.reason == "metric recognized, no number stated"
