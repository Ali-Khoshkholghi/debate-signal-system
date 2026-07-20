"""Regression test for a bug found in a live NVDA run: Bull's claims
"Japan's robotics leaders joined the Cosmos Coalition" and "H200 AI chips
are being shipped to China" are specific reported events, not assertions
about how the market feels about NVDA. Both were classified
claim_type="news_sentiment", anchor-extracted with a fabricated
claimed_sentiment (e.g. "neutral"), then compared against the aggregate
news_result sentiment label (positive) and scored CONTRADICTED. This
flipped Bull from what should have been 7 verified/1 contradicted (0.75
credibility, likely INCONCLUSIVE vs Bear's 0.714) to 7 verified/3
contradicted (0.4 credibility), handing Bear a clear win.

Root cause, confirmed live before any fix (see conversation diagnosis):
- extract_claims' news_sentiment definition was "a claim about news
  coverage or sentiment toward the stock" -- broad enough that a
  headline-derived event claim with no metric and no vague/opinion
  framing had nowhere else to go among the five claim_type buckets, so it
  landed in news_sentiment by elimination, not genuine judgment.
- Unlike ClaimAnchor (fundamentals/price/risk), SentimentAnchor had no
  no_match equivalent -- _extract_sentiment_anchor was structurally forced
  to emit positive/negative/neutral for every claim routed to it, so even
  a claim that isn't really a sentiment assertion got coerced into one and
  compared literally against real data.

Fixed by (1) narrowing extract_claims' news_sentiment definition to
genuine aggregate-sentiment/market-mood assertions, explicitly excluding
specific reported events/actions/deals, and (2) adding anchor_type="value"/
"no_match" to SentimentAnchor (mirroring ClaimAnchor), so a claim that
still lands in news_sentiment despite (1) -- there is genuinely no other
bucket for a metric-free reported event -- can be honestly flagged
unverifiable at the anchor-extraction stage instead of forced into a false
comparison. This second fix is the actual safety net: (1) alone can't
fully prevent misclassification, since the five-category taxonomy has no
category for "specific event, no metric, drawn from a headline."

The @pytest.mark.live tests call the REAL extract_claims/
_extract_sentiment_anchor (real Gemini call, no mocking) with the exact
real claim texts from the live run, plus three more event-shaped claims
(manufacturing partnership, acquisition, new facility) used during
diagnosis to confirm the bug was systemic rather than phrasing-specific,
and one genuine sentiment claim to guard against over-narrowing the
classifier prompt. Skipped by default; run explicitly with:
    python -m pytest tests/test_news_sentiment_event_claim_no_match.py -m live -s

The non-live tests mock call_structured_model (same pattern as
tests/test_verify_claim.py) to deterministically check verify_claim's
handling of the new anchor_type, in particular that its reason string is
distinct from the other "unverifiable" reasons.
"""
import pytest

import judge.judge_agent as judge_agent_mod
from judge.judge_agent import _extract_sentiment_anchor, extract_claims, verify_claim
from judge.schemas import Claim, SentimentAnchor
from tests.conftest import make_agent_result, make_structured_result

# The exact real claim texts from the live NVDA run that produced the false
# CONTRADICTED verdicts.
COSMOS_COALITION_CLAIM = "Japan's robotics leaders joined the Cosmos Coalition"
H200_SHIPMENT_CLAIM = "H200 AI chips are being shipped to China"

# Additional event-shaped claims used during diagnosis to confirm the bug
# wasn't specific to this phrasing -- unrelated domains (manufacturing,
# M&A, facilities), none containing sentiment language.
FOXCONN_PARTNERSHIP_CLAIM = "NVDA announced a new manufacturing partnership with Foxconn to build a chip assembly plant in Arizona."
RUNAI_ACQUISITION_CLAIM = "The company completed its acquisition of Run:ai this quarter."
TOKYO_LAB_CLAIM = "NVDA also opened a new research lab in Tokyo last month."

EVENT_CLAIMS = [
    COSMOS_COALITION_CLAIM,
    H200_SHIPMENT_CLAIM,
    FOXCONN_PARTNERSHIP_CLAIM,
    RUNAI_ACQUISITION_CLAIM,
    TOKYO_LAB_CLAIM,
]

# A genuine aggregate-sentiment claim -- must keep classifying as
# news_sentiment/checkable=true and keep resolving to a real
# anchor_type="value", not get swept into no_match by an over-narrowed
# prompt.
GENUINE_SENTIMENT_CLAIM = "The overall news sentiment around NVDA in recent headlines has been very positive."


def _news_evidence(sentiment="positive", confidence=0.8) -> dict:
    output = {
        "ticker": "NVDA",
        "headlines": ["stub headline"],
        "headline_sentiments": [sentiment],
        "sentiment": sentiment,
        "agreement_ratio": 1.0,
        "source_diversity": 1.0,
        "confidence": confidence,
        "reasoning": "stub reasoning",
    }
    return {"news_result": make_agent_result("news_agent", output=output, guardrail={"passed": True, "reason": "ok", "checks": {}})}


def test_verify_claim_sentiment_no_match_is_unverifiable_not_contradicted(monkeypatch):
    """Mocked version of the fix: pretend the anchor extraction correctly
    returns anchor_type="no_match" (post-fix behavior) for an event claim,
    and confirm verify_claim scores it unverifiable with a distinct
    reason -- never contradicted, regardless of the real aggregate
    sentiment (positive) on file."""
    claim = Claim(claim_type="news_sentiment", claim_text=COSMOS_COALITION_CLAIM, checkable=True)
    evidence = _news_evidence(sentiment="positive")
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=SentimentAnchor(anchor_type="no_match")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "unverifiable"
    assert result.claimed_value is None
    assert result.reason == "not a sentiment claim -- no evidence field exists"


def test_sentiment_no_match_reason_is_distinct_from_other_unverifiable_reasons(monkeypatch):
    """The 'not actually a sentiment claim' reason must stay separable from
    the pre-existing 'could not extract a sentiment anchor' and low-
    confidence reasons, mirroring how no_match/named_but_no_value stay
    separable on the numeric-anchor side (see
    tests/test_anchor_qualitative_no_value.py)."""
    claim = Claim(claim_type="news_sentiment", claim_text=COSMOS_COALITION_CLAIM, checkable=True)
    evidence = _news_evidence(sentiment="positive")

    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=SentimentAnchor(anchor_type="no_match")),
    )
    no_match_result = verify_claim(claim, evidence)

    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=None, error="schema validation failed: ..."),
    )
    could_not_extract_result = verify_claim(claim, evidence)

    assert no_match_result.verdict == could_not_extract_result.verdict == "unverifiable"
    assert no_match_result.reason != could_not_extract_result.reason
    assert no_match_result.reason == "not a sentiment claim -- no evidence field exists"


def test_verify_claim_genuine_sentiment_claim_still_verifies_normally(monkeypatch):
    """Guard against over-fixing: a genuine sentiment anchor (anchor_type=
    "value") must still flow through the existing verified/contradicted
    comparison against real aggregate sentiment, unaffected by the new
    no_match branch."""
    claim = Claim(claim_type="news_sentiment", claim_text=GENUINE_SENTIMENT_CLAIM, checkable=True)
    evidence = _news_evidence(sentiment="positive")
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=SentimentAnchor(anchor_type="value", claimed_sentiment="positive")),
    )

    result = verify_claim(claim, evidence)

    assert result.verdict == "verified"
    assert result.claimed_value == "positive"


@pytest.mark.live
@pytest.mark.parametrize("claim_text", EVENT_CLAIMS)
def test_extract_sentiment_anchor_event_claim_is_no_match_not_fabricated(claim_text):
    """Real call to Gemini/Groq -- for each of the five event-shaped claims
    (the two from the live run, plus three more spanning unrelated
    domains), _extract_sentiment_anchor must recognize there's no genuine
    aggregate-sentiment assertion here and return anchor_type="no_match",
    not fabricate a positive/negative/neutral label. Forces claim_type=
    "news_sentiment" directly (bypassing extract_claims' own classification,
    which may or may not still route these here) to isolate this step as
    the safety net regardless of upstream classification."""
    claim = Claim(claim_type="news_sentiment", claim_text=claim_text, checkable=True)

    anchor = _extract_sentiment_anchor(claim)

    assert anchor is not None
    assert anchor.anchor_type == "no_match", (
        f"regression: model fabricated claimed_sentiment={anchor.claimed_sentiment!r} "
        "for a specific reported event with no sentiment framing"
    )
    assert anchor.claimed_sentiment is None


@pytest.mark.live
def test_extract_sentiment_anchor_genuine_sentiment_claim_still_returns_value():
    """Companion to the no_match tests above: a real aggregate-sentiment
    assertion must still resolve to anchor_type="value" with a real
    claimed_sentiment, proving the no_match branch didn't over-narrow this
    step into treating every claim as unverifiable."""
    claim = Claim(claim_type="news_sentiment", claim_text=GENUINE_SENTIMENT_CLAIM, checkable=True)

    anchor = _extract_sentiment_anchor(claim)

    assert anchor is not None
    assert anchor.anchor_type == "value"
    assert anchor.claimed_sentiment == "positive"


@pytest.mark.live
def test_live_event_claims_never_score_contradicted_end_to_end():
    """End-to-end regression proof, closest to the actual live-run bug:
    run the REAL extract_claims classifier on a transcript containing the
    two exact real claims from the NVDA run, then verify_claim each
    resulting claim against real-shaped news evidence whose aggregate
    sentiment is "positive" -- the same real_sentiment that caused the
    original false CONTRADICTED verdicts. Regardless of which of the two
    fixes catches it (a tighter classifier prompt, or the anchor-level
    no_match safety net), no claim derived from these two sentences may
    ever come back "contradicted": these are specific reported events, and
    the aggregate sentiment label was never wrong about anything Bull
    actually asserted."""
    transcript = (
        "Bull: Japan's robotics leaders joined the Cosmos Coalition. "
        "Separately, H200 AI chips are being shipped to China."
    )
    claims = extract_claims(transcript)
    assert claims, "extract_claims returned nothing for a two-sentence transcript"

    evidence = _news_evidence(sentiment="positive")
    for claim in claims:
        if not claim.checkable:
            continue
        result = verify_claim(claim, evidence)
        assert result.verdict != "contradicted", (
            f"regression: claim {claim.claim_text!r} (claim_type={claim.claim_type!r}) "
            f"scored CONTRADICTED -- reason: {result.reason!r}"
        )
