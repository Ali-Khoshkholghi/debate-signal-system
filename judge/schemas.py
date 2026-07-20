"""Pydantic shapes for the Judge's own LLM calls, validated by
call_structured_model() (see stock-signal-system's models/router.py, reused
here via evidence.tools -- see README.md "Dependency on stock-signal-
system"). Mirrors that project's models/schemas.py convention: shape-only
schemas, not value-level rules -- claim_type is constrained by the Literal
itself (an out-of-set value fails validation and triggers
call_structured_model's retry/fallback, same as any other schema mismatch).
"""
from typing import Literal

from pydantic import BaseModel

ClaimType = Literal["fundamentals", "price_technical", "risk_technical", "news_sentiment", "rhetorical"]


class Claim(BaseModel):
    claim_type: ClaimType
    claim_text: str
    # True for claims the Judge should later attempt to verify against a
    # real evidence agent (price/fundamentals/risk/news); False for
    # rhetorical/subjective claims that have no evidence source to check
    # against. Always False when claim_type == "rhetorical", but kept as
    # its own field (not derived) since the model sets both independently
    # and a mismatch between them is itself a useful validation signal.
    checkable: bool


class ExtractedClaims(BaseModel):
    claims: list[Claim]


# --- Anchor extraction (verify_claim's second, narrower call_structured_model
# call -- see judge/judge_agent.py). Identifies the specific factual anchor
# inside a claim (a metric + asserted value, or a relationship between two
# metrics) so it can be checked against real evidence data, without judging
# whether any interpretation built on top of that anchor is reasonable --
# that stays the Judge's future ruling job, not this step's. ---

AnchorType = Literal["value", "relational", "no_match", "named_but_no_value"]
ComparisonOperator = Literal["above", "below"]


class ClaimAnchor(BaseModel):
    """Shape for fundamentals/price_technical/risk_technical claims.
    Exactly one of the two field groups below is populated, selected by
    anchor_type -- "value" claims (e.g. "P/E ratio of 23") populate
    metric/claimed_value; "relational" claims (e.g. "trading above the
    50-day SMA") populate metric_a/operator/metric_b. "no_match" means the
    claim references a concept with no corresponding evidence field at all
    (e.g. "momentum") -- neither group is populated in that case.
    "named_but_no_value" means a real metric IS named (e.g. "P/E ratio")
    but only a vague qualitative descriptor ("high", "strong",
    "significant") is asserted about it, no actual number/category --
    populates metric only, never claimed_value. Distinct from "no_match"
    (whose field genuinely doesn't exist) so the two stay separable if this
    data is ever analyzed later, even though verify_claim scores both as
    unverifiable today."""
    anchor_type: AnchorType
    metric: str | None = None
    claimed_value: str | None = None
    metric_a: str | None = None
    operator: ComparisonOperator | None = None
    metric_b: str | None = None


SentimentAnchorType = Literal["value", "no_match"]


class SentimentAnchor(BaseModel):
    """Shape for news_sentiment claims. anchor_type="value" is a genuine
    aggregate-sentiment/market-mood assertion (e.g. "the recent headlines
    are very positive") -- populates claimed_sentiment. anchor_type=
    "no_match" means the claim reached this step tagged news_sentiment but
    isn't actually asserting anything about the one aggregate sentiment
    field -- most commonly a specific reported event/action (e.g. "Japan's
    robotics leaders joined the Cosmos Coalition") with no sentiment
    framing at all. Mirrors ClaimAnchor's no_match: the field this claim
    would need to be checked against doesn't exist, so verify_claim must
    score it unverifiable rather than coercing a placeholder positive/
    negative/neutral label and comparing it against real data."""
    anchor_type: SentimentAnchorType
    claimed_sentiment: Literal["positive", "negative", "neutral"] | None = None


class ArgumentTurn(BaseModel):
    """Shape for a Bull/Bear debate turn (see judge/judge_agent.py's
    bull_round1/bear_round1/bull_round2/bear_round2). The two fields are a
    STRUCTURAL separation, not a prompting convention: rebuttal_note is
    meta-commentary about round 1's verification results (e.g. "my P/E
    claim held up"), argument is the actual debate content. A live
    integration test proved wording alone can't reliably keep an extractor
    from grabbing debate-mechanics commentary as if it were a new market
    claim -- identical phrasing was classified inconsistently across two
    speakers. extract_claims() is therefore called ONLY on .argument, never
    on .rebuttal_note, enforced in code (_verify_round), not by asking the
    model to phrase things carefully."""
    rebuttal_note: str | None = None
    argument: str
