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

AnchorType = Literal["value", "relational", "no_match"]
ComparisonOperator = Literal["above", "below"]


class ClaimAnchor(BaseModel):
    """Shape for fundamentals/price_technical/risk_technical claims.
    Exactly one of the two field groups below is populated, selected by
    anchor_type -- "value" claims (e.g. "P/E ratio of 23") populate
    metric/claimed_value; "relational" claims (e.g. "trading above the
    50-day SMA") populate metric_a/operator/metric_b. "no_match" means the
    claim references a concept with no corresponding evidence field at all
    (e.g. "momentum") -- neither group is populated in that case."""
    anchor_type: AnchorType
    metric: str | None = None
    claimed_value: str | None = None
    metric_a: str | None = None
    operator: ComparisonOperator | None = None
    metric_b: str | None = None


class SentimentAnchor(BaseModel):
    """Shape for news_sentiment claims -- always a single categorical
    assertion, no relational/no_match variant needed since a news_sentiment
    claim is by construction about the one aggregate sentiment field."""
    claimed_sentiment: Literal["positive", "negative", "neutral"]


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
