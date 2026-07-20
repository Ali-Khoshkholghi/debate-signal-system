"""Regression test for a bug found via a live run: Gemini's free-tier rate
limit (confirmed via a real 429 RESOURCE_EXHAUSTED error, 5 req/min) forced
escalation to Groq mid-debate, and Groq's real API rejected every one of
those fallback calls with a confirmed, real error:
    "'messages' must contain the word 'json' in some form, to use
    'response_format' of type 'json_object'."
_extract_numeric_anchor's system prompt was the one structured-output
prompt in this codebase that never said the word "json" anywhere (every
other prompt -- stock-signal-system's own agents, extract_claims,
_extract_sentiment_anchor, the argument-generation prompts -- says
"Respond with ONLY a JSON object..."). So whenever Gemini failed, this
specific anchor extraction had no working fallback at all.

Mocked at models.router's _call_gemini/_call_groq level (NOT at
call_structured_model, which would hide the very prompt-content bug this
guards against) so the REAL system prompt text defined in judge_agent.py
flows through the real retry/escalation logic in models/router.py, checked
against a fake Groq that mimics its real, confirmed constraint.
"""
import models.router as router_mod
from judge.judge_agent import (
    FUNDAMENTALS_ALL_FIELDS,
    _extract_numeric_anchor,
    _extract_sentiment_anchor,
)
from judge.schemas import Claim
from tests.conftest import make_model_result


def _gemini_always_fails(*a, **k):
    raise Exception("gemini: 429 RESOURCE_EXHAUSTED. Quota exceeded, limit: 5 requests/minute")


def _groq_enforcing_real_json_keyword_requirement(system, user, response_json=False):
    """Stands in for models.router._call_groq, mimicking Groq's real,
    confirmed constraint: a response_format=json_object request must
    contain the literal word "json" somewhere in system or user, or the
    real API 400s. Raises exactly the way a real call would if that
    constraint isn't met, so call_model_with_routing's own exception
    handling -- not a stand-in -- is what produces the eventual result."""
    if response_json and "json" not in system.lower() and "json" not in user.lower():
        raise Exception(
            "Error code: 400 - {'error': {'message': \"'messages' must "
            "contain the word 'json' in some form, to use 'response_format' "
            "of type 'json_object'.\"}}"
        )
    return make_model_result(
        model_used="llama-3.3-70b-versatile",
        escalated=True,
        text='{"anchor_type": "value", "metric": "profit_margin", "claimed_value": "27.15"}',
    )


def test_extract_numeric_anchor_survives_gemini_failure_via_groq_fallback(monkeypatch):
    """When Gemini fails (the real scenario from the live run) and the call
    escalates to Groq, the REAL _extract_numeric_anchor system prompt must
    satisfy Groq's real json-keyword requirement -- otherwise the fallback
    that's supposed to be the safety net is silently broken, and every
    claim checked while Gemini is rate-limited comes back "could not
    extract a factual anchor" instead of being verified."""
    monkeypatch.setattr(router_mod, "_call_gemini", _gemini_always_fails)
    monkeypatch.setattr(router_mod, "_call_groq", _groq_enforcing_real_json_keyword_requirement)

    claim = Claim(claim_type="fundamentals", claim_text="27.15% profit margin", checkable=True)
    anchor = _extract_numeric_anchor(claim, FUNDAMENTALS_ALL_FIELDS)

    assert anchor is not None, (
        "Groq fallback failed -- _extract_numeric_anchor's system prompt likely "
        "doesn't contain the literal word 'json', which Groq's real API requires "
        "for response_format=json_object requests"
    )
    assert anchor.anchor_type == "value"
    assert anchor.metric == "profit_margin"


def test_extract_sentiment_anchor_survives_gemini_failure_via_groq_fallback(monkeypatch):
    """Same regression class, checked against _extract_sentiment_anchor's
    prompt too -- it already says "JSON object" today, so this should pass
    even before the fix; included so both anchor helpers are covered by
    the same real-fallback-constraint check going forward."""
    monkeypatch.setattr(router_mod, "_call_gemini", _gemini_always_fails)

    def _groq_sentiment(system, user, response_json=False):
        if response_json and "json" not in system.lower() and "json" not in user.lower():
            raise Exception(
                "Error code: 400 - {'error': {'message': \"'messages' must "
                "contain the word 'json' in some form, to use "
                "'response_format' of type 'json_object'.\"}}"
            )
        return make_model_result(
            model_used="llama-3.3-70b-versatile",
            escalated=True,
            text='{"anchor_type": "value", "claimed_sentiment": "positive"}',
        )

    monkeypatch.setattr(router_mod, "_call_groq", _groq_sentiment)

    claim = Claim(claim_type="news_sentiment", claim_text="positive news sentiment", checkable=True)
    anchor = _extract_sentiment_anchor(claim)

    assert anchor is not None
    assert anchor.claimed_sentiment == "positive"
