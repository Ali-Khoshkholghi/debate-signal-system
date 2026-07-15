"""extract_claims(): the Judge's claim-extraction step, tested in isolation
before any debate/graph orchestration exists (see judge/judge_agent.py
docstring -- highest-risk step, since everything downstream depends on
claims being classified correctly).

Mocked tests exercise the parsing/schema layer deterministically (the LLM
call itself is stubbed, mirroring stock-signal-system's
tests/test_news_agent.py pattern of monkeypatching call_structured_model at
the module level). The paired @pytest.mark.live test hits the real
Gemini/Groq call to prove the model's actual judgment holds, not just the
plumbing -- skipped by default per pytest.ini, run explicitly with:
    python -m pytest tests/ -m live
"""
import pytest

import judge.judge_agent as judge_agent_mod
from judge.judge_agent import extract_claims
from judge.schemas import Claim, ExtractedClaims
from tests.conftest import make_structured_result

# One checkable fundamentals claim, one rhetorical (unverifiable) claim --
# the minimal pair needed to prove the extractor splits them correctly.
TRANSCRIPT = (
    "Bull: The P/E ratio of 23 shows this is fairly valued given the "
    "sector average.\n"
    "Bull: This stock has incredible momentum and investors should be "
    "excited about where it's headed."
)


def test_extract_claims_splits_checkable_fundamentals_from_rhetorical_claim(monkeypatch):
    """Schema/parsing layer only -- the model call is mocked so this is
    deterministic. Proves extract_claims returns Claim objects correctly
    split by claim_type/checkable; the model's real judgment is covered
    separately by the live test below."""
    mocked_claims = ExtractedClaims(claims=[
        Claim(
            claim_type="fundamentals",
            claim_text="The P/E ratio of 23 shows this is fairly valued.",
            checkable=True,
        ),
        Claim(
            claim_type="rhetorical",
            claim_text="This stock has incredible momentum and investors should be excited.",
            checkable=False,
        ),
    ])
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=mocked_claims),
    )

    claims = extract_claims(TRANSCRIPT)

    assert len(claims) == 2
    fundamentals_claims = [c for c in claims if c.claim_type == "fundamentals"]
    rhetorical_claims = [c for c in claims if c.claim_type == "rhetorical"]
    assert len(fundamentals_claims) == 1
    assert fundamentals_claims[0].checkable is True
    assert len(rhetorical_claims) == 1
    assert rhetorical_claims[0].checkable is False


def test_extract_claims_degrades_to_empty_list_when_model_output_unparseable(monkeypatch):
    """If call_structured_model exhausts every retry/fallback attempt
    (data=None), extract_claims must not raise -- it degrades to an empty
    list, same shape as stock-signal-system's own agents degrading on
    unusable model output."""
    monkeypatch.setattr(
        judge_agent_mod, "call_structured_model",
        lambda *a, **k: make_structured_result(data=None, error="schema validation failed: ..."),
    )

    assert extract_claims(TRANSCRIPT) == []


@pytest.mark.live
def test_extract_claims_against_live_model_splits_checkable_from_rhetorical():
    """Real call to Gemini/Groq -- proves the model's actual judgment holds,
    not just the schema plumbing exercised above. Wording in claim_text is
    model-dependent and not asserted on; only the claim_type/checkable
    split is."""
    claims = extract_claims(TRANSCRIPT)

    assert len(claims) >= 2
    claim_types = {c.claim_type for c in claims}
    assert "fundamentals" in claim_types
    assert "rhetorical" in claim_types

    fundamentals_claims = [c for c in claims if c.claim_type == "fundamentals"]
    rhetorical_claims = [c for c in claims if c.claim_type == "rhetorical"]
    assert any(c.checkable is True for c in fundamentals_claims)
    assert all(c.checkable is False for c in rhetorical_claims)
