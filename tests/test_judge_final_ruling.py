"""judge_final_ruling(): the deterministic credibility-based ruling
computed purely from VerificationResults, no LLM vibe-check over the
transcript (see judge/judge_agent.py's docstring and the project decision
log's "no agent grades its own work" principle).

Entries are hand-built in the same shape _verify_round() produces:
{"round": N, "speaker": "bull"/"bear", "claim": {...}, "verification": {"verdict": ...}}
-- "claim" content doesn't matter for judge_final_ruling, only "speaker"
and "verification"."verdict" do, so it's a stub dict here.
"""
from judge.judge_agent import FINAL_RULING_MARGIN, judge_final_ruling


def _entry(speaker: str, verdict: str, round_num: int = 1) -> dict:
    return {"round": round_num, "speaker": speaker, "claim": {}, "verification": {"verdict": verdict}}


def test_clear_bull_win():
    """Bull: 4 verified, 0 contradicted (credibility 1.0). Bear: 1
    verified, 3 contradicted (credibility -0.5). Gap (1.5) is far beyond
    FINAL_RULING_MARGIN, and bull_credibility > 0 -- Bull wins."""
    entries = (
        [_entry("bull", "verified")] * 4
        + [_entry("bear", "verified")] * 1
        + [_entry("bear", "contradicted")] * 3
    )

    result = judge_final_ruling({"verified_claims": entries})

    ruling = result["final_ruling"]
    assert ruling["winner"] == "Bull"
    assert ruling["bull_credibility"] == 1.0
    assert ruling["bear_credibility"] == -0.5


def test_clear_bear_win():
    """Mirror of the Bull-win case, sides swapped."""
    entries = (
        [_entry("bear", "verified")] * 4
        + [_entry("bull", "verified")] * 1
        + [_entry("bull", "contradicted")] * 3
    )

    result = judge_final_ruling({"verified_claims": entries})

    ruling = result["final_ruling"]
    assert ruling["winner"] == "Bear"
    assert ruling["bear_credibility"] == 1.0
    assert ruling["bull_credibility"] == -0.5


def test_inconclusive_via_low_margin():
    """Bull: 3 verified, 1 contradicted (credibility 0.5). Bear: 7
    verified, 3 contradicted (credibility 0.4). Both sides have a positive,
    similar track record -- the 0.1 gap is under FINAL_RULING_MARGIN
    (0.2), so this must be Inconclusive, not a Bull win by a hair."""
    assert FINAL_RULING_MARGIN == 0.2, "test assumes the documented placeholder margin; update if it's retuned"
    entries = (
        [_entry("bull", "verified")] * 3
        + [_entry("bull", "contradicted")] * 1
        + [_entry("bear", "verified")] * 7
        + [_entry("bear", "contradicted")] * 3
    )

    result = judge_final_ruling({"verified_claims": entries})

    ruling = result["final_ruling"]
    assert ruling["winner"] == "Inconclusive"
    assert ruling["bull_credibility"] == 0.5
    assert ruling["bear_credibility"] == 0.4


def test_inconclusive_when_both_sides_have_zero_checkable_claims():
    """Every claim on both sides is unverifiable/rhetorical -- neither side
    has any verified/contradicted claims to compute a credibility from.
    Both default to 0.0, so this must be Inconclusive."""
    entries = [
        _entry("bull", "unverifiable"),
        _entry("bear", "unverifiable"),
        _entry("bull", "unverifiable"),
    ]

    result = judge_final_ruling({"verified_claims": entries})

    ruling = result["final_ruling"]
    assert ruling["winner"] == "Inconclusive"
    assert ruling["bull_credibility"] == 0.0
    assert ruling["bear_credibility"] == 0.0


def test_silent_side_does_not_win_against_a_side_with_contradicted_claims():
    """Bull made zero checkable claims (credibility 0.0 by definition).
    Bear made 3 claims, all contradicted (credibility -1.0). Naive diff
    math (0.0 - (-1.0) = 1.0 > MARGIN) would call this a Bull win, but Bull
    never demonstrated anything true -- the "win by silence" case this
    design explicitly guards against. Must resolve to Inconclusive, never
    Bull."""
    entries = [
        _entry("bull", "unverifiable"),
        _entry("bear", "contradicted"),
        _entry("bear", "contradicted"),
        _entry("bear", "contradicted"),
    ]

    result = judge_final_ruling({"verified_claims": entries})

    ruling = result["final_ruling"]
    assert ruling["winner"] != "Bull"
    assert ruling["winner"] == "Inconclusive"
    assert ruling["bull_credibility"] == 0.0
    assert ruling["bear_credibility"] == -1.0
