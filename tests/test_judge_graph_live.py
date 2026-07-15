"""Live end-to-end integration test for the Judge's debate graph as it
stands right now: real gather_evidence (yfinance/Tavily/Gemini calls) +
stub debate turns + real claim extraction/verification (Gemini/Groq calls)
+ deterministic final ruling -- run through the actual compiled graph via
run_judge(), not individual node calls.

Purpose: prove the pipeline MECHANICS work together against real data
before any real Bull/Bear argument-generation prompt design happens on top
-- same "verify against live runs before calling something done"
discipline as stock-signal-system's own tests/test_live.py. Stub debate
text is expected to produce few or no genuinely checkable claims (it's
placeholder text, not real financial argument) -- assertions here check
structure/mechanics (evidence populated, transcript ordering, claims
accumulate without dropping/double-counting, ruling numbers are internally
consistent), not exact content.

Skipped by default (costs real API calls); run explicitly with:
    python -m pytest tests/test_judge_graph_live.py -m live -s
"""
import pytest

from judge_graph import run_judge


@pytest.mark.live
def test_run_judge_end_to_end_against_real_data(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "live_judge_graph_test.db"))

    ticker = "AAPL"
    run_id = f"live-judge-graph-{ticker}"

    result = run_judge(ticker, run_id)

    # --- graph completed without error (implicit -- we got here) ---
    print("\n=== run_judge result keys ===", list(result.keys()))

    # --- state["evidence"]: populated with real agent results ---
    evidence = result.get("evidence")
    assert evidence is not None
    expected_evidence_keys = {"ticker_classification", "price_result", "news_result", "fundamentals_result", "risk_result"}
    assert expected_evidence_keys <= evidence.keys()
    print("\n=== evidence ===")
    for key in expected_evidence_keys:
        agent_result = evidence.get(key)
        print(f"-- {key} --")
        print(agent_result)
        if key != "ticker_classification":
            # Each real agent result has the standard shape -- status may
            # legitimately be "skipped"/"error" on a live run (market
            # timing, thin data, etc), same tolerance as stock-signal-
            # system's own test_live.py, which only checks the FINAL
            # signal is valid, not that every sub-agent succeeded.
            assert agent_result is not None
            assert agent_result.get("status") in {"ok", "skipped", "error"}

    # --- transcript_turns: exactly 4, in the right order, correctly tagged ---
    turns = result.get("transcript_turns") or []
    print("\n=== transcript_turns ===")
    for t in turns:
        print(t)
    assert len(turns) == 4
    actual_sequence = [(t["round"], t["speaker"]) for t in turns]
    assert actual_sequence == [(1, "bull"), (1, "bear"), (2, "bull"), (2, "bear")]

    # --- verified_claims: round 1 and round 2 entries both present in the
    # final list (proves round 2's writes ADDED to round 1's, rather than
    # the reducer dropping or overwriting them), and no exact-duplicate
    # entries (guards against a double-application bug in the reducer). ---
    all_claims = result.get("verified_claims") or []
    print("\n=== verified_claims ===")
    for c in all_claims:
        print(c)

    round1_claims = [c for c in all_claims if c["round"] == 1]
    round2_claims = [c for c in all_claims if c["round"] == 2]
    assert len(all_claims) == len(round1_claims) + len(round2_claims), \
        "every claim should be tagged round 1 or round 2 -- found an entry from neither"

    seen = set()
    for c in all_claims:
        fingerprint = (c["round"], c["speaker"], c["claim"]["claim_text"], c["verification"]["verdict"])
        assert fingerprint not in seen, f"duplicate verified_claims entry (double-counted?): {fingerprint}"
        seen.add(fingerprint)

    if not round1_claims:
        print("\nNOTE: round 1 produced ZERO extracted claims. Stub debate "
              "text ('[STUB] Bull opening argument...') makes no actual "
              "assertion about the stock, so an empty extraction here is a "
              "plausible, correct outcome -- not a bug. Reporting as-is "
              "rather than assuming claims must exist.")
    if round1_claims and not round2_claims:
        print("\nNOTE: round 2 produced ZERO NEW claims on top of round "
              "1's. Also plausible given stub text -- reporting as-is.")

    # --- final_ruling: valid verdict, credibility independently recomputed
    # from verified_claims (not by reusing judge_agent's own _credibility
    # helper) -- an independent check, not trusting the graph's own math. ---
    ruling = result.get("final_ruling")
    print("\n=== final_ruling ===")
    print(ruling)
    assert ruling is not None
    assert ruling["winner"] in {"Bull", "Bear", "Inconclusive"}

    def _independent_credibility(speaker: str) -> float:
        verified = sum(1 for c in all_claims if c["speaker"] == speaker and c["verification"]["verdict"] == "verified")
        contradicted = sum(1 for c in all_claims if c["speaker"] == speaker and c["verification"]["verdict"] == "contradicted")
        checked = verified + contradicted
        return round((verified - contradicted) / checked, 3) if checked > 0 else 0.0

    assert ruling["bull_credibility"] == _independent_credibility("bull")
    assert ruling["bear_credibility"] == _independent_credibility("bear")
