"""Debate/Judge orchestration -- not designed yet.

Placeholder confirming the reuse seam works end to end: the Judge calls
evidence.tools.gather_evidence() for the agent evidence, and can use
evidence.tools.call_model_with_routing / call_structured_model for its own
LLM calls via the same routing logic stock-signal-system's agents use.
"""
from evidence.tools import ensure_run, gather_evidence


def run_judge(ticker: str, run_id: str) -> dict:
    ensure_run(run_id, ticker)
    evidence = gather_evidence(ticker, run_id)
    # TODO: debate logic -- multiple perspectives argue over `evidence`,
    # Judge weighs them into a verdict. Out of scope for this scaffold.
    return {"ticker": ticker, "run_id": run_id, "evidence": evidence, "verdict": None}
