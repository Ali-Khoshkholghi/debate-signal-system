from typing import TypedDict


class DebateState(TypedDict, total=False):
    """Placeholder shape for the Judge's own state -- named judge_state.py
    (not state.py, which the installed stock-signal-system package owns).
    Fields will grow once the debate/judge orchestration itself is designed;
    for now this only carries what evidence/tools.gather_evidence produces.
    """

    ticker: str
    run_id: str
    ticker_classification: dict | None
    price_result: dict | None
    news_result: dict | None
    fundamentals_result: dict | None
    risk_result: dict | None
    verdict: dict | None
