"""Evidence-gathering tools for the Judge.

Thin wrappers around the `stock-signal-system` package (installed as a pip
dependency -- see ../requirements.txt), reusing its agents instead of
duplicating them. Each agent function has the same shape in that repo:
`agent(state: dict) -> dict`, reading/writing a shared state dict and
logging to its own db as a side effect (see db.store in that package).

Call order mirrors stock-signal-system's own graph.py wiring: classify_ticker
first (fundamentals_agent and news_agent read its output), then price before
risk (risk reuses price's volatility data and auto-skips without it).
"""
import agents.classify_ticker_agent as classify_ticker_agent
import agents.fundamentals_agent as fundamentals_agent
import agents.news_agent as news_agent
import agents.price_agent as price_agent
import agents.risk_agent as risk_agent
import db.store as store

# Re-exported so the Judge's own LLM calls can use the same routing/
# escalation logic as the evidence agents, rather than a second call path.
from models.router import call_model_with_routing, call_structured_model  # noqa: F401


def ensure_run(run_id: str, ticker: str) -> None:
    """Create the `runs` row stock-signal-system's db logging requires
    (agent_logs/cost_logs have a FOREIGN KEY to runs.run_id). Call once per
    debate, before gathering evidence for that run."""
    store.init_db()
    store.start_run(run_id, ticker)


def gather_evidence(ticker: str, run_id: str) -> dict:
    """Run the five evidence agents for one ticker, in their real dependency
    order, and return the per-agent result dicts the Judge can weigh."""
    state = {"ticker": ticker, "run_id": run_id}

    state.update(classify_ticker_agent.classify_ticker_agent(state))
    state.update(price_agent.price_agent(state))
    state.update(news_agent.news_agent(state))
    state.update(fundamentals_agent.fundamentals_agent(state))
    state.update(risk_agent.risk_agent(state))

    return {
        "ticker_classification": state.get("ticker_classification"),
        "price_result": state.get("price_result"),
        "news_result": state.get("news_result"),
        "fundamentals_result": state.get("fundamentals_result"),
        "risk_result": state.get("risk_result"),
    }
