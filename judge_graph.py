"""LangGraph wiring for the Judge's debate. Named judge_graph.py, not
graph.py -- that name is owned by the installed stock-signal-system
package (see README.md "Reserved names").

Sequential only: every node has exactly one incoming edge, no
add_conditional_edges, no multi-source list-form joins. Confirmed safe
against the join-mechanics risk stock-signal-system's own graph.py
documents (a conditional edge into a shared join silently never firing for
the branch not taken) -- that risk requires a conditional edge AND a
downstream node with multiple predecessors, and this graph has neither.

The one accumulation mechanism here (operator.add reducers on
transcript_turns/verified_claims, see judge_state.py) was smoke-tested in
isolation first, under strictly sequential execution -- a throwaway 3-node
linear graph confirming each reducer append lands in order with no
concurrent-write ambiguity, before this real 9-node graph was built.
"""
from langgraph.graph import END, START, StateGraph

from evidence.tools import ensure_run
from judge.judge_agent import (
    bear_round1,
    bear_round2,
    bull_round1,
    bull_round2,
    gather_evidence_node,
    judge_final_ruling,
    judge_verify_round1,
    judge_verify_round2,
)
from judge_state import DebateState


def build_judge_graph():
    graph = StateGraph(DebateState)
    graph.add_node("gather_evidence", gather_evidence_node)
    graph.add_node("bull_round1", bull_round1)
    graph.add_node("bear_round1", bear_round1)
    graph.add_node("judge_verify_round1", judge_verify_round1)
    graph.add_node("bull_round2", bull_round2)
    graph.add_node("bear_round2", bear_round2)
    graph.add_node("judge_verify_round2", judge_verify_round2)
    graph.add_node("judge_final_ruling", judge_final_ruling)

    graph.add_edge(START, "gather_evidence")
    graph.add_edge("gather_evidence", "bull_round1")
    graph.add_edge("bull_round1", "bear_round1")
    graph.add_edge("bear_round1", "judge_verify_round1")
    graph.add_edge("judge_verify_round1", "bull_round2")
    graph.add_edge("bull_round2", "bear_round2")
    graph.add_edge("bear_round2", "judge_verify_round2")
    graph.add_edge("judge_verify_round2", "judge_final_ruling")
    graph.add_edge("judge_final_ruling", END)

    return graph.compile()


def run_judge(ticker: str, run_id: str) -> dict:
    """Entry point: gathers one evidence snapshot, runs both debate rounds
    with verification after each, and produces a final ruling. See
    build_judge_graph() for the graph shape."""
    ensure_run(run_id, ticker)
    app = build_judge_graph()
    return app.invoke({"ticker": ticker, "run_id": run_id, "transcript_turns": [], "verified_claims": []})
