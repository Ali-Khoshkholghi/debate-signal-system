import operator
from typing import Annotated, TypedDict


class DebateState(TypedDict, total=False):
    """State for the Judge's debate graph (see judge_graph.py). Named
    judge_state.py, not state.py, which the installed stock-signal-system
    package owns.

    evidence is a frozen snapshot -- written once by gather_evidence_node
    and never rewritten by any later node, so every claim from both rounds
    is checked against the same data instead of independently re-fetched
    live values (see judge/judge_agent.py::verify_claim's docstring).

    transcript_turns/verified_claims use operator.add reducers so each node
    only returns its own increment rather than the full accumulated
    history. Safe here specifically because exactly one node runs at a
    time in this graph (no concurrent writes) -- confirmed with an
    isolated smoke test (a throwaway 3-node linear graph) before this
    state/graph was built, mirroring the "test any non-trivial join first"
    discipline stock-signal-system's own graph.py documents.
    """

    ticker: str
    run_id: str

    evidence: dict | None

    # {"round": 1, "speaker": "bull", "argument": "...", "rebuttal_note": "..." | None}
    # argument is the debate content (what extract_claims consumes);
    # rebuttal_note is meta-commentary about verification results and is
    # NEVER passed to extract_claims (see ArgumentTurn in judge/schemas.py).
    transcript_turns: Annotated[list[dict], operator.add]

    # {"round": 1, "speaker": "bull", "claim": <Claim.model_dump()>,
    #  "verification": <dataclasses.asdict(VerificationResult)>}
    verified_claims: Annotated[list[dict], operator.add]

    final_ruling: dict | None
