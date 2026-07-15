"""Shared fixtures for debate-signal-system's own tests."""
import pytest

from models.router import ModelCallResult, StructuredCallResult


def make_agent_result(
    agent: str,
    status: str = "ok",
    output: dict | None = None,
    guardrail: dict | None = None,
    model_used: str | None = "gemini-2.5-flash",
    escalated: bool = False,
    duration_s: float = 0.1,
    error: str | None = None,
) -> dict:
    """Build a fake per-agent result dict, matching the shape stock-signal-
    system's agents return under e.g. state['fundamentals_result'] -- used
    here to build fake evidence dicts for verify_claim tests without
    running the real agents. Mirrors that package's own
    tests/conftest.py::make_agent_result."""
    return {
        "agent": agent,
        "status": status,
        "output": output,
        "guardrail": guardrail if guardrail is not None else {"passed": status == "ok", "reason": "stub", "checks": {}},
        "model_used": model_used,
        "escalated": escalated,
        "duration_s": duration_s,
        "error": error,
    }


def make_model_result(
    text: str = "Reasonable one-sentence analysis.",
    model_used: str = "gemini-2.5-flash",
    escalated: bool = False,
    error: str | None = None,
    tokens_in: int = 50,
    tokens_out: int = 20,
    cost_usd: float = 0.0001,
) -> ModelCallResult:
    """Build a fake ModelCallResult, standing in for a real
    call_model_with_routing(...)/_call_gemini/_call_groq call. Mirrors
    stock-signal-system's own tests/conftest.py::make_model_result."""
    return ModelCallResult(
        text=text,
        model_used=model_used,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        escalated=escalated,
        error=error,
    )


def make_structured_result(
    data=None,
    text: str = "",
    model_used: str = "gemini-2.5-flash",
    escalated: bool = False,
    error: str | None = None,
    tokens_in: int = 50,
    tokens_out: int = 20,
    cost_usd: float = 0.0001,
) -> StructuredCallResult:
    """Build a fake StructuredCallResult, standing in for a real
    call_structured_model(...) call. `data` is the already-validated
    Pydantic model instance (e.g. ExtractedClaims(...)), or None to
    simulate every retry/fallback attempt failing schema validation.
    Mirrors stock-signal-system's own tests/conftest.py::make_structured_result."""
    return StructuredCallResult(
        data=data,
        text=text,
        model_used=model_used,
        escalated=escalated,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        error=error,
    )


@pytest.fixture
def stub_store(monkeypatch):
    """Replace stock-signal-system's db.store logging with no-ops so tests
    that call the reused agents never write to a real db file. Mirrors that
    package's own tests/conftest.py::stub_store fixture."""
    import db.store as store

    monkeypatch.setattr(store, "log_agent_result", lambda *a, **k: None)
    monkeypatch.setattr(store, "log_cost", lambda *a, **k: None)
    monkeypatch.setattr(store, "start_run", lambda *a, **k: None)
    return store
