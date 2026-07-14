"""Shared fixtures for debate-signal-system's own tests."""
import pytest


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
