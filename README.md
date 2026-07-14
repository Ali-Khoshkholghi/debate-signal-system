# debate-signal-system

A multi-agent Judge that debates a stock signal, using [stock-signal-system](https://github.com/Ali-Khoshkholghi/stock-signal-system)'s
price/fundamentals/risk/news/classify_ticker agents as its evidence-gathering
tools.

This is an independent project — its own repo, venv, dependencies, and test
suite — not a fork or branch of stock-signal-system. It depends on that
project the way any Python project depends on a library: as a pip package.

## Dependency on stock-signal-system

`evidence/tools.py` is the seam where reuse happens: it imports
`classify_ticker_agent`, `price_agent`, `fundamentals_agent`, `risk_agent`,
`news_agent`, and `models.router`'s `call_model_with_routing` /
`call_structured_model` from the installed `stock-signal-system` package,
and wires them in the dependency order that package's own `graph.py` uses
(classify first; price before risk, since risk reuses price's volatility
data).

No code is copied. Bugfixes and improvements to those agents in
stock-signal-system are picked up here by bumping the dependency version —
see `requirements.txt`.

**Reserved names**: `stock-signal-system` installs as flat, unnamespaced
top-level modules (`agents`, `db`, `models`, `guardrails`, `config`, `state`,
`graph`) — matching that project's own internal imports exactly, so nothing
there had to change to make it installable. This project's own modules
therefore avoid those names (e.g. `judge_config.py` instead of `config.py`,
`judge_state.py` instead of `state.py`).

**Database isolation**: the reused agents log to a SQLite db via
`config.DB_PATH`, which stock-signal-system now reads from the
`STOCK_SIGNAL_DB_PATH` env var if set (see `.env.example`) — so this
project's runs don't write into stock-signal-system's own database.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # pinned to stock-signal-system's v0.1.0 tag
cp .env.example .env   # fill in TAVILY_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY
pytest
```

`requirements.txt` installs `stock-signal-system` pinned to its `v0.1.0` git
tag — this is what makes the repo resolvable by anyone who clones
`debate-signal-system` on its own, without a sibling checkout.

For local development against an unreleased change in stock-signal-system,
use `requirements-dev.txt` instead — it installs that project editable from
the sibling checkout (`../stock_signal_system - Reliability-Trust`) so edits
there are picked up immediately without re-tagging:

```bash
pip install -r requirements-dev.txt
```

## Layout

```
judge_config.py       Project settings (Judge model, API keys)
judge_state.py         Debate state shape
evidence/tools.py       Reuse seam: wraps stock-signal-system's agents as evidence tools
judge/judge_agent.py    Debate/Judge orchestration (placeholder -- not designed yet)
main.py                 CLI entry point
tests/                  pytest suite (stubs db writes by default; `-m live` for real API calls)
```

## Status

Scaffold only. `judge/judge_agent.py` proves the reuse path works end to
end (calls `evidence.tools.gather_evidence`) but the actual debate/judging
logic isn't designed yet.
