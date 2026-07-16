# debate-signal-system

A Bull/Bear/Judge debate system for stock signals: two LLM-driven debaters
argue opposite cases over two rounds, and every checkable factual claim
either side makes gets verified against the same real evidence snapshot
both of them argued from — [stock-signal-system](https://github.com/Ali-Khoshkholghi/stock-signal-system)'s
price/fundamentals/risk/news agents, reused as evidence tools, not
duplicated. The final ruling is a deterministic tally of verified-vs-
contradicted claims, never an LLM's read of who argued more persuasively.

This differs from TradingAgents-style debate frameworks in one specific
way: there, a judge model scores which side sounded more convincing. Here,
the judge never grades rhetoric — it checks whether each side's specific
numbers and relationships actually match the data.

This is an independent project — its own repo, venv, dependencies, and
test suite — not a fork or branch of stock-signal-system. It depends on
that project the way any Python project depends on a library: as a pip
package.

## Architecture

`judge_graph.py` wires an 8-node, strictly sequential LangGraph — every
node has exactly one incoming edge, no conditional routing, no multi-
source joins:

```
gather_evidence -> bull_round1 -> bear_round1 -> judge_verify_round1
                 -> bull_round2 -> bear_round2 -> judge_verify_round2
                 -> judge_final_ruling
```

`gather_evidence` runs once, before round 1, and is never rewritten —
every claim from both rounds is checked against that same frozen
snapshot, not independently re-fetched live values. Each round has each
speaker argue, then a `judge_verify_roundN` node extracts every checkable
claim from that round's arguments and checks it against the snapshot.
`judge_final_ruling` is a deterministic tally (see below) — no LLM call
decides the winner, following the same "no agent grades its own work"
principle stock-signal-system's guardrails follow.

**`rebuttal_note` / `argument` structural separation.** Each debate turn
(`ArgumentTurn` in `judge/schemas.py`) has two fields: `argument` (the
actual market claims) and `rebuttal_note` (meta-commentary about the
prior round's fact-check results, e.g. "I concede the P/E claim was
contradicted"). Claim extraction runs on `argument` only, enforced in
code, not by prompting the model to phrase things carefully — a live
integration test proved wording alone isn't reliable here: identical
debate-mechanics commentary ("round 1 had 0 verified claims...") got
classified inconsistently as a real market claim depending on which
speaker said it. Splitting it into a field the extractor never sees
closed that gap structurally, not by asking the model to try harder.

**Three-tier model fallback: Gemini -> Groq -> Cerebras.** Every LLM call
in this project — claim extraction, anchor extraction, argument
generation — goes through the same `call_model_with_routing`/
`call_structured_model` stock-signal-system uses for its own agents.
Gemini is primary; Groq is the first fallback; Cerebras is a third tier,
added because Groq's ~100K-tokens/day free-tier budget was the binding
constraint on live batch testing (Cerebras' free tier runs roughly 1M
tokens/day, no card required). Not hypothetical: during this project's
live testing, with Gemini's daily quota and Groq's daily token budget
both fully exhausted, multiple complete debate runs — evidence gathering,
both rounds, verification, final ruling — still finished successfully
purely on the Cerebras tier.

## Dependency on stock-signal-system

`evidence/tools.py` is the reuse seam: it imports `classify_ticker_agent`,
`price_agent`, `fundamentals_agent`, `risk_agent`, `news_agent`, and
`models.router`'s `call_model_with_routing`/`call_structured_model` from
the installed `stock-signal-system` package, wired in that package's own
dependency order (classify first; price before risk, since risk reuses
price's volatility data). No code is copied — bugfixes and improvements
land here by bumping the pinned version.

**Reserved names**: `stock-signal-system` installs as flat, unnamespaced
top-level modules (`agents`, `db`, `models`, `guardrails`, `config`,
`state`, `graph`) — matching that project's own internal imports. This
project's own modules avoid those names (`judge_config.py` instead of
`config.py`, `judge_state.py` instead of `state.py`, `judge_graph.py`
instead of `graph.py`).

**Database isolation**: the reused agents log to a SQLite db via
`config.DB_PATH`, which reads from the `STOCK_SIGNAL_DB_PATH` env var if
set — so this project's runs don't write into stock-signal-system's own
database.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # pinned to stock-signal-system's v0.1.1 tag
cp .env.example .env   # fill in TAVILY_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY
pytest
```

Run it:

```bash
python main.py AAPL
```

This prints the formatted debate report (`report.py`): the evidence
summary per agent, the full transcript, every extracted claim with its
verdict, and the final ruling. Errors (API rate limits, parse failures,
etc.) are logged to `debate.log`, not the console — `main.py` prints a
one-line pointer ("N error(s) logged... see debate.log") if any occurred,
so the report stays readable even when a fallback tier had to kick in.

For local development against an unreleased change in stock-signal-system,
use `requirements-dev.txt` instead — it installs that project editable
from the sibling checkout so edits there are picked up immediately
without re-tagging.

## Real failures found and handled

These happened during live batch testing across real tickers, not in a
unit test — each was reproduced with the exact real claim text before
being fixed, and each was outcome-flipping (changed which side won a
real debate), not cosmetic.

**1. Vague qualitative language fabricated a numeric value ("high" P/E)**

A Bull round-2 claim — *"AAPL's strong fundamentals, including a high P/E
ratio and significant revenue growth"* — named a real metric (P/E ratio)
but never asserted an actual number, only the word "high". Anchor
extraction nonetheless returned `anchor_type="value", metric="pe_ratio",
claimed_value="high"`, which the comparison step then checked against the
real `pe_ratio` (38.16) and scored `CONTRADICTED` — a single parsing
artifact that flipped the whole debate's ruling from Bull-favorable to
Bear winning.

Root cause: the anchor schema's "no matching evidence field" branch was
defined too narrowly ("the field doesn't exist at all"), so it was never
a truthful option once a real metric name was mentioned — the model was
left inventing a placeholder to satisfy the schema. Fixed by adding a
fourth anchor type, `named_but_no_value`: metric recognized, no real
number/category actually stated, scored `unverifiable`, never
`contradicted`.

**2. Risk claims checked against the wrong evidence field (volatility_20d
vs. volatility_annualized)**

A Bear claim — *"the volatility of 2.3 over 20 days"* — exactly matched
`risk_result`'s own `volatility_20d` (2.3). But `RISK_TECHNICAL_FIELDS`
only listed `volatility_annualized` as a valid metric for risk claims, so
the claim was forced onto that field instead and checked against 36.51 —
`CONTRADICTED`. Bear would have gone 8/8 clean (credibility 1.0, tying
Bull); scoring 7/8 with one false contradiction (credibility 0.75)
flipped the ruling from what should have been Inconclusive to "Bull
wins".

Root cause: `risk_agent` copies `price_agent`'s `volatility_20d` straight
into its own output alongside the `volatility_annualized` it derives from
it, so `risk_result` actually carries both period variants — but the
allow-list only exposed one of them. Fixed by adding `volatility_20d` to
the allow-list and its tolerance-handling set, plus explicit
period-matching guidance in the anchor-extraction prompt ("if a claim
states a period like '20 days' or 'annualized', match the field whose
name reflects that period, not just the general concept") — the same fix
also covers `price_technical`'s analogous `sma_20`/`sma_50` case
defensively.

**3. Categorical values (trend, risk level) misclassified as vague
language**

*"This is a medium-risk name given its volatility profile"* was
misclassified as `named_but_no_value` (no real value stated) even though
"medium" IS `risk_level`'s actual value, not a vague adjective the way
"high" is for an open-ended field like P/E. Reproduced identically for
`price_technical`'s `trend` field ("strongly uptrending", "a clear
downtrend"). The extraction prompt's "vague qualitative language"
guidance was written for open-ended numeric fields and didn't distinguish
those from categorical fields with a small, fixed set of real values.

Fixed by adding explicit categorical-field guidance to the same prompt:
descriptive/adjectival phrasing that names or clearly implies one of a
categorical field's real values ("uptrending" implies `trend=up`,
"medium-risk" implies `risk_level=medium`) is a definite value assertion,
not vague language.

**4. Unit-shorthand numbers ("$38.9 billion") failed to parse and were
marked contradicted**

A Bull claim — *"TQQQ has a massive asset base of over $38.9 billion"* —
against a real `total_assets` of `38968848384` (exactly $38.97B) was
scored `CONTRADICTED`. The comparison step only ever stripped `%` and `,`
before calling `float()` on the claimed value; the word "billion"
survived into the string, `float()` raised, and the exception handler
treated any parse failure as "no match". Confirmed airtight: the
identical real number, spelled out in full digits ("$38,968,848,384") in
a later round, verified correctly, since only the comma needed stripping
there. This contaminated two live TQQQ runs — Bull would have gone from
5/6 (credibility 0.667) to 6/6 (1.0), tying Bear and flipping the ruling
from "Bear wins" to Inconclusive.

Fixed with deterministic unit-suffix normalization (`billion`/`bn`/`b`,
`million`/`mm`/`m`, `thousand`/`k`, case-insensitive) in the comparison
layer — deliberately not in the extraction prompt. The extraction step
was already working correctly (it faithfully transcribed "38.9 billion"
exactly as claimed); pushing the unit conversion into the LLM would mean
asking it to multiply a decimal by a 9-digit number correctly every time,
which is exactly the kind of computation this project consistently keeps
in deterministic code instead of an LLM's hands.

## Known limitations

Full detail in [`LIMITATIONS.md`](LIMITATIONS.md). Headline items:

- **Small live-test sample** — 5 tickers tested end-to-end so far (AAPL,
  TSLA, MSFT, TQQQ, GOOGL). Enough to find and fix four real bugs, not
  enough to be confident the verification layer is bug-free.
- **Free-tier API quotas still apply** — three fallback tiers raise the
  daily ceiling, they don't remove it; Cerebras' free tier has its own
  transient "high traffic" limits, observed directly during testing.
- **Fixed debate structure** — always exactly two sides (Bull/Bear) and
  two rounds. No N-way debates, no variable round counts.
- **Verification checks factual anchors only** — never whether the
  interpretation built on a verified fact is reasonable. See the Roadmap
  below.

## Roadmap

Headline items:

- **Inconclusive rulings occurred often** — 3 of 5 live-tested debates
  landed at an exact 0.0 credibility gap. Open question: genuinely
  balanced arguers producing real ties, or the credibility formula being
  too coarse at typical claim counts to tell a genuine tie from a case
  that should have been decisive.
- **Inconclusive doesn't distinguish a genuine tie from a close call** —
  any gap under MARGIN reports as a flat "Inconclusive" today, even
  though the gap size is already computed. A "weak lean toward X (gap:
  0.068)" label is a low-cost future improvement.
- **Verification doesn't check interpretation quality**, only factual
  anchors — e.g. that a P/E of 40 really is 40, never whether "this shows
  fair value" is a reasonable read of that number.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full roadmap.

## Layout

```
judge_config.py         Project settings (Judge model, API keys) -- not yet wired up, see Roadmap
judge_state.py           Debate state shape
judge_graph.py           LangGraph wiring (8-node sequential graph)
evidence/tools.py         Reuse seam: wraps stock-signal-system's agents as evidence tools
judge/judge_agent.py      Debate/Judge orchestration: argument generation, claim extraction, verification, final ruling
judge/schemas.py          Pydantic shapes for the Judge's own LLM calls
report.py                 Console report formatter
main.py                   CLI entry point
tests/                    pytest suite (stubs db writes by default; `-m live` for real API calls)
```
