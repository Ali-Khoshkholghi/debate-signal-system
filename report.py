"""Console report formatter for the Judge's debate output.

Mirrors stock-signal-system's own main.py::print_report style (agent-by-
agent evidence blocks, PASS/FAIL guardrails, a prominent final-verdict
banner) applied to DebateState's debate structure instead of that
project's flat AgentState. Presentation only -- reads the dict returned by
judge_graph.run_judge(), never touches graph/node/state logic, same
separation P1 keeps between its agents and main.py's printing code.
"""

EVIDENCE_AGENTS = [
    ("Price Agent", "price_result"),
    ("News Agent", "news_result"),
    ("Fundamentals Agent", "fundamentals_result"),
    ("Risk Agent", "risk_result"),
]


def _print_evidence_block(label: str, agent_result: dict | None) -> None:
    if not agent_result:
        print(f"\n[{label}]")
        print("  did not run")
        return

    output = agent_result.get("output") or {}
    guardrail = agent_result.get("guardrail") or {}

    print(f"\n[{label}]")
    print(f"  Status: {agent_result.get('status')}")

    if label == "Price Agent" and output:
        print(f"  Current price: {output.get('current_price')}")
        print(f"  Change: {output.get('change_pct')}%")
        print(f"  20d SMA: {output.get('sma_20')}   50d SMA: {output.get('sma_50')}")
        print(f"  20d volatility: {output.get('volatility_20d')}%")
        print(f"  Trend: {output.get('trend')}")
    elif label == "News Agent" and output:
        print(f"  Relevant headlines used: {len(output.get('headlines', []))}")
        for h in output.get("headlines", [])[:5]:
            print(f"    - {h}")
        if output.get("sentiment") is not None:
            print(f"  Sentiment: {output.get('sentiment')}  (agreement: {output.get('agreement_ratio')}, source diversity: {output.get('source_diversity')})")
    elif label == "Fundamentals Agent" and output:
        if output.get("quote_type") == "ETF":
            print(f"  Asset type: ETF")
            print(f"  Category: {output.get('category')}")
            print(f"  Net expense ratio: {output.get('expense_ratio')}%")
            print(f"  NAV: {output.get('nav_price')}")
            print(f"  Total assets: {output.get('total_assets')}")
            print(f"  Dividend yield: {output.get('dividend_yield')}%")
        else:
            print(f"  P/E ratio: {output.get('pe_ratio')}")
            print(f"  Revenue growth YoY: {output.get('revenue_growth_yoy')}%")
            print(f"  EPS: {output.get('eps')}")
            print(f"  Profit margin: {output.get('profit_margin')}%")
    elif label == "Risk Agent" and output:
        print(f"  Annualized volatility: {output.get('volatility_annualized')}%")
        print(f"  Risk level: {output.get('risk_level')}")

    if agent_result.get("error"):
        print(f"  Error: {agent_result['error']}")

    print(f"  Guardrail: {'PASS' if guardrail.get('passed') else 'FAIL'} -- {guardrail.get('reason')}")
    for check, ok in (guardrail.get("checks") or {}).items():
        print(f"    - {check}: {'PASS' if ok else 'FAIL'}")


def print_evidence_summary(ticker: str, evidence: dict | None) -> None:
    print("\n" + "-" * 60)
    print(f"EVIDENCE SUMMARY -- {ticker}")
    print("-" * 60)

    if not evidence:
        print("  no evidence gathered")
        return

    classification = evidence.get("ticker_classification") or {}
    if classification:
        print(f"\n[Ticker Classification]")
        print(f"  Quote type: {classification.get('quote_type')}")
        print(f"  Name: {classification.get('long_name') or classification.get('short_name')}")

    for label, key in EVIDENCE_AGENTS:
        _print_evidence_block(label, evidence.get(key))


def print_transcript(transcript_turns: list[dict]) -> None:
    print("\n" + "-" * 60)
    print("DEBATE TRANSCRIPT")
    print("-" * 60)

    if not transcript_turns:
        print("  no debate turns recorded")
        return

    rounds = sorted({t["round"] for t in transcript_turns})
    for round_num in rounds:
        print(f"\n=== Round {round_num} ===")
        for turn in [t for t in transcript_turns if t["round"] == round_num]:
            print(f"\n{turn['speaker'].upper()}:")
            print(f"  {turn['argument']}")
            if turn.get("rebuttal_note"):
                print(f"  ↳ Rebuttal note: {turn['rebuttal_note']}")


VERDICT_LABELS = {"verified": "VERIFIED", "contradicted": "CONTRADICTED", "unverifiable": "UNVERIFIABLE"}


def _print_claim_line(entry: dict) -> None:
    claim = entry["claim"]
    verification = entry["verification"]
    verdict_label = VERDICT_LABELS.get(verification["verdict"], verification["verdict"].upper())

    print(f"  [{verdict_label}] {claim['claim_text']}")
    print(f"      reason: {verification['reason']}")
    if verification.get("claimed_value") is not None or verification.get("real_value") is not None:
        print(f"      claimed: {verification.get('claimed_value')}   real: {verification.get('real_value')}")


def print_verified_claims(verified_claims: list[dict]) -> None:
    print("\n" + "-" * 60)
    print("VERIFIED CLAIMS")
    print("-" * 60)

    if not verified_claims:
        print("  no claims extracted")
        return

    rounds = sorted({c["round"] for c in verified_claims})
    for round_num in rounds:
        print(f"\n=== Round {round_num} ===")
        for speaker in ("bull", "bear"):
            entries = [c for c in verified_claims if c["round"] == round_num and c["speaker"] == speaker]
            if not entries:
                continue
            print(f"\n{speaker.upper()}:")
            for entry in entries:
                _print_claim_line(entry)


def print_final_ruling(final_ruling: dict | None) -> None:
    print("\n" + "=" * 60)
    if not final_ruling:
        print("FINAL RULING: N/A (ruling not produced)")
        print("=" * 60)
        return

    print(f"FINAL RULING: {final_ruling.get('winner', 'N/A').upper()}")
    print("=" * 60)
    print(f"  Bull credibility: {final_ruling.get('bull_credibility')}  "
          f"(verified: {final_ruling.get('bull_verified')}, contradicted: {final_ruling.get('bull_contradicted')})")
    print(f"  Bear credibility: {final_ruling.get('bear_credibility')}  "
          f"(verified: {final_ruling.get('bear_verified')}, contradicted: {final_ruling.get('bear_contradicted')})")


def print_debate_report(ticker: str, run_id: str, result: dict) -> None:
    print("=" * 60)
    print(f"DEBATE SIGNAL REPORT -- {ticker}")
    print("=" * 60)
    print(f"Run ID: {run_id}")

    print_evidence_summary(ticker, result.get("evidence"))
    print_transcript(result.get("transcript_turns") or [])
    print_verified_claims(result.get("verified_claims") or [])
    print_final_ruling(result.get("final_ruling"))

    print("\n" + "=" * 60)
