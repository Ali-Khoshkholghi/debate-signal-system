# Known limitations (detail)

**Small live-test sample.** 6 tickers tested end-to-end (AAPL, TSLA,
MSFT, TQQQ, GOOGL, NVDA), across two sessions. That sample found and
fixed five real, outcome-flipping bugs — which says more about how much
was left to find than about how clean the code is now. No claim of
correctness beyond "these 6 debates, checked by hand, now come out
right."

**Free-tier API quotas still apply.** The Gemini -> Groq -> Cerebras
fallback chain raises the effective daily ceiling, it doesn't remove it.
Gemini's free tier caps at 20 requests/day; Groq's at ~100K tokens/day;
Cerebras' is far larger (~1M tokens/day) but not unlimited, and was
observed hitting its own transient `queue_exceeded` ("high traffic")
errors during testing — on one occasion all three tiers failed within the
same two-second window, truncating a debate round's argument generation
to empty text. Not a code bug; a real shared-infrastructure risk worth
knowing about.

**Fixed debate structure.** Always exactly two sides (Bull/Bear), always
exactly two rounds. No support for N-way debates, additional personas, or
a variable round count based on how contested the claims are.

**Verification checks factual anchors only, not interpretation quality.**
`verify_claim` confirms whether a claimed number/relationship/sentiment
matches real data. It never evaluates whether the conclusion built on top
of that fact is reasonable — a verified P/E of 40 and the claim "this
means the stock is fairly valued" both currently score identically
regardless of whether 40 actually supports that conclusion for this
sector/growth profile. See the Roadmap.

**News verification is aggregate-sentiment-only.** The only news evidence
available to check a claim against is one aggregate sentiment label
(`positive`/`negative`/`neutral`) across all fetched headlines — there's
no per-headline quote verification. In practice this mostly worked in the
system's favor during testing (it correctly caught cherry-picked real
quotes being used to imply a false overall narrative), but it means a
claim that accurately quotes one real headline can still be scored
`contradicted` if that headline doesn't represent the aggregate.

**Comparison tolerance is a relative band, not exact match.** Numeric
claims are matched within a 10% relative tolerance (or 2 percentage
points for percent-scale fields) — deliberately, so a rough transcript
paraphrase like "around 23" isn't falsely contradicted by a real value of
23.4. The tradeoff: a claim that's genuinely ~9% off from the real number
still verifies.

## Roadmap

Directionally, not a spec:

- **Inconclusive rulings occurred often — open question, not yet a
  diagnosed flaw.** Across the 5-ticker live batch (post-bug-fixes), 3 of
  5 debates landed at an exact 0.0 credibility gap (Inconclusive) — only
  2 produced a decisive winner. Two competing explanations, not yet
  distinguished: (a) genuinely balanced evidence between two competent,
  honest arguers may legitimately produce real ties this often — correct
  behavior, not a flaw; or (b) the credibility formula may be too coarse
  at typical claim counts (5-12 per side) to distinguish a genuine tie
  from a case that should have been decisive. Five debates is too small a
  sample to tell which explanation is right; revisit as more tickers get
  tested.
- **Surface the credibility gap size on Inconclusive rulings, not just the
  label.** Any gap below `FINAL_RULING_MARGIN` (0.2) currently reports as
  a flat "Inconclusive" — a genuine 0.0 tie and a close 0.15 near-miss
  look identical in the output, even though the gap is already computed
  in `judge_final_ruling` and simply discarded before it reaches the
  report. Low-cost fix: surface the actual gap and a "weak lean toward X"
  label alongside "Inconclusive" (e.g. `INCONCLUSIVE -- weak lean toward
  Bull (gap: 0.068)` vs. `INCONCLUSIVE -- no lean (gap: 0.0)`), without
  changing the underlying decision not to force a winner under genuine
  uncertainty. Not yet built.
- **Verifying reasoning/interpretation quality, not just facts** — see
  above. Not implemented; explicitly deferred in the verification code's
  own docstrings today.
- **Wiring up a dedicated Judge model.** `judge_config.py` defines
  `JUDGE_MODEL_NAME`/`ANTHROPIC_API_KEY` for this but nothing currently
  imports it — every LLM call goes through the same router the evidence
  agents use.
