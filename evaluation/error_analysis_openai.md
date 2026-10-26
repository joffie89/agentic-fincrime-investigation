# OpenAI decision-evaluation error analysis

This report analyses `evaluation/decision_results_openai.json` against
`evaluation/decision_cases.json`. The run contains 50 completed cases, no workflow
failures, and no LLM fallbacks.

## Summary

- Recommendation accuracy: **37/50 (74%)**.
- Human-review routing accuracy: **49/50 (98%)**.
- Exact citation match rate: **25/50 (50%)**.
- There was **one missed escalation** and no unnecessary escalations.
- All 13 recommendation errors were in the MEDIUM group. Clean and HIGH cases
  had perfect recommendation classification.
- The dominant citation issue was `DEMO-AML-006` appearing as an extra relative
  to the labels: 16 occurrences, all on cases where `new_payee` was `true`.

## 1. Missed escalation

The only missed escalation was **young account shows a payment velocity spike**
(`decision-medium-003`).

### Full labelled input

```json
{
  "transaction_id": "decision-medium-003",
  "amount": 90.0,
  "account_age_days": 9,
  "new_payee": false,
  "average_transaction_amount": 100.0,
  "transactions_last_24h": 14,
  "origin_country": "GB",
  "destination_country": "GB",
  "origin_account_id": "medium-origin-003",
  "destination_account_id": "medium-destination-003"
}
```

| Field | Expected | Actual |
|---|---|---|
| Case group | MEDIUM | MEDIUM workflow input |
| Recommendation | `REQUEST_MORE_INFO` | `CLEAR` |
| Requires human review | `true` | `false` |
| Policy references | `DEMO-AML-001`, `DEMO-AML-002` | none |
| LLM fallback | n/a | `false` |
| Workflow error | n/a | none |

The transaction has two deterministic signals: an account younger than 30 days
and more than 10 transactions in 24 hours. Despite those signals, the assessment
returned `CLEAR`, cited neither expected policy, and was automatically completed.

### Routing and assessment details

The two evaluation JSON files do **not** record the persisted routing object, exact
confidence, `evidence_sufficient`, or LLM reasons. Those exact values and reasons
cannot be recovered from these files.

The following can be inferred from the current MEDIUM-risk routing guard, but was
not serialized in the evaluation result:

- Routing rule: `automatic_completion_criteria_met`.
- Confidence: exact value unavailable; it must have been **at least 0.9**.
- `evidence_sufficient`: exact saved value unavailable; it must have been `true`.
- The assessment's `requires_human_review` must have been `false`, and no other
  review rule can have fired.
- LLM reasons: **not recorded; no reliable reconstruction is possible**.

The completed result has investigation ID
`d5226228-962f-4784-9074-44c6404fbdc2`, end-to-end latency
`3105.5252000223845 ms`, and estimated LLM cost `$0.0008476`.

## 2. Recommendation confusion matrices

Rows are expected recommendations and columns are actual recommendations. All 50
cases completed, so there is no null/failure column population.

### Clean group (15 cases)

| Expected \ Actual | `CLEAR` | `REQUEST_MORE_INFO` | `ESCALATE_TO_HUMAN` | Total |
|---|---:|---:|---:|---:|
| `CLEAR` | 15 | 0 | 0 | 15 |
| `REQUEST_MORE_INFO` | 0 | 0 | 0 | 0 |
| `ESCALATE_TO_HUMAN` | 0 | 0 | 0 | 0 |

### HIGH group (15 cases)

| Expected \ Actual | `CLEAR` | `REQUEST_MORE_INFO` | `ESCALATE_TO_HUMAN` | Total |
|---|---:|---:|---:|---:|
| `CLEAR` | 0 | 0 | 0 | 0 |
| `REQUEST_MORE_INFO` | 0 | 0 | 0 | 0 |
| `ESCALATE_TO_HUMAN` | 0 | 0 | 15 | 15 |

### MEDIUM group (20 cases)

| Expected \ Actual | `CLEAR` | `REQUEST_MORE_INFO` | `ESCALATE_TO_HUMAN` | Total |
|---|---:|---:|---:|---:|
| `CLEAR` | 0 | 0 | 0 | 0 |
| `REQUEST_MORE_INFO` | 7 | 7 | 0 | 14 |
| `ESCALATE_TO_HUMAN` | 0 | 6 | 0 | 6 |

The 13 recommendation mismatches were:

- `REQUEST_MORE_INFO` to `CLEAR` (7): `decision-medium-001`,
  `decision-medium-003`, `decision-medium-004`, `decision-medium-006`,
  `decision-medium-008`, `decision-medium-011`, `decision-medium-013`.
- `ESCALATE_TO_HUMAN` to `REQUEST_MORE_INFO` (6):
  `decision-medium-015`, `decision-medium-016`, `decision-medium-017`,
  `decision-medium-018`, `decision-medium-019`, `decision-medium-020`.

Routing prevented 12 of these 13 recommendation errors from becoming a missed
human review. Only `decision-medium-003` was auto-completed.

## 3. Citation mismatch analysis

There were 25 exact citation matches and 25 mismatches.

| Mismatch type | Cases | Meaning |
|---|---:|---|
| MISSING only | 8 | At least one expected ID was absent; no unexpected ID was cited. |
| EXTRA only | 5 | At least one unexpected ID was cited; no expected ID was absent. |
| BOTH | 12 | Expected IDs were absent and unexpected IDs were cited. |
| **Total** | **25** | |

### Policy-ID frequency

Counts below are occurrences across mismatched cases; one case can contribute
multiple IDs.

| Missing policy ID | Count |
|---|---:|
| `DEMO-AML-004` | 9 |
| `DEMO-AML-001` | 6 |
| `DEMO-AML-005` | 6 |
| `DEMO-AML-002` | 4 |

| Extra policy ID | Count |
|---|---:|
| `DEMO-AML-006` | 16 |
| `DEMO-AML-016` | 2 |

### Every citation mismatch

| Case ID | Expected `policy_references` | Actual `policy_references` | Type | Missing IDs | Extra IDs |
|---|---|---|---|---|---|
| `decision-high-001` | `001, 005` | `001, 005, 006` | **EXTRA** | — | `006` |
| `decision-high-002` | `001, 004, 005` | `001, 006, 005` | **BOTH** | `004` | `006` |
| `decision-high-004` | `001, 002, 005` | `001, 002, 006` | **BOTH** | `005` | `006` |
| `decision-high-005` | `001, 002, 005` | `001, 002, 006` | **BOTH** | `005` | `006` |
| `decision-high-006` | `001, 004, 005` | `001, 006, 005` | **BOTH** | `004` | `006` |
| `decision-high-008` | `001, 002, 005` | `001, 002, 006` | **BOTH** | `005` | `006` |
| `decision-high-009` | `001, 002, 005` | `001, 002, 006` | **BOTH** | `005` | `006` |
| `decision-high-010` | `001, 004, 005` | `001, 005, 006` | **BOTH** | `004` | `006` |
| `decision-high-012` | `001, 002, 005` | `001, 002, 006` | **BOTH** | `005` | `006` |
| `decision-high-014` | `001, 004, 005` | `001, 005, 006` | **BOTH** | `004` | `006` |
| `decision-high-015` | `001, 002, 005` | `001, 002, 006` | **BOTH** | `005` | `006` |
| `decision-medium-001` | `001` | none | **MISSING** | `001` | — |
| `decision-medium-002` | `001` | `001, 006` | **EXTRA** | — | `006` |
| `decision-medium-003` | `001, 002` | none | **MISSING** | `001, 002` | — |
| `decision-medium-004` | `001, 002` | none | **MISSING** | `001, 002` | — |
| `decision-medium-005` | `001` | `001, 006, 016` | **EXTRA** | — | `006, 016` |
| `decision-medium-006` | `001` | none | **MISSING** | `001` | — |
| `decision-medium-009` | `001, 004` | `001, 006` | **BOTH** | `004` | `006` |
| `decision-medium-010` | `001, 002, 004` | `001, 004, 006` | **BOTH** | `002` | `006` |
| `decision-medium-011` | `001, 002, 004` | none | **MISSING** | `001, 002, 004` | — |
| `decision-medium-012` | `001, 004` | `001, 004, 006` | **EXTRA** | — | `006` |
| `decision-medium-013` | `001, 004` | none | **MISSING** | `001, 004` | — |
| `decision-medium-014` | `001, 002, 004` | `001, 002` | **MISSING** | `004` | — |
| `decision-medium-015` | `001, 003` | `001, 003, 016` | **EXTRA** | — | `016` |
| `decision-medium-018` | `001, 003, 004` | `001, 003` | **MISSING** | `004` | — |

All abbreviated IDs in this table use the `DEMO-AML-` prefix. Classification is
set-based, matching the evaluation metric; display order preserves the saved lists.

## 4. Five most common failure patterns

Patterns overlap: a single case may appear in more than one pattern.

1. **`DEMO-AML-006` was extra relative to the labels — 16 cases.**
   `decision-high-001`, `decision-high-002`, `decision-high-004`,
   `decision-high-005`, `decision-high-006`, `decision-high-008`,
   `decision-high-009`, `decision-high-010`, `decision-high-012`,
   `decision-high-014`, `decision-high-015`, `decision-medium-002`,
   `decision-medium-005`, `decision-medium-009`, `decision-medium-010`,
   `decision-medium-012`. Every one has `new_payee=true`, so this looks like a
   systematic difference between the labelled signal mapping and the broader
   policy applicability inferred by the model, rather than random IDs.

2. **`DEMO-AML-004` was missing — 9 cases.**
   `decision-high-002`, `decision-high-006`, `decision-high-010`,
   `decision-high-014`, `decision-medium-009`, `decision-medium-011`,
   `decision-medium-013`, `decision-medium-014`, `decision-medium-018`.
   The recurring omission weakens grounding for cross-border cases.

3. **Expected `REQUEST_MORE_INFO` became `CLEAR` — 7 cases.**
   `decision-medium-001`, `decision-medium-003`, `decision-medium-004`,
   `decision-medium-006`, `decision-medium-008`, `decision-medium-011`,
   `decision-medium-013`. Six returned no policy references; routing caught all
   but `decision-medium-003`, requiring review for the other six.

4. **`DEMO-AML-005` was missing from HIGH cases — 6 cases.**
   `decision-high-004`, `decision-high-005`, `decision-high-008`,
   `decision-high-009`, `decision-high-012`, `decision-high-015`. In every case,
   the actual list contained extra `DEMO-AML-006` in place of the expected
   human-review/evidence policy.

5. **Expected `ESCALATE_TO_HUMAN` became `REQUEST_MORE_INFO` — 6 cases.**
   `decision-medium-015`, `decision-medium-016`, `decision-medium-017`,
   `decision-medium-018`, `decision-medium-019`, `decision-medium-020`. These
   are the graph-connected MEDIUM cases. All six still routed to human review,
   limiting the operational impact of the softer recommendation.

`DEMO-AML-001` omission also occurred in six cases
(`decision-medium-001`, `decision-medium-003`, `decision-medium-004`,
`decision-medium-006`, `decision-medium-011`, `decision-medium-013`) and is tied
in frequency with patterns 4 and 5.

## Main conclusion

The safety routing layer was substantially stronger than the raw recommendation:
it reduced 13 recommendation errors to one routing miss. The remaining critical
failure combined three errors in the same MEDIUM case: an unexpectedly clear
recommendation, no citations despite two expected policies, and automatic
completion. Citation accuracy is also being depressed by a systematic label/model
disagreement around `DEMO-AML-006`; that should be separated from genuine missing
evidence before using the 50% exact-match score as a model-quality verdict.
