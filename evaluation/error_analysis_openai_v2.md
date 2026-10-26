# OpenAI decision-evaluation error analysis v2

Source: `evaluation/decision_results_openai.json`, generated
`2026-10-24T13:54:49.991202+00:00` in OpenAI/vector mode, joined to case IDs
from `evaluation/decision_cases.json`.

## Summary

- There are 11 LLM fallbacks. Every one was labelled
  `ESCALATE_TO_HUMAN` and the conservative fallback also returned
  `ESCALATE_TO_HUMAN`.
- The result file does not record the fallback exception type/message or a raw
  retrieved-ID field. No persisted log file containing those exceptions exists
  in the workspace, and fallback retrieval sets cannot be reconstructed.
- For the 39 non-fallback cases, 58 of 71 expected policy occurrences were in
  the recoverable retrieved set: **81.69% micro retrieval recall**. The 13
  missing occurrences explain all 13 non-fallback citation mismatches.
- Excluding fallbacks, recommendation accuracy is **36/39 (92.31%)** and
  routing accuracy is **39/39 (100.00%)**.

## 1. Fallback cases

For all rows below, `workflow_error` is `null`, confidence is `0.0`,
`evidence_sufficient` is `false`, both policy lists are empty, and the only
recorded reason is `AI assessment failed; human review is required.` The
exception type/message was written only to stdout by the application logger;
it was not serialized into this result file or a workspace log. LLM usage is
present for all 11 cases, showing that a response was received before the
fallback occurred, but the saved data cannot identify which parsing,
allow-list, validation, or post-processing check failed.

| Case ID | Expected recommendation | Exception type | Exception message |
|---|---|---|---|
| `decision-high-003` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-high-004` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-high-007` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-high-008` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-high-013` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-medium-003` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-medium-004` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-medium-010` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-medium-011` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-medium-015` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |
| `decision-medium-020` | `ESCALATE_TO_HUMAN` | not recorded | not recorded |

## 2. Retrieval recall and citation impact

The result schema has no `retrieved_policy_ids` field. For a successful
assessment, the prompt requires the three retrieved policies to be partitioned
between `policy_references` and `policies_considered`; every non-fallback row
does contain three unique IDs across that union. The table therefore uses the
union of `policy_references` and `policies_considered` as the recoverable
retrieved set.
Fallback construction clears both lists, so the original retrieved set for
those 11 rows is **not recoverable** and an empty list must not be interpreted
as zero retrieval.

IDs in the table abbreviate the `DEMO-AML-` prefix. The 15 clean cases are
grouped because all have the same comparison.

| Case ID | Expected IDs | Recoverable retrieved IDs | Expected IDs missing |
|---|---|---|---|
| `decision-clean-001` through `015` | none | `001, 004, 019` | none |
| `decision-high-001` | `001, 005, 006` | `001, 005, 006` | none |
| `decision-high-002` | `001, 004, 005, 006` | `001, 005, 006` | `004` |
| `decision-high-003` | `001, 002, 005` | not recorded (fallback) | unknown |
| `decision-high-004` | `001, 002, 005, 006` | not recorded (fallback) | at least 1; ID unknown |
| `decision-high-005` | `001, 002, 005, 006` | `001, 002, 006` | `005` |
| `decision-high-006` | `001, 004, 005, 006` | `001, 005, 006` | `004` |
| `decision-high-007` | `001, 002, 005` | not recorded (fallback) | unknown |
| `decision-high-008` | `001, 002, 005, 006` | not recorded (fallback) | at least 1; ID unknown |
| `decision-high-009` | `001, 002, 005, 006` | `001, 002, 006` | `005` |
| `decision-high-010` | `001, 004, 005, 006` | `001, 005, 006` | `004` |
| `decision-high-011` | `001, 002, 005` | `001, 002, 005` | none |
| `decision-high-012` | `001, 002, 005, 006` | `001, 002, 006` | `005` |
| `decision-high-013` | `001, 002, 005` | not recorded (fallback) | unknown |
| `decision-high-014` | `001, 004, 005, 006` | `001, 005, 006` | `004` |
| `decision-high-015` | `001, 002, 005, 006` | `001, 002, 006` | `005` |
| `decision-medium-001` | `001` | `001, 004, 019` | none |
| `decision-medium-002` | `001, 006` | `001, 006, 016` | none |
| `decision-medium-003` | `001, 002` | not recorded (fallback) | unknown |
| `decision-medium-004` | `001, 002, 006` | not recorded (fallback) | unknown |
| `decision-medium-005` | `001, 006` | `001, 006, 016` | none |
| `decision-medium-006` | `001` | `001, 005, 016` | none |
| `decision-medium-007` | `001, 002` | `001, 002, 016` | none |
| `decision-medium-008` | `001, 004` | `001, 004, 019` | none |
| `decision-medium-009` | `001, 004, 006` | `001, 006, 016` | `004` |
| `decision-medium-010` | `001, 002, 004, 006` | not recorded (fallback) | at least 1; ID unknown |
| `decision-medium-011` | `001, 002, 004` | not recorded (fallback) | unknown |
| `decision-medium-012` | `001, 004, 006` | `001, 004, 006` | none |
| `decision-medium-013` | `001, 004` | `001, 005, 016` | `004` |
| `decision-medium-014` | `001, 002, 004` | `001, 002, 017` | `004` |
| `decision-medium-015` | `001, 003, 006` | not recorded (fallback) | unknown |
| `decision-medium-016` | `001, 003` | `001, 003, 019` | none |
| `decision-medium-017` | `001, 002, 003, 006` | `001, 002, 003` | `006` |
| `decision-medium-018` | `001, 003, 004` | `001, 003, 019` | `004` |
| `decision-medium-019` | `001, 002, 003` | `001, 002, 003` | none |
| `decision-medium-020` | `001, 003, 006` | not recorded (fallback) | unknown |

### Recall totals

- Observable non-fallback cases with non-empty expected sets: 24.
- Expected policy occurrences: 71; present in the recoverable retrieved sets:
  58; missing: 13.
- Micro recall: **58/71 = 81.69%**. Mean per-case recall across those 24
  cases: **84.38%**.
- Per-policy recall: `001` 24/24, `002` 9/9, `003` 4/4, `004` 2/10,
  `005` 6/10, and `006` 13/14. The 13 known misses are eight `004`s, four
  `005`s, and one `006`.
- Nine known misses occur in cases with four expected policies but `top_k=3`,
  so at least one miss is structurally unavoidable. The other four are `004`
  ranking misses in `decision-medium-009`, `013`, `014`, and `018`.

Across all 50 cases there are 106 expected policy occurrences, but exact recall
cannot be calculated because 35 expected occurrences belong to fallbacks whose
retrieved sets were discarded. The defensible bound is **58/106 (54.72%) to
90/106 (84.91%)**. At least 16 expected occurrences were not retrieved: the 13
observed misses plus at least one in each of the three four-policy fallback
cases, because retrieval returns only three policies.

### Is retrieval the cause?

- **Citation mismatches:** yes for every observable non-fallback mismatch. All
  13 cases with a missing expected retrieval are citation mismatches, and the
  other 26 non-fallback cases are exact matches. In those 13 cases the cited
  IDs are exactly the retrieved subset of the expected IDs, with no extras.
- **Fallback citation mismatches:** the remaining 11 citation mismatches have
  empty references because fallback sanitization clears citations. Their raw
  retrieval sets are unavailable.
- **Fallbacks themselves:** not proven. A missing expected policy could lead
  the model to emit an ID outside the retrieved allow-list, which would cause a
  validation fallback, but neither the raw model output, retrieved set, nor
  exception was saved. The available evidence cannot establish that retrieval
  caused any particular fallback.

## 3. Accuracy excluding fallbacks

| Metric | All 50 cases | Excluding 11 fallbacks |
|---|---:|---:|
| Recommendation accuracy | 47/50 (94.00%) | **36/39 (92.31%)** |
| Human-review routing accuracy | 50/50 (100.00%) | **39/39 (100.00%)** |

The three remaining recommendation errors are `decision-medium-005`,
`decision-medium-006`, and `decision-medium-013`: each expected
`ESCALATE_TO_HUMAN` but returned `REQUEST_MORE_INFO`. All three were still
routed to human review, so they did not create routing errors. Recommendation
accuracy decreases after filtering because all 11 fallback recommendations
were conservative escalations that happened to match their expected labels.
