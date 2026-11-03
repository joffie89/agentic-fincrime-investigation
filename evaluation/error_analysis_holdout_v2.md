# Held-out v2 decision error analysis

## Scope and method

This is an offline analysis of `decision_results_openai_holdout_v2.json`, cross-referenced with `decision_cases_holdout_v2.json`. It does not rerun the workflow. The results contain 100 cases evaluated in three runs (300 case-run observations). `R1`, `R2`, and `R3` below refer to those recorded runs.

An **unnecessary review** is a case with `expected_requires_human_review=false` and `actual_requires_human_review=true`. The cause is classified from the pre-floor `original_llm_recommendation`:

- **LLM over-escalation:** the LLM itself returned `ESCALATE_TO_HUMAN`; the recorded routing rule was `policy_assessment_requires_human_review`.
- **Routing-rule-caused review:** the LLM returned `REQUEST_MORE_INFO`, but `medium_risk_assessment_not_clear` set the human-review flag. The final recommendation remained `REQUEST_MORE_INFO`.

The recorded aggregates reconcile with the case-level data:

| Metric | R1 | R2 | R3 |
|---|---:|---:|---:|
| Unnecessary human reviews | 17 | 17 | 16 |
| LLM-only unnecessary escalations | 12 | 12 | 11 |
| Routing-rule-caused unnecessary reviews | 5 | 5 | 5 |
| LLM-only missed escalations, out of 40 expected | 2 | 2 | 3 |
| System missed escalations, out of 40 expected | 0 | 0 | 0 |

## Unnecessary human reviews

### Clean cases

None of the 25 clean cases was routed to human review in any run.

### Single-signal cases

#### Velocity-only: LLM over-escalation

These cases were expected to return `REQUEST_MORE_INFO` without human review. In every listed unnecessary-review observation, the LLM returned `ESCALATE_TO_HUMAN`, the final recommendation remained an escalation, and `policy_assessment_requires_human_review` fired.

| Case | Unexpected-review runs | Routing rule | Cause | Shortened LLM reasons and run variation |
|---|---|---|---|---|
| `holdout v2 single velocity 001` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 37 transactions exceeded 10; no other risk signals. R1 also said cross-border activity was context, not a risk signal. |
| `holdout v2 single velocity 002` | R1-R2 only | `policy_assessment_requires_human_review` | LLM over-escalation | R1-R2: 23 transactions exceeded 10. **Varied in R3:** the LLM returned `REQUEST_MORE_INFO`, citing high velocity but no other signals; `automatic_completion_criteria_met` fired and no human review was required. |
| `holdout v2 single velocity 003` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 13 transactions exceeded 10. R2 added that the US-to-CA route was context, not a risk signal. |
| `holdout v2 single velocity 004` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 36 transactions exceeded 10. |
| `holdout v2 single velocity 005` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 16 transactions exceeded 10; velocity was the stated concern. |
| `holdout v2 single velocity 006` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 15 transactions exceeded 10. R1 said velocity alone mandated escalation and that cross-border activity was not itself a signal. |
| `holdout v2 single velocity 007` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 18 transactions exceeded 10; other risk signals were absent and cross-border activity was contextual only. |
| `holdout v2 single velocity 008` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 43 transactions exceeded 10. R3 explicitly noted that the other signals were absent and cross-border activity was not itself a signal. |
| `holdout v2 single velocity 009` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 15 transactions exceeded 10. |
| `holdout v2 single velocity 010` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | 14 transactions exceeded 10. Some runs mentioned the US-to-CA route as context; R2 said velocity alone required review. |

The velocity-only subset therefore contributed 29 unnecessary-review observations: 10 in R1, 10 in R2, and 9 in R3. `single velocity 002` was the only unnecessary-review case whose routing outcome varied across runs.

#### Unusual-amount-only: routing-rule-caused review

These cases were also expected to return `REQUEST_MORE_INFO` without human review. The LLM returned the expected recommendation in all three runs, but `medium_risk_assessment_not_clear` set `actual_requires_human_review=true`.

| Case | Unexpected-review runs | Routing rule | Cause | Shortened LLM reasons |
|---|---|---|---|---|
| `holdout v2 single unusual amount 021` | R1-R3 | `medium_risk_assessment_not_clear` | Routing rule | Amount exceeded 5x average; no other signal, so escalation was not required. |
| `holdout v2 single unusual amount 022` | R1-R3 | `medium_risk_assessment_not_clear` | Routing rule | Amount exceeded 5x average; no other signal was identified. |
| `holdout v2 single unusual amount 023` | R1-R3 | `medium_risk_assessment_not_clear` | Routing rule | Amount exceeded 5x average; no other signal, so further information was needed. |
| `holdout v2 single unusual amount 024` | R1-R3 | `medium_risk_assessment_not_clear` | Routing rule | Amount exceeded 5x average; no other signal, so the LLM described a cautious review. |
| `holdout v2 single unusual amount 025` | R1-R3 | `medium_risk_assessment_not_clear` | Routing rule | Amount exceeded 5x average; no other signal, so more information or cautious review was needed. |

The unusual-amount-only subset contributed 15 unnecessary-review observations, five in every run. No young-account-only or new-payee-only case was routed unexpectedly.

### Boundary cases

Both boundary cases below were expected to return `REQUEST_MORE_INFO` without human review. The LLM returned `ESCALATE_TO_HUMAN` in every run, and `policy_assessment_requires_human_review` fired.

| Case | Unexpected-review runs | Routing rule | Cause | Shortened LLM reasons |
|---|---|---|---|---|
| `holdout v2 boundary transactions 11 004` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | Exactly 11 transactions was above the greater-than-10 velocity threshold; the other signals were absent. |
| `holdout v2 boundary age 30 and transactions 11 008` | R1-R3 | `policy_assessment_requires_human_review` | LLM over-escalation | Exactly 11 transactions triggered velocity; exactly 30 days was not young. Other signals were absent, and cross-border activity was described as context rather than a signal. |

The boundary group contributed six unnecessary-review observations, two in every run.

### Multi-signal or network-link cases

None was an unnecessary human review because all 40 cases in this group were labelled `ESCALATE_TO_HUMAN` and expected human review. The LLM-only misses within this group are recorded separately below.

## LLM-only missed escalations

There were seven LLM-only miss observations across three unique multi-signal cases. In each observation, the LLM returned `REQUEST_MORE_INFO`; the rubric floor raised the final recommendation to `ESCALATE_TO_HUMAN`, so the system recorded no missed escalations.

| Case | Missed in | Pre-floor LLM -> final | Shortened LLM reasons and run variation |
|---|---|---|---|
| `holdout v2 multi young and unusual amount 002` | R1-R3 | `REQUEST_MORE_INFO` -> `ESCALATE_TO_HUMAN` | Identified an 8-day-old account and an amount over 5x average. R1 explicitly said that "only 2" signals did not make escalation mandatory; R2-R3 also noted no velocity or network signal. |
| `holdout v2 multi young and new payee 023` | R1-R3 | `REQUEST_MORE_INFO` -> `ESCALATE_TO_HUMAN` | Identified an account under 30 days old and a new payee, while noting no unusual amount or other signal. |
| `holdout v2 multi young and new payee 001` | R3 only | `REQUEST_MORE_INFO` -> `ESCALATE_TO_HUMAN` | R3 identified the 23-day-old account and new payee, then noted that amount, velocity, and network signals were absent and cross-border activity was contextual. In R1-R2 the LLM returned `ESCALATE_TO_HUMAN` and said the two signals required escalation. |

## Policy citations omitted by the LLM

`deterministic_citations` records expected policy references absent from the LLM output and subsequently added by code. Across all 300 observations, every one of the 117 additions was the same policy:

| Omitted policy | Title | R1 | R2 | R3 | Total additions | Unique affected cases |
|---|---|---:|---:|---:|---:|---:|
| `DEMO-AML-004` | Cross-border and geographic context | 37 | 38 | 42 | 117 | 44 |

No other policy ID appeared in `deterministic_citations`. Retrieval itself contained all 189 expected policy references in each run; the recorded issue was omission from the LLM's cited references. Deterministic additions brought the final exact citation match to 100/100 in every run.

The `DEMO-AML-004` omissions were distributed as follows:

| Case group | R1 | R2 | R3 | Total additions | Unique affected cases |
|---|---:|---:|---:|---:|---:|
| Clean | 13 | 13 | 13 | 39 | 13 |
| Single-signal, velocity-only | 5 | 5 | 5 | 15 | 6 |
| Single-signal, other | 4 | 4 | 5 | 13 | 6 |
| Boundary | 2 | 3 | 3 | 8 | 3 |
| Multi-signal or network-link | 13 | 13 | 16 | 42 | 16 |

## Factual count summary

| Finding or cause | Unique cases across runs | R1 | R2 | R3 | Case-run observations |
|---|---:|---:|---:|---:|---:|
| Unnecessary review caused by LLM over-escalation | 12 | 12 | 12 | 11 | 35 |
| Unnecessary review caused by `medium_risk_assessment_not_clear` | 5 | 5 | 5 | 5 | 15 |
| **All unnecessary reviews** | **17** | **17** | **17** | **16** | **50** |
| LLM-only missed escalation (separate error class) | 3 | 2 | 2 | 3 | 7 |
| System missed escalation | 0 | 0 | 0 | 0 | 0 |
| LLM citation omission repaired deterministically (`DEMO-AML-004`) | 44 | 37 | 38 | 42 | 117 |
