"""Reproducible unsupervised anomaly scoring for relationship features."""

from __future__ import annotations

import random

from sklearn.ensemble import IsolationForest

from app.models.domain import GraphAnalysisResult

SYNTHETIC_TRAINING_SEED = 17_041
MODEL_RANDOM_STATE = 17_041
SYNTHETIC_TRAINING_ROWS = 512

# Two percent of the synthetic reference population establishes the model's
# outlier threshold. This is a modelling assumption for the demo, not an
# estimate of fraud prevalence.
CONTAMINATION = 0.02

RelationshipFeatureVector = tuple[int, int, int, int, float]


def generate_synthetic_training_data() -> list[RelationshipFeatureVector]:
    """Return fixed typical relationship vectors for independent accounts.

    Every row represents a separate synthetic origin account with a handful of
    counterparties and one transfer to the investigated destination. Because
    GraphAnalyzer includes the current transaction, a value of one for
    ``repeated_transaction_count`` represents no repeated direct transfers.
    """

    generator = random.Random(SYNTHETIC_TRAINING_SEED)
    rows: list[RelationshipFeatureVector] = []

    for _ in range(SYNTHETIC_TRAINING_ROWS):
        counterparty_count = generator.randint(2, 5)
        origin_degree = counterparty_count
        shared_counterparty_count = generator.choices(
            population=(0, 1),
            weights=(4, 1),
            k=1,
        )[0]
        repeated_transaction_count = 1
        concentration_ratio = round(generator.uniform(0.15, 0.55), 6)
        rows.append(
            (
                counterparty_count,
                origin_degree,
                shared_counterparty_count,
                repeated_transaction_count,
                concentration_ratio,
            )
        )

    return rows


class RelationshipAnomalyScorer:
    """Score graph metrics against a fixed synthetic IsolationForest baseline."""

    def __init__(self) -> None:
        self._model = IsolationForest(
            contamination=CONTAMINATION,
            random_state=MODEL_RANDOM_STATE,
        )
        self._model.fit(generate_synthetic_training_data())

    def score(
        self,
        graph_analysis_result: GraphAnalysisResult,
    ) -> tuple[float, bool]:
        """Return an anomaly score and whether the fitted model flags the row.

        The score is the negated IsolationForest decision function, so positive
        values fall beyond the fitted outlier threshold and are anomalous.
        """

        features: RelationshipFeatureVector = (
            graph_analysis_result.counterparty_count,
            graph_analysis_result.origin_degree,
            graph_analysis_result.shared_counterparty_count,
            graph_analysis_result.repeated_transaction_count,
            float(graph_analysis_result.concentration_ratio),
        )
        anomaly_score = -float(self._model.decision_function([features])[0])
        is_anomalous = bool(self._model.predict([features])[0] == -1)
        return anomaly_score, is_anomalous
