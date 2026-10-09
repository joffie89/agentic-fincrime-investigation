"""Small, deterministic NetworkX analysis for an investigation transaction.

The graph is deliberately simple: accounts are nodes and each transaction is a
separate directed edge.  A ``MultiDiGraph`` preserves repeated transactions
between the same two accounts, while metrics are calculated direction-agnostic
because both incoming and outgoing relationships matter during an investigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

import networkx as nx

from app.models.domain import GraphAnalysisResult
from app.models.transaction import Transaction


class TransactionConflictError(ValueError):
    """Raised when a transaction ID conflicts with historical details."""


@dataclass(frozen=True, slots=True)
class AccountTransactionEdge:
    """A synthetic or historical account-to-account transaction edge."""

    transaction_id: str
    origin_account_id: str
    destination_account_id: str
    amount: float


# Written solely for this demo.  The compact network includes repeated direct
# transfers and counterparties shared by two accounts so every metric can be
# demonstrated without customer data or an external graph database.
SYNTHETIC_RELATIONSHIPS: tuple[AccountTransactionEdge, ...] = (
    AccountTransactionEdge("hist-001", "acct-demo-origin", "acct-demo-dest", 900.0),
    AccountTransactionEdge("hist-002", "acct-demo-origin", "acct-demo-dest", 1_100.0),
    AccountTransactionEdge("hist-003", "acct-demo-origin", "acct-shared-1", 125.0),
    AccountTransactionEdge("hist-004", "acct-demo-dest", "acct-shared-1", 140.0),
    AccountTransactionEdge("hist-005", "acct-shared-2", "acct-demo-origin", 80.0),
    AccountTransactionEdge("hist-006", "acct-demo-dest", "acct-shared-2", 95.0),
    AccountTransactionEdge("hist-007", "acct-demo-origin", "acct-other-1", 75.0),
    AccountTransactionEdge("hist-008", "acct-demo-origin", "acct-other-2", 60.0),
    AccountTransactionEdge("hist-009", "acct-unrelated-1", "acct-unrelated-2", 50.0),
)


class GraphAnalyzer:
    """Calculate transparent relationship signals for a transaction.

    The supplied transaction is included in a temporary graph for analysis. It
    is never written back to the historical graph, so repeated calls are
    deterministic and do not inflate counts.
    """

    HIGH_CONNECTIVITY_COUNTERPARTIES = 5
    SHARED_COUNTERPARTIES = 2
    REPEATED_DIRECT_TRANSACTIONS = 3
    CONCENTRATION_RATIO = 0.75

    def __init__(
        self,
        relationships: Iterable[AccountTransactionEdge | Mapping[str, object]] | None = None,
    ) -> None:
        self._graph = nx.MultiDiGraph()
        seen_transaction_ids: set[str] = set()
        source = SYNTHETIC_RELATIONSHIPS if relationships is None else relationships
        for relationship in source:
            edge = self._coerce_relationship(relationship)
            if edge.transaction_id in seen_transaction_ids:
                raise ValueError(
                    f"Duplicate historical transaction_id: {edge.transaction_id}"
                )
            seen_transaction_ids.add(edge.transaction_id)
            self._add_relationship(edge)

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Return a defensive copy of the historical graph."""

        return self._graph.copy()

    def analyze(self, transaction: Transaction) -> GraphAnalysisResult:
        """Analyze an investigation transaction against historical relationships."""

        current = self._transaction_edge(transaction)
        graph = self._graph.copy()
        existing = self._find_transaction(graph, current.transaction_id)
        if existing is None:
            self._add_relationship(current, graph=graph)
        elif not self._same_transaction(existing, current):
            raise TransactionConflictError(
                "Current transaction_id collides with a different historical transaction"
            )

        origin = current.origin_account_id
        destination = current.destination_account_id
        origin_edges = self._incident_edges(graph, origin)
        destination_edges = self._incident_edges(graph, destination)

        origin_counterparties = self._counterparties(origin_edges, origin)
        destination_counterparties = self._counterparties(destination_edges, destination)
        counterparty_count = len(origin_counterparties)
        origin_degree = len(origin_edges)

        shared = (origin_counterparties & destination_counterparties) - {
            origin,
            destination,
        }
        shared_counterparty_count = len(shared)

        direct_edges = [
            edge
            for edge in origin_edges
            if {edge[0], edge[1]} == {origin, destination}
        ]
        repeated_transaction_count = len(direct_edges)

        total_origin_amount = sum(
            float(edge[3]["amount"]) for edge in origin_edges
        )
        direct_amount = sum(
            float(edge[3]["amount"]) for edge in direct_edges
        )
        concentration_ratio = (
            round(direct_amount / total_origin_amount, 6)
            if total_origin_amount > 0
            else 0.0
        )

        signals: list[str] = []
        if counterparty_count >= self.HIGH_CONNECTIVITY_COUNTERPARTIES:
            signals.append("HIGH_CONNECTIVITY")
        if shared_counterparty_count >= self.SHARED_COUNTERPARTIES:
            signals.append("SHARED_COUNTERPARTIES")
        if repeated_transaction_count >= self.REPEATED_DIRECT_TRANSACTIONS:
            signals.append("REPEATED_DIRECT_TRANSACTIONS")
        # A single first-time transfer necessarily has ratio 1.0, so require at
        # least one other incident relationship before treating concentration as
        # meaningful.
        if len(origin_edges) >= 2 and concentration_ratio >= self.CONCENTRATION_RATIO:
            signals.append("COUNTERPARTY_CONCENTRATION")

        return GraphAnalysisResult(
            origin_account_id=origin,
            destination_account_id=destination,
            counterparty_count=counterparty_count,
            origin_degree=origin_degree,
            shared_counterparty_count=shared_counterparty_count,
            repeated_transaction_count=repeated_transaction_count,
            concentration_ratio=concentration_ratio,
            signals=signals,
            suspicious=bool(signals),
        )

    @staticmethod
    def _coerce_relationship(
        relationship: AccountTransactionEdge | Mapping[str, object],
    ) -> AccountTransactionEdge:
        if isinstance(relationship, AccountTransactionEdge):
            edge = relationship
        else:
            try:
                edge = AccountTransactionEdge(
                    transaction_id=str(relationship["transaction_id"]),
                    origin_account_id=str(relationship["origin_account_id"]),
                    destination_account_id=str(relationship["destination_account_id"]),
                    amount=float(relationship["amount"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid account transaction relationship") from exc
        GraphAnalyzer._validate_edge(edge)
        return edge

    @staticmethod
    def _transaction_edge(transaction: Transaction) -> AccountTransactionEdge:
        try:
            origin_account_id = getattr(transaction, "origin_account_id", None)
            destination_account_id = getattr(transaction, "destination_account_id", None)
            if not origin_account_id or not str(origin_account_id).strip():
                origin_country = str(getattr(transaction, "origin_country", "unknown")).strip()
                origin_account_id = f"origin:{origin_country or 'unknown'}"
            if not destination_account_id or not str(destination_account_id).strip():
                destination_country = str(
                    getattr(transaction, "destination_country", "unknown")
                ).strip()
                destination_account_id = f"destination:{destination_country or 'unknown'}"
            edge = AccountTransactionEdge(
                transaction_id=str(transaction.transaction_id),
                origin_account_id=str(origin_account_id),
                destination_account_id=str(destination_account_id),
                amount=float(transaction.amount),
            )
        except AttributeError as exc:
            raise ValueError(
                "Transaction must include transaction_id and amount"
            ) from exc
        GraphAnalyzer._validate_edge(edge)
        return edge

    @staticmethod
    def _validate_edge(edge: AccountTransactionEdge) -> None:
        if not edge.transaction_id.strip():
            raise ValueError("transaction_id must not be blank")
        if not edge.origin_account_id.strip() or not edge.destination_account_id.strip():
            raise ValueError("Account identifiers must not be blank")
        if edge.origin_account_id == edge.destination_account_id:
            raise ValueError("Origin and destination accounts must differ")
        if not isfinite(edge.amount) or edge.amount <= 0:
            raise ValueError("Transaction amount must be a positive finite number")

    def _add_relationship(
        self,
        edge: AccountTransactionEdge,
        *,
        graph: nx.MultiDiGraph | None = None,
    ) -> None:
        target = self._graph if graph is None else graph
        target.add_edge(
            edge.origin_account_id,
            edge.destination_account_id,
            key=edge.transaction_id,
            transaction_id=edge.transaction_id,
            amount=edge.amount,
        )

    @staticmethod
    def _find_transaction(
        graph: nx.MultiDiGraph,
        transaction_id: str,
    ) -> AccountTransactionEdge | None:
        for origin, destination, _, data in graph.edges(keys=True, data=True):
            if data.get("transaction_id") == transaction_id:
                return AccountTransactionEdge(
                    transaction_id=transaction_id,
                    origin_account_id=str(origin),
                    destination_account_id=str(destination),
                    amount=float(data["amount"]),
                )
        return None

    @staticmethod
    def _same_transaction(
        left: AccountTransactionEdge,
        right: AccountTransactionEdge,
    ) -> bool:
        return (
            left.transaction_id == right.transaction_id
            and left.origin_account_id == right.origin_account_id
            and left.destination_account_id == right.destination_account_id
            and left.amount == right.amount
        )

    @staticmethod
    def _incident_edges(
        graph: nx.MultiDiGraph,
        account_id: str,
    ) -> list[tuple[str, str, str, dict[str, object]]]:
        incoming = list(graph.in_edges(account_id, keys=True, data=True))
        outgoing = list(graph.out_edges(account_id, keys=True, data=True))
        # Self-loops are rejected, so incoming and outgoing cannot overlap.
        return incoming + outgoing

    @staticmethod
    def _counterparties(
        edges: Iterable[tuple[str, str, str, Mapping[str, object]]],
        account_id: str,
    ) -> set[str]:
        return {
            destination if origin == account_id else origin
            for origin, destination, _, _ in edges
        }
