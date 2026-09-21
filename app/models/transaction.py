"""Transaction input model."""

from pydantic import StrictBool

from app.models.base import (
    DomainModel,
    Identifier,
    NonNegativeInt,
    PositiveFiniteFloat,
    ShortText,
)


class Transaction(DomainModel):
    """Validated transaction and account context used by an investigation."""

    transaction_id: Identifier
    amount: PositiveFiniteFloat
    account_age_days: NonNegativeInt
    new_payee: StrictBool
    average_transaction_amount: PositiveFiniteFloat
    transactions_last_24h: NonNegativeInt
    origin_country: ShortText
    destination_country: ShortText
    origin_account_id: Identifier | None = None
    destination_account_id: Identifier | None = None
