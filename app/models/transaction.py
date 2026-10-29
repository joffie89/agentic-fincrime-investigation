"""Transaction input model."""

from pydantic import StrictBool, model_validator

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

    @model_validator(mode="after")
    def validate_distinct_accounts(self) -> "Transaction":
        if (
            self.origin_account_id is not None
            and self.destination_account_id is not None
            and self.origin_account_id == self.destination_account_id
        ):
            raise ValueError("origin_account_id and destination_account_id must differ")
        return self
