"""Transaction input model."""

import re

from pydantic import StrictBool, field_validator, model_validator

from app.models.base import (
    ISO_3166_1_ALPHA_2_CODES,
    AccountIdentifier,
    CountryCode,
    DomainModel,
    Identifier,
    NonNegativeInt,
    PositiveFiniteFloat,
)

_ACCOUNT_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")


class Transaction(DomainModel):
    """Validated transaction and account context used by an investigation."""

    transaction_id: Identifier
    amount: PositiveFiniteFloat
    account_age_days: NonNegativeInt
    new_payee: StrictBool
    average_transaction_amount: PositiveFiniteFloat
    transactions_last_24h: NonNegativeInt
    origin_country: CountryCode
    destination_country: CountryCode
    origin_account_id: AccountIdentifier | None = None
    destination_account_id: AccountIdentifier | None = None

    @field_validator("origin_country", "destination_country", mode="before")
    @classmethod
    def validate_country_code(cls, value: object) -> object:
        """Require an assigned uppercase ISO code without normalising input."""

        if isinstance(value, str) and (
            _COUNTRY_CODE_PATTERN.fullmatch(value) is None or value not in ISO_3166_1_ALPHA_2_CODES
        ):
            raise ValueError(
                "country codes must be assigned ISO 3166-1 alpha-2 codes "
                "using exactly two uppercase letters"
            )
        return value

    @field_validator("origin_account_id", "destination_account_id", mode="before")
    @classmethod
    def validate_account_identifier_characters(cls, value: object) -> object:
        """Reject whitespace and free text before global string trimming runs."""

        if isinstance(value, str) and _ACCOUNT_IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "account identifiers may contain only letters, digits, hyphens, "
                "and underscores, with a maximum length of 64 characters"
            )
        return value

    @model_validator(mode="after")
    def validate_distinct_accounts(self) -> "Transaction":
        if (
            self.origin_account_id is not None
            and self.destination_account_id is not None
            and self.origin_account_id == self.destination_account_id
        ):
            raise ValueError("origin_account_id and destination_account_id must differ")
        return self
