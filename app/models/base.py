"""Shared validation primitives for API-facing domain models."""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
)


class DomainModel(BaseModel):
    """Base model that rejects unknown data and validates later assignments."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
    )


Identifier = Annotated[StrictStr, Field(min_length=1, max_length=128)]
ShortText = Annotated[StrictStr, Field(min_length=1, max_length=256)]
LongText = Annotated[StrictStr, Field(min_length=1, max_length=50_000)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
RiskScore = Annotated[StrictInt, Field(ge=0, le=100)]
PositiveFiniteFloat = Annotated[
    StrictFloat,
    Field(gt=0, allow_inf_nan=False),
]
UnitInterval = Annotated[
    StrictFloat,
    Field(ge=0, le=1, allow_inf_nan=False),
]
