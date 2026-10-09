"""API-key authentication dependency."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Reject requests that do not contain the configured API key."""

    expected = request.app.state.settings.api_key.get_secret_value()
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
