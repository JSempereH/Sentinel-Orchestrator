"""Bearer-token auth dependency shared by every job endpoint."""

import secrets

from fastapi import Header, HTTPException

from .config import settings


def require_token(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {settings.worker_api_token}"
    # secrets.compare_digest, not `!=`: a plain string comparison short-circuits
    # on the first mismatched byte, which leaks how many leading characters of
    # the token a guess got right through response-time differences.
    if not settings.worker_api_token or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
