"""Shared HTTP error handling for API-backed clients."""

from __future__ import annotations

import httpx

from evalkit.models.base import ModelError, RetryableModelError

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def post_json(url: str, headers: dict[str, str], payload: dict, timeout_s: float) -> dict:
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout_s)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise RetryableModelError(f"network error calling {url}: {exc}") from exc

    if resp.status_code in RETRYABLE_STATUS:
        raise RetryableModelError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise ModelError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()
