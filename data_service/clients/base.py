from __future__ import annotations

from typing import Any

import httpx
from django.conf import settings


class DownstreamError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        service: str = "",
        unavailable: bool = False,
        timeout: bool = False,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.service = service
        self.unavailable = unavailable
        self.timeout = timeout
        super().__init__(message)


def http_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        float(settings.REQUEST_READ_TIMEOUT),
        connect=float(settings.REQUEST_CONNECT_TIMEOUT),
    )


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def extract_error(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            return err
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
    return fallback


def request_json(
    *,
    method: str,
    url: str,
    service: str,
    token: str | None = None,
    json_body: dict | None = None,
    require_auth: bool = False,
) -> tuple[int, Any]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if require_auth:
        if not token:
            raise DownstreamError(
                f"{service} API key is not configured.",
                service=service,
                unavailable=True,
            )
        headers.update(auth_headers(token))
    if json_body is not None:
        headers["Content-Type"] = "application/json"

    try:
        with httpx.Client(timeout=http_timeout()) as client:
            response = client.request(method, url, headers=headers, json=json_body)
    except httpx.ConnectError as exc:
        raise DownstreamError(
            f"{service} is unavailable.",
            service=service,
            unavailable=True,
        ) from exc
    except httpx.TimeoutException as exc:
        raise DownstreamError(
            f"{service} request timed out.",
            service=service,
            unavailable=True,
            timeout=True,
        ) from exc
    except httpx.HTTPError as exc:
        raise DownstreamError(
            f"{service} request failed.",
            service=service,
            unavailable=True,
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        if response.status_code >= 400:
            raise DownstreamError(
                f"{service} returned an error response.",
                status_code=response.status_code,
                service=service,
                unavailable=response.status_code >= 500,
            ) from exc
        raise DownstreamError(
            f"{service} returned a malformed response.",
            status_code=response.status_code,
            service=service,
        ) from exc

    if response.status_code >= 400:
        raise DownstreamError(
            extract_error(payload, f"{service} request failed."),
            status_code=response.status_code,
            service=service,
            unavailable=response.status_code >= 500,
        )
    return response.status_code, payload


def get_unauthenticated(url: str, service: str, *, not_ready_message: str) -> Any:
    try:
        with httpx.Client(timeout=http_timeout()) as client:
            response = client.get(url)
    except httpx.ConnectError as exc:
        raise DownstreamError(
            f"{service} is unavailable.",
            service=service,
            unavailable=True,
        ) from exc
    except httpx.TimeoutException as exc:
        raise DownstreamError(
            f"{service} request timed out.",
            service=service,
            unavailable=True,
            timeout=True,
        ) from exc
    except httpx.HTTPError as exc:
        raise DownstreamError(
            f"{service} unreachable.",
            service=service,
            unavailable=True,
        ) from exc
    if response.status_code >= 400:
        raise DownstreamError(
            not_ready_message,
            status_code=response.status_code,
            service=service,
            unavailable=True,
        )
    try:
        return response.json()
    except ValueError:
        return {"status": "ok"}
