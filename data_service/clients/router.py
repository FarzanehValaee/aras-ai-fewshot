from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any

from django.conf import settings
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from data_service.clients.base import DownstreamError, get_unauthenticated, http_timeout

def _default_chat_model() -> str:
    from repro_config import load as load_repro_config

    return str(load_repro_config()["chat_model"])


DEFAULT_CHAT_MODEL = _default_chat_model()
_model_override: contextvars.ContextVar[str] = contextvars.ContextVar(
    "data_chat_model", default=""
)


@contextmanager
def use_chat_model(name: str | None):
    token = _model_override.set(str(name or "").strip())
    try:
        yield
    finally:
        _model_override.reset(token)


class RouterClient:
    """Client for aras-ai-router (:8101). Does not modify Router."""

    def __init__(self) -> None:
        self.base_url = settings.ROUTER_BASE_URL.rstrip("/")
        self.token = settings.ROUTER_API_KEY

    def health(self) -> None:
        get_unauthenticated(
            f"{self.base_url}/api/v1/health/",
            "Router",
            not_ready_message="Router health check failed.",
        )

    def ready(self) -> None:
        get_unauthenticated(
            f"{self.base_url}/api/v1/ready/",
            "Router",
            not_ready_message="Router is not ready.",
        )

    def chat(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or _model_override.get() or DEFAULT_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        # Rayen thinking models reject or stall on a small max_tokens budget.
        if max_tokens is not None and int(max_tokens) >= 1024:
            kwargs["max_tokens"] = int(max_tokens)

        client = self._openai()
        try:
            completion = client.chat.completions.create(**kwargs)
        except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
            self._raise_router(exc)
        payload = completion.model_dump()
        if not isinstance(payload, dict):
            raise DownstreamError(
                "Router returned a malformed response.",
                service="Router",
            )
        return payload

    def _openai(self) -> OpenAI:
        if not self.token:
            raise DownstreamError(
                "Router API key is not configured.",
                service="Router",
                unavailable=True,
            )
        return OpenAI(
            api_key=self.token,
            base_url=f"{self.base_url}/api/v1",
            timeout=http_timeout(),
            max_retries=0,
        )

    @staticmethod
    def _raise_router(exc: Exception) -> None:
        if isinstance(exc, APITimeoutError):
            raise DownstreamError(
                "Router request timed out.",
                service="Router",
                timeout=True,
                unavailable=True,
            ) from exc
        if isinstance(exc, APIConnectionError):
            raise DownstreamError(
                "Router is unavailable.",
                service="Router",
                unavailable=True,
            ) from exc
        status_code = getattr(exc, "status_code", None)
        body = ""
        try:
            response = getattr(exc, "response", None)
            body = getattr(response, "text", "") or ""
        except Exception:
            body = ""
        detail = ""
        if status_code:
            detail += f" HTTP {status_code}"
        if body:
            detail += f": {body}"
        raise DownstreamError(
            f"Router request failed.{detail}".strip(),
            status_code=status_code,
            service="Router",
            unavailable=bool(status_code and status_code >= 500),
        ) from exc

    @staticmethod
    def _message_text(message: dict | None) -> str:
        if not isinstance(message, dict):
            return ""
        for key in ("content", "reasoning", "reasoning_content"):
            val = message.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if not isinstance(val, list):
                continue
            parts: list[str] = []
            for part in val:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            joined = "".join(parts).strip()
            if joined:
                return joined
        return ""

    @classmethod
    def extract_answer(cls, payload: dict) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                text = cls._message_text(message if isinstance(message, dict) else None)
                if text:
                    return text
                raw = first.get("text")
                if isinstance(raw, str) and raw.strip():
                    return raw
        for key in ("output_text", "answer"):
            raw = payload.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw
        raise DownstreamError(
            "Router returned a malformed response.",
            service="Router",
        )

    def complete(self, prompt: str, **kwargs) -> str:
        return self.extract_answer(self.chat(prompt=prompt, **kwargs))
