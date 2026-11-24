"""Lightweight question planner via Router (adapted from Numbers question_planner)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from data_service.clients.base import DownstreamError
from data_service.clients.router import RouterClient

logger = logging.getLogger(__name__)


@dataclass
class QuestionPlan:
    original_question: str
    resolved_question: str
    corrections: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    used_heuristic: bool = False


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = _strip_json_fence(text)
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(raw[start : i + 1])
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def plan_question(
    question: str,
    *,
    schema: str,
    history: str = "",
    router: RouterClient | None = None,
) -> QuestionPlan:
    q = (question or "").strip()
    fallback = QuestionPlan(
        original_question=q,
        resolved_question=q,
        used_heuristic=True,
    )
    if not q:
        return fallback

    prompt = (
        "You plan analytics questions before SQL generation. "
        "Return ONLY JSON with keys: resolved_question, corrections, ambiguities, "
        "assumptions, plan_steps (arrays of strings). "
        "Fix obvious typos/metric names using the schema. "
        "Do not invent tables/columns not in the schema. "
        "Keep resolved_question in natural English. "
        "Do not translate SQL filter values. "
        "Do not replace the user's words with english_name identifiers; "
        "record any column mapping in assumptions or plan_steps.\n\n"
        f"Schema:\n{schema[:5000]}\n\n"
        f"History:\n{history or '(none)'}\n\n"
        f"Question:\n{q}\n"
    )
    client = router or RouterClient()
    try:
        from repro_config import load as load_repro_config

        text = client.complete(
            prompt,
            temperature=float(load_repro_config()["temperature"]["planner"]),
            max_tokens=500,
        )
    except DownstreamError as exc:
        logger.warning("planner router failed: %s", exc.message)
        return fallback

    data = _extract_json_object(text)
    if not data:
        return fallback

    resolved = str(data.get("resolved_question") or q).strip() or q

    def _list(key: str) -> list[str]:
        val = data.get(key) or []
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        return []

    return QuestionPlan(
        original_question=q,
        resolved_question=resolved,
        corrections=_list("corrections"),
        ambiguities=_list("ambiguities"),
        assumptions=_list("assumptions"),
        plan_steps=_list("plan_steps"),
        used_heuristic=False,
    )
