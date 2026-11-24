"""Detect ranking direction (top vs bottom) from the user question, not a default."""

from __future__ import annotations

import re
from typing import Any

# Local plan duck-type (avoids aras-ai-app import). Full QuestionPlan deferred.
QuestionPlan = Any

# DESC = highest / best-selling. ASC = lowest / worst / "bad status".
_DESC_RE = re.compile(
    r"best[\s-]*sell|\bhighest\b|\btop\b|\bbest\b|good\s+status",
    re.I,
)
_ASC_RE = re.compile(
    r"worst[\s-]*sell|\blowest\b|\bbottom\b|\bworst\b|bad\s+status|\bweak\b",
    re.I,
)
_NEUTRAL_RANK_RE = re.compile(
    r"\brank\b|second|third",
    re.I,
)
_STORE_RE = re.compile(r"store|branch|shop", re.I)
_PRODUCT_RE = re.compile(r"product|articles|sku|\bitem\b", re.I)
_BREAKDOWN_RE = re.compile(
    r"grouped by|breakdown|\bper\s+(store|product|day|month)",
    re.I,
)
_SALES_RE = re.compile(r"sales|amount|total|\bsum\b", re.I)
_AVG_RE = re.compile(r"average|\bavg\b|\bmean\b", re.I)
# "how many" must not match inside unrelated words.
_COUNT_RE = re.compile(r"\bcount\b|how many|number of|transaction|record", re.I)


def mentions_store(text: str) -> bool:
    return bool(_STORE_RE.search(text or ""))


def mentions_product(text: str) -> bool:
    return bool(_PRODUCT_RE.search(text or ""))


def mentions_ranking(text: str) -> bool:
    blob = text or ""
    return bool(_DESC_RE.search(blob) or _ASC_RE.search(blob) or _NEUTRAL_RANK_RE.search(blob))


def rank_direction(text: str) -> str | None:
    """
    Return ``"asc"``, ``"desc"``, or None.

    Original-question text must win over planner/few-shot wording: if the user
    said "bad status", that is ASC even if some other string mentions best-selling.
    When both directions appear in the same string, ASC wins (the harmful
    default used to be DESC).
    """
    blob = text or ""
    has_asc = bool(_ASC_RE.search(blob))
    has_desc = bool(_DESC_RE.search(blob))
    if has_asc:
        return "asc"
    if has_desc:
        return "desc"
    return None


def question_rank_direction(question: str, plan: QuestionPlan | None = None) -> str | None:
    """Prefer the user's wording; only fall back to the plan if the question is silent."""
    from_q = rank_direction(question or "")
    if from_q:
        return from_q
    if not plan:
        return None
    parts = [
        getattr(plan, "resolved_question", "") or "",
        " ".join(getattr(plan, "plan_steps", None) or []),
        " ".join(getattr(plan, "assumptions", None) or []),
        " ".join(getattr(plan, "corrections", None) or []),
    ]
    return rank_direction(" ".join(parts))


def wants_ranking(question: str, plan: QuestionPlan | None = None) -> bool:
    """True only when the *user question* asks to rank. Planner text is ignored."""
    del plan
    return mentions_ranking(question or "")


def wants_breakdown(question: str) -> bool:
    """True when the user asked for a per-entity split, not a ranking."""
    return bool(_BREAKDOWN_RE.search(question or ""))


def wants_sales_metric(question: str) -> bool:
    return bool(_SALES_RE.search(question or ""))


def wants_average(question: str) -> bool:
    return bool(_AVG_RE.search(question or ""))


def wants_count(question: str) -> bool:
    return bool(_COUNT_RE.search(question or ""))


def allows_grouped_ranking_sql(question: str, plan: QuestionPlan | None = None) -> bool:
    """GROUP BY store/product ranking results belong in the answer only if asked."""
    return wants_ranking(question, plan) or wants_breakdown(question or "")


def sql_order_direction(sql: str) -> str | None:
    """Last ORDER BY direction in ``sql``. Bare ORDER BY is ASC (SQL default)."""
    sql_l = (sql or "").lower()
    matches = list(re.finditer(r"order\s+by\b([^;]*)", sql_l))
    if not matches:
        return None
    tail = matches[-1].group(1)
    dirs = re.findall(r"\b(asc|desc)\b", tail)
    if dirs:
        return dirs[0]
    return "asc"


def ranking_label(entity: str, direction: str | None) -> str:
    """Human heading for a grouped ranking result."""
    if entity == "store":
        if direction == "asc":
            return "Lowest-selling stores:"
        if direction == "desc":
            return "Top-selling stores:"
        return "Sales by store:"
    if entity == "product":
        if direction == "asc":
            return "Lowest-selling products:"
        if direction == "desc":
            return "Top-selling products:"
        return "Sales by product:"
    return "Grouped results:"


def ranking_chart_title(entity: str, direction: str | None) -> str:
    label = ranking_label(entity, direction).rstrip(":")
    return label


def order_by_clause(
    direction: str | None,
    column: str = "total_sales",
    *,
    ranking: bool = True,
) -> str:
    """SQL ORDER BY fragment. No ORDER BY for a plain breakdown."""
    if direction == "asc":
        return f"ORDER BY {column} ASC"
    if direction == "desc" or ranking:
        return f"ORDER BY {column} DESC"
    return ""


def supplement_note(entity: str, direction: str | None) -> str:
    if entity == "store":
        if direction == "asc":
            return "Follow-up query: 5 lowest-selling stores last week"
        if direction == "desc":
            return "Follow-up query: rank the 5 top stores last week"
        return "Follow-up query: sales by store"
    if direction == "asc":
        return "Follow-up query: 5 lowest-selling products last week"
    if direction == "desc":
        return "Follow-up query: 5 top-selling products last week"
    return "Follow-up query: sales by product"


_DESC_FLIP_RE = re.compile(
    r"best[\s-]*sell[\w]*|\bhighest\b|\btop\s*sellers?\b",
    re.I,
)


def sanitize_rewritten_question(original: str, rewritten: str) -> str:
    """
    Keep ranking direction of the user question if an LLM rewrite flipped it.

    "bad status" must not become "best-selling". A non-ranking question must not
    gain a top-N ranking just because that pattern is common in few-shots.
    """
    original = (original or "").strip()
    text = (rewritten or "").strip() or original
    if not original:
        return text

    orig_dir = rank_direction(original)
    new_dir = rank_direction(text)
    orig_l = original.casefold()
    text_l = (text or "").casefold()

    if orig_dir == "asc" and new_dir != "asc":
        text = _DESC_FLIP_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip(" —-")
        if rank_direction(text) != "asc":
            text = f"{text} — lowest-selling / lowest value".strip(" —")
        if "why" in orig_l and "why" not in (text or "").casefold():
            text = f"{text} why".strip()
        return text or original

    if orig_dir == "desc" and new_dir == "asc":
        return original

    if orig_dir is None and not wants_ranking(original) and new_dir == "desc":
        text = _DESC_FLIP_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip(" —-")
        return text or original

    if "why" in orig_l and "why" not in text_l:
        text = f"{text} why".strip()

    return text or original
