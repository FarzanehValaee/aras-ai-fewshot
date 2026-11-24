"""Few-shot selection strategies for the ablation study.

Every strategy sees the same pool and the same k. Pool rows are not created
or rewritten here. ``full`` is the router: keep the predicted skeleton class,
then rank by skeleton similarity. The other four change only that choice.
"""

from __future__ import annotations

import random
import threading
from typing import Any

from repro_config import load as load_repro_config
from data_service.agent.fewshot_skeleton import (
    HARD,
    MEDIUM,
    _draft_feature_score,
    draft_is_trusted,
    hard_markers,
    mentioned_tables,
    question_skeleton,
    sql_skeleton,
)

STRATEGIES = (
    "full",
    "global_skeleton",
    "class_only_random",
    "random",
    "gold_class_oracle",
)
OFFICIAL_LABELS = frozenset({"easy", "medium", "hard", "extra"})

_tls = threading.local()


def set_question_gold_hardness(label: str | None) -> None:
    """Official gold hardness for the question currently being selected."""
    _tls.gold_hardness = label


def question_gold_hardness() -> str | None:
    label = getattr(_tls, "gold_hardness", None)
    if isinstance(label, str) and label.strip().lower() in OFFICIAL_LABELS:
        return label.strip().lower()
    return None


def normalize_strategy(name: str | None) -> str:
    label = (name or "full").strip().lower().replace("-", "_")
    if label not in STRATEGIES:
        allowed = ", ".join(STRATEGIES)
        raise ValueError(f"unknown selection_strategy {name!r}; expected one of {allowed}")
    return label


def predicted_classes(question: str, draft_sql: str | None) -> set[str] | None:
    """Router class for this draft. None means the draft did not name a class."""
    draft = sql_skeleton(draft_sql or "")
    trusted = bool(draft) and draft_is_trusted(question, draft)
    if trusted and draft is not None:
        return {draft.band}
    if draft is not None and hard_markers(question) and not trusted:
        return {MEDIUM, HARD}
    return None


def example_class(example: Any, skeleton: Any) -> str:
    """Official label when the row carries one, otherwise its skeleton band."""
    for attr in ("official_hardness", "hardness"):
        raw = getattr(example, attr, None)
        if isinstance(raw, str) and raw.strip().lower() in OFFICIAL_LABELS:
            return raw.strip().lower()
    return skeleton.band


def select_fewshots(
    question: str,
    examples: list,
    *,
    strategy: str = "full",
    k: int | None = None,
    seed: int | None = None,
    draft_sql: str | None = None,
    catalog: dict | None = None,
    exclude_ids: set[int] | None = None,
    gold_hardness: str | None = None,
) -> list:
    """Return up to ``k`` pool rows. Does not build or modify the pool."""
    cfg = load_repro_config()
    if k is None:
        k = int(cfg["k"])
    if seed is None:
        seed = int(cfg["seed"])
    name = normalize_strategy(strategy)
    limit = max(0, min(int(cfg["k_cap"]), int(k)))
    gold = _clean_gold(gold_hardness if gold_hardness is not None else question_gold_hardness())
    classes = predicted_classes(question, draft_sql)
    ranked, scanned_all = _score_pool(
        question,
        examples,
        draft_sql=draft_sql,
        catalog=catalog,
        exclude_ids=exclude_ids,
    )
    if name == "global_skeleton":
        picked = [row[1] for row in ranked[:limit]]
        allowed_n = scanned_all
    elif name == "random":
        picked = _sample([row[1] for row in ranked], limit, seed, question, name)
        allowed_n = scanned_all
    elif name == "class_only_random":
        filtered = [row for row in ranked if _in_class(row, classes, oracle=None)]
        picked = _sample([row[1] for row in filtered], limit, seed, question, name)
        allowed_n = len(filtered)
    elif name == "gold_class_oracle":
        filtered = [row for row in ranked if _in_class(row, None, oracle=gold)]
        picked = [row[1] for row in filtered[:limit]]
        allowed_n = len(filtered)
    else:
        filtered = [row for row in ranked if _in_class(row, classes, oracle=None)]
        picked = _take_ranked(filtered, limit)
        allowed_n = len(filtered)
    _tls.meta = {
        "selection_strategy": name,
        "predicted_class": _class_label(classes),
        "gold_hardness": gold,
        "selected_fewshot_ids": [
            int(ex.id) for ex in picked if getattr(ex, "id", None) is not None
        ],
        "similarity_scores": _scores_for(picked, ranked)
        if name in {"full", "global_skeleton", "gold_class_oracle"}
        else [None] * len(picked),
        "num_candidates_scanned": {
            "with_class_filter": allowed_n,
            "without_class_filter": scanned_all,
        },
        "skeleton_predicted": _draft_public(draft_sql),
    }
    return picked


def last_selection_meta() -> dict | None:
    meta = getattr(_tls, "meta", None)
    return dict(meta) if isinstance(meta, dict) else None


def _clean_gold(label: str | None) -> str | None:
    if isinstance(label, str) and label.strip().lower() in OFFICIAL_LABELS:
        return label.strip().lower()
    return None


def _class_label(classes: set[str] | None) -> str | None:
    if not classes:
        return None
    if classes == {MEDIUM, HARD}:
        return "medium+hard"
    if len(classes) == 1:
        return next(iter(classes))
    return "+".join(sorted(classes))


def _draft_public(draft_sql: str | None) -> dict | None:
    draft = sql_skeleton(draft_sql or "")
    if draft is None:
        return None
    return {
        "band": draft.band,
        "tables": sorted(draft.tables),
        "join_count": draft.join_count,
        "aggs": sorted(draft.aggs),
        "agg_count": draft.agg_count,
        "where": draft.where,
        "group_by": draft.group_by,
        "having": draft.having,
        "order_by": draft.order_by,
        "cte": draft.cte,
        "subquery": draft.subquery,
    }


def _in_class(row: tuple, classes: set[str] | None, oracle: str | None) -> bool:
    _key, _ex, skeleton, _overlap, _pattern, label = row
    if oracle is not None:
        return label == oracle
    if classes is None:
        return True
    return skeleton.band in classes


def _sample(rows: list, k: int, seed: int, question: str, strategy: str) -> list:
    ordered = sorted(
        rows,
        key=lambda ex: (
            int(getattr(ex, "id", 0) or 0),
            str(getattr(ex, "ideal_sql", "") or ""),
        ),
    )
    if k <= 0 or not ordered:
        return []
    if k >= len(ordered):
        return ordered
    rng = random.Random(f"{int(seed)}:{strategy}:{question}")
    return rng.sample(ordered, k)


def _take_ranked(rows: list[tuple], limit: int) -> list:
    """Same post-filter as the router: on-table first, then one skeleton band."""
    if not rows or limit <= 0:
        return []
    on_table = [row for row in rows if row[3] > 0]
    if on_table:
        pool = on_table
    else:
        off = [row for row in rows if row[4]]
        pool = off[:1]
    if not pool:
        return []
    cfg = load_repro_config()
    cap = limit
    raw_bands = {row[2].band for row in pool[: max(cap, int(cfg["k_cap"]))]}
    if len(raw_bands) > 1:
        cap = min(cap, int(cfg["mixed_band_cap"]))
    winner = pool[0][2].band
    picked: list = []
    for _key, ex, skeleton, _overlap, _pattern, _label in pool:
        if skeleton.band != winner:
            continue
        picked.append(ex)
        if len(picked) >= cap:
            break
    return picked


def _scores_for(picked: list, ranked: list[tuple]) -> list:
    by_id = {
        int(ex.id): list(key)
        for key, ex, _skeleton, _overlap, _pattern, _label in ranked
        if getattr(ex, "id", None) is not None
    }
    return [
        by_id.get(int(ex.id)) if getattr(ex, "id", None) is not None else None
        for ex in picked
    ]


def _score_pool(
    question: str,
    examples: list,
    *,
    draft_sql: str | None,
    catalog: dict | None,
    exclude_ids: set[int] | None,
) -> tuple[list[tuple], int]:
    """Score every usable pool row. Callers apply the class filter."""
    skip = exclude_ids or set()
    draft = sql_skeleton(draft_sql or "")
    trusted = bool(draft) and draft_is_trusted(question, draft)
    q_skel = question_skeleton(question, catalog)
    q_tables = set(mentioned_tables(question, catalog))
    if draft is not None:
        q_tables |= set(draft.tables)
    scored: list[tuple] = []
    for ex in examples:
        ex_id = getattr(ex, "id", None)
        if ex_id is not None and ex_id in skip:
            continue
        sql = (getattr(ex, "ideal_sql", None) or "").strip()
        uq = (getattr(ex, "user_question", None) or "").strip()
        if not sql or not uq:
            continue
        skeleton = sql_skeleton(sql)
        if skeleton is None:
            continue
        table_overlap = len(skeleton.tables & q_tables)
        same_pattern = bool(q_skel) and question_skeleton(uq, catalog) == q_skel
        priority = int(getattr(ex, "priority", 100) or 100)
        feature = _draft_feature_score(draft, skeleton) if trusted and draft else 0
        if trusted:
            key = (table_overlap, feature, 1 if same_pattern else 0, -priority)
        else:
            key = (1 if same_pattern else 0, table_overlap, -priority)
        scored.append(
            (key, ex, skeleton, table_overlap, same_pattern, example_class(ex, skeleton))
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored, len(scored)
