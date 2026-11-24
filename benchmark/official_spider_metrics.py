#!/usr/bin/env python3
"""Score SQL with the vendored official Spider evaluation.py.

Exact match is Evaluator.eval_exact_match. Execution accuracy is
eval_exec_match, which runs both statements on the Spider sqlite file.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

OFFICIAL_DIR = Path(__file__).resolve().parent / "spider_official"
_PREFIX = re.compile(r"\b[A-Za-z0-9_]+\.(?=[A-Za-z0-9_*])")
_LIMIT = re.compile(r"\s+LIMIT\s+\d+\s*;?\s*$", re.IGNORECASE)

_EMPTY_SQL: dict[str, Any] = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}

_official: tuple[Any, Any] | None = None


def _modules() -> tuple[Any, Any]:
    global _official
    if _official is not None:
        return _official
    nltk_dir = Path.home() / "nltk_data"
    if nltk_dir.is_dir():
        os.environ.setdefault("NLTK_DATA", str(nltk_dir))
    if str(OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DIR))
    import evaluation as spider_evaluation
    import process_sql

    _official = (spider_evaluation, process_sql)
    return _official


def to_spider_sql(sql: str) -> str:
    """Drop MySQL quoting, schema prefixes, and a trailing LIMIT."""
    text = (sql or "").replace("`", "")
    text = _PREFIX.sub("", text)
    text = _LIMIT.sub("", text)
    return text.strip()


def sqlite_db(db_id: str) -> Path | None:
    root = (os.environ.get("SPIDER_DB_DIR") or "").strip()
    if not root or not db_id:
        return None
    path = Path(root) / db_id / f"{db_id}.sqlite"
    return path if path.is_file() else None


def _parse(schema_map: dict[str, list[str]], sql: str) -> dict:
    _evaluation, process_sql = _modules()
    schema = process_sql.Schema(schema_map)
    return process_sql.get_sql(schema, sql), schema


def _rebuild(schema: Any, parsed: dict, kmap: dict | None) -> dict:
    evaluation, _process_sql = _modules()
    foreign = kmap or {}
    valid = evaluation.build_valid_col_units(parsed["from"]["table_units"], schema)
    rebuilt = evaluation.rebuild_sql_val(parsed)
    return evaluation.rebuild_sql_col(valid, rebuilt, foreign)


def score_sql(
    gold_sql: str,
    pred_sql: str,
    schema_map: dict[str, list[str]],
    *,
    db_id: str = "",
    kmap: dict | None = None,
) -> dict[str, int | None]:
    """Official exact match, and official execution accuracy when sqlite exists.

    Values are 0/1, or None when that official metric could not be run.
    """
    evaluation, _process_sql = _modules()
    gold = to_spider_sql(gold_sql)
    pred = to_spider_sql(pred_sql)
    exact: int | None = None
    official_ex: int | None = None
    error = ""
    gold_parsed = None
    pred_parsed = None
    schema = None
    if gold and schema_map:
        try:
            gold_parsed, schema = _parse(schema_map, gold)
            gold_parsed = _rebuild(schema, gold_parsed, kmap)
        except Exception as exc:
            error = str(exc).splitlines()[0][:240]
            gold_parsed = None
    if gold_parsed is not None and schema is not None:
        try:
            pred_parsed, _schema = _parse(schema_map, pred) if pred else (_EMPTY_SQL, schema)
            if pred:
                pred_parsed = _rebuild(schema, pred_parsed, kmap)
            else:
                pred_parsed = dict(_EMPTY_SQL)
        except Exception:
            pred_parsed = dict(_EMPTY_SQL)
        try:
            exact = int(
                evaluation.Evaluator().eval_exact_match(
                    copy.deepcopy(pred_parsed), copy.deepcopy(gold_parsed)
                )
            )
        except Exception as exc:
            exact = None
            error = error or str(exc).splitlines()[0][:240]
    db = sqlite_db(db_id)
    if db is not None and gold_parsed is not None and pred_parsed is not None:
        try:
            official_ex = int(
                bool(
                    evaluation.eval_exec_match(
                        str(db),
                        pred,
                        gold,
                        copy.deepcopy(pred_parsed),
                        copy.deepcopy(gold_parsed),
                    )
                )
            )
        except Exception as exc:
            official_ex = None
            error = error or str(exc).splitlines()[0][:240]
    return {
        "exact_match": exact,
        "official_ex": official_ex,
        "official_error": error,
    }


def summarize_metrics(rows: list[dict]) -> str:
    """Text block with homemade set-match EX and both official metrics."""

    def _rate(key: str) -> str:
        values = [row.get(key) for row in rows if row.get(key) in (0, 1)]
        if not values and key == "ex":
            values = [row.get("score") for row in rows if row.get("score") in (0, 1)]
        if not values:
            return "n/a"
        hit = sum(int(value) for value in values)
        return f"{hit}/{len(values)} ({100.0 * hit / len(values):.2f}%)"

    return "\n".join(
        [
            f"HOMEMADE_EX {_rate('ex')}",
            f"EXACT_SET_MATCH {_rate('exact_match')}",
            f"EXECUTION {_rate('official_ex')}",
        ]
    )


def attach_official(
    row: dict,
    *,
    gold_sql: str,
    pred_sql: str,
    schema_map: dict[str, list[str]],
    db_id: str,
) -> dict:
    """Write official exact_match and official_ex onto an eval row."""
    scored = score_sql(gold_sql, pred_sql, schema_map, db_id=db_id)
    row["exact_match"] = scored["exact_match"]
    row["official_ex"] = scored["official_ex"]
    if scored["official_error"]:
        row["official_error"] = scored["official_error"]
    return row


def validate_dump(path: Path, *, limit: int = 100) -> int:
    """Compare stored homemade EX with a fresh official score. Print disagreements."""
    from benchmark.eval_instrumentation import ddl_schema

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        homemade = row.get("ex") if row.get("ex") in (0, 1) else row.get("score")
        if homemade not in (0, 1):
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    checked: list[dict] = []
    sqlite_ready = 0
    for row in rows:
        db_id = str(row.get("db_id") or "")
        schema = ddl_schema(db_id)
        scored = score_sql(
            str(row.get("query") or row.get("gold_sql") or ""),
            str(row.get("pred_sql") or row.get("predicted_sql") or ""),
            schema,
            db_id=db_id,
        )
        merged = dict(row)
        merged["ex"] = row.get("ex") if row.get("ex") in (0, 1) else row.get("score")
        merged["exact_match"] = scored["exact_match"]
        merged["official_ex"] = scored["official_ex"]
        checked.append(merged)
        if scored["official_ex"] in (0, 1):
            sqlite_ready += 1
    print(f"validated {len(checked)} questions from {path}")
    print(summarize_metrics(checked))
    if sqlite_ready == 0:
        print(
            "official execution is unavailable "
            "(set SPIDER_DB_DIR to <db_id>/<db_id>.sqlite); "
            "no homemade-vs-official execution discrepancies to report"
        )
        return 0
    found = discrepancies(checked, limit=len(checked))
    if not found:
        print("no discrepancies")
        return 0
    print(f"discrepancies {len(found)}")
    for item in found:
        print(json.dumps(item, ensure_ascii=False))
    return len(found)


def discrepancies(
    rows: list[dict],
    *,
    limit: int = 100,
) -> list[dict]:
    """Rows where homemade EX and official execution accuracy disagree."""
    found: list[dict] = []
    for row in rows[:limit]:
        homemade = row.get("ex")
        if homemade not in (0, 1):
            homemade = row.get("score")
        official = row.get("official_ex")
        if homemade not in (0, 1) or official not in (0, 1):
            continue
        if int(homemade) == int(official):
            continue
        found.append(
            {
                "db_id": row.get("db_id"),
                "question": row.get("question"),
                "homemade_ex": int(homemade),
                "official_ex": int(official),
                "exact_match": row.get("exact_match"),
                "gold_sql": row.get("gold_sql") or row.get("query"),
                "predicted_sql": row.get("predicted_sql") or row.get("pred_sql"),
            }
        )
    return found
