#!/usr/bin/env python3
"""Per-question eval log. Does not choose examples or write SQL.

Official hardness is Spider's Evaluator.eval_hardness (evaluation.py) on the
parsed gold SQL. Skeletons and similarity scores are cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from repro_config import load as load_repro_config
from repro_config import public_config, seed_everything

ROOT = Path("/var/www/aras-ai-data")
OFFICIAL_DIR = Path(__file__).resolve().parent / "spider_official"
_REPRO = load_repro_config()
seed_everything(int(_REPRO["seed"]))
RESULTS_DIR = Path(_REPRO["paths"]["results_dir"])
CACHE_DIR = RESULTS_DIR / "cache"
EVAL_SEED = int(_REPRO["seed"])
EVAL_TEMPERATURE = float(_REPRO["temperature"]["eval"])
PRICE_IN = float(_REPRO["prices"]["usd_per_mtok_in"])
PRICE_OUT = float(_REPRO["prices"]["usd_per_mtok_out"])

_MISSING = object()
_tls = threading.local()
_installed = False
_sim_cache: "DiskCache | None" = None
_official: tuple[Any, Any] | None = None
_log_lock = threading.Lock()
_ddl_schemas: dict[str, dict[str, list[str]]] | None = None


class DiskCache:
    """Append-only JSONL cache. Keys are content hashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                key, sep, payload = line.partition("\t")
                if not sep:
                    continue
                try:
                    self.data[key] = json.loads(payload)
                except json.JSONDecodeError:
                    continue

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self.data:
                return _MISSING
            return self.data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self.data:
                return
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(key + "\t" + json.dumps(value, ensure_ascii=False) + "\n")
                handle.flush()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def begin_question() -> "StageClock":
    clock = StageClock()
    _tls.clock = clock
    _tls.calls = []
    _tls.trace = None
    _tls.inner_trace = None
    return clock


def current_trace() -> dict | None:
    return getattr(_tls, "trace", None)


def note_inner_trace(trace: dict) -> None:
    """Jaccard retrieval records its own comparisons. Selection is unchanged."""
    _tls.inner_trace = trace


class StageClock:
    def __init__(self) -> None:
        self.latency_ms: dict[str, float] = {}
        self._open: dict[str, float] = {}
        self.current = "llm"
        self.t0 = time.perf_counter()

    def __call__(self, stage: str, status: str, **kwargs: Any) -> None:
        del kwargs
        now = time.perf_counter()
        self.current = str(stage)
        if status == "running":
            self._open[self.current] = now
            return
        if status not in {"completed", "failed"}:
            return
        start = self._open.pop(self.current, None)
        if start is None:
            return
        delta = (now - start) * 1000.0
        self.latency_ms[self.current] = round(
            self.latency_ms.get(self.current, 0.0) + delta, 3
        )

    def total_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000.0, 3)


def _load_official() -> tuple[Any, Any]:
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


def get_hardness(parsed_sql: dict) -> str:
    """Official Spider hardness. `parsed_sql` is process_sql.get_sql output."""
    spider_evaluation, _process_sql = _load_official()
    label = spider_evaluation.Evaluator().eval_hardness(parsed_sql)
    if label not in {"easy", "medium", "hard", "extra"}:
        raise RuntimeError(f"official eval_hardness returned {label!r}")
    return label


def schema_from_catalog(catalog: Any) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    if not isinstance(catalog, dict):
        return schema
    for key, cols in catalog.items():
        table = str(key).split(".")[-1].strip().lower()
        if not table:
            continue
        names: list[str] = []
        if isinstance(cols, list):
            for column in cols:
                if isinstance(column, dict):
                    name = (
                        column.get("english_name")
                        or column.get("name")
                        or column.get("source_name")
                        or ""
                    )
                else:
                    name = column
                text = str(name or "").strip().lower()
                if text:
                    names.append(text)
        schema[table] = names
    return schema


_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+`([^`]+)`\s*\((.*?)\)\s*ENGINE=",
    re.IGNORECASE | re.DOTALL,
)
_CREATE_COL = re.compile(r"^\s*`([^`]+)`", re.MULTILINE)


def schema_from_create_sql(ddl: str) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    for match in _CREATE_TABLE.finditer(ddl or ""):
        table = match.group(1).strip().lower()
        cols = [col.strip().lower() for col in _CREATE_COL.findall(match.group(2))]
        schema[table] = cols
    return schema


def official_hardness(gold_sql: str, schema: dict[str, list[str]]) -> str:
    _spider_evaluation, process_sql = _load_official()
    parsed = process_sql.get_sql(process_sql.Schema(schema), gold_sql)
    return get_hardness(parsed)


def _catalog_fingerprint(catalog: Any) -> str:
    if not isinstance(catalog, dict):
        return ""
    parts: list[str] = []
    for key in sorted(catalog, key=lambda item: str(item)):
        cols = catalog[key]
        names: list[str] = []
        if isinstance(cols, list):
            for column in cols:
                if isinstance(column, dict):
                    names.append(
                        str(
                            column.get("english_name")
                            or column.get("name")
                            or column.get("source_name")
                            or ""
                        )
                    )
                else:
                    names.append(str(column))
        parts.append(str(key) + "=" + ",".join(names))
    return sha("\n".join(parts))


def _skeleton_to_json(skeleton: Any) -> Any:
    if skeleton is None:
        return None
    return {
        "band": skeleton.band,
        "tables": sorted(skeleton.tables),
        "join_count": skeleton.join_count,
        "aggs": sorted(skeleton.aggs),
        "agg_count": skeleton.agg_count,
        "where": skeleton.where,
        "group_by": skeleton.group_by,
        "having": skeleton.having,
        "order_by": skeleton.order_by,
        "cte": skeleton.cte,
        "subquery": skeleton.subquery,
    }


def _skeleton_from_json(payload: Any) -> Any:
    from data_service.agent.fewshot_skeleton import SqlSkeleton

    if payload is None:
        return None
    return SqlSkeleton(
        tables=frozenset(payload.get("tables") or []),
        join_count=int(payload.get("join_count") or 0),
        aggs=frozenset(payload.get("aggs") or []),
        agg_count=int(payload.get("agg_count") or 0),
        where=bool(payload.get("where")),
        group_by=bool(payload.get("group_by")),
        having=bool(payload.get("having")),
        order_by=bool(payload.get("order_by")),
        cte=bool(payload.get("cte")),
        subquery=bool(payload.get("subquery")),
    )


def install_runtime_cache() -> None:
    """Cache sql/question skeletons and force temperature=0 plus a fixed seed.

    The wrapped retriever still returns the original selection.
    """
    global _installed, _sim_cache
    if _installed:
        return
    _installed = True
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    skeleton_cache = DiskCache(CACHE_DIR / "skeletons.jsonl")
    _sim_cache = DiskCache(CACHE_DIR / "similarity.jsonl")

    import data_service.agent.fewshot_skeleton as skeletons
    from data_service.clients.router import RouterClient

    orig_sql = skeletons.sql_skeleton
    orig_question = skeletons.question_skeleton

    def cached_sql(sql: str):
        key = "sql:" + sha(str(sql or ""))
        hit = skeleton_cache.get(key)
        if hit is not _MISSING:
            return _skeleton_from_json(hit)
        value = orig_sql(sql)
        skeleton_cache.put(key, _skeleton_to_json(value))
        return value

    def cached_question(question: str, catalog: dict | None = None) -> str:
        key = "q:" + sha(str(question or "") + "\0" + _catalog_fingerprint(catalog))
        hit = skeleton_cache.get(key)
        if hit is not _MISSING:
            return str(hit)
        value = orig_question(question, catalog)
        skeleton_cache.put(key, value)
        return value

    skeletons.sql_skeleton = cached_sql
    skeletons.question_skeleton = cached_question

    orig_openai = RouterClient._openai

    def _openai(self):
        client = orig_openai(self)
        create = client.chat.completions.create

        def wrapped(**kwargs):
            kwargs["temperature"] = EVAL_TEMPERATURE
            kwargs["seed"] = EVAL_SEED
            started = time.perf_counter()
            response = create(**kwargs)
            _record_call(response, kwargs, (time.perf_counter() - started) * 1000.0)
            return response

        client.chat.completions.create = wrapped
        return client

    RouterClient._openai = _openai  # type: ignore[method-assign]


def _record_call(response: Any, kwargs: dict, latency_ms: float) -> None:
    calls = getattr(_tls, "calls", None)
    if calls is None:
        return
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    cost = None
    try:
        payload = response.model_dump()
    except Exception:
        payload = {}
    usage_payload = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage_payload, dict):
        for key in ("cost", "total_cost"):
            if usage_payload.get(key) is not None:
                cost = float(usage_payload[key])
                break
    if cost is None:
        cost = (prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT) / 1_000_000.0
    clock = getattr(_tls, "clock", None)
    stage = clock.current if clock is not None else "llm"
    same_stage = sum(1 for call in calls if call.get("stage") == stage)
    calls.append(
        {
            "stage": stage if same_stage == 0 else f"{stage}#{same_stage + 1}",
            "model": str(getattr(response, "model", None) or kwargs.get("model") or ""),
            "temperature": EVAL_TEMPERATURE,
            "seed": EVAL_SEED,
            "latency_ms": round(latency_ms, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
        }
    )


def install_retrieval_trace(retrieve) -> Any:
    """Call the real retriever, then record scores. The return value is its list."""

    def tracing(question, examples, *args, **kwargs):
        _tls.inner_trace = None
        picked = retrieve(question, examples, *args, **kwargs)
        inner = getattr(_tls, "inner_trace", None)
        if inner is not None:
            _tls.trace = inner
            return picked
        _tls.trace = _cached_skeleton_trace(
            question, examples, picked, kwargs
        )
        return picked

    return tracing


def _cached_skeleton_trace(question, examples, picked, kwargs) -> dict:
    assert _sim_cache is not None
    draft_sql = kwargs.get("draft_sql")
    catalog = kwargs.get("catalog")
    top_k = kwargs.get("top_k", 5)
    exclude_ids = kwargs.get("exclude_ids") or set()
    key = sha(
        json.dumps(
            {
                "q": question,
                "draft": draft_sql or "",
                "k": int(top_k),
                "exclude": sorted(int(i) for i in exclude_ids),
                "catalog": _catalog_fingerprint(catalog),
                "examples": [
                    [
                        getattr(ex, "id", None),
                        str(getattr(ex, "user_question", "") or ""),
                        str(getattr(ex, "ideal_sql", "") or ""),
                        int(getattr(ex, "priority", 100) or 100),
                    ]
                    for ex in examples
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    hit = _sim_cache.get(key)
    if hit is not _MISSING:
        return hit
    measured = measure_skeleton_retrieval(
        question,
        examples,
        draft_sql=draft_sql,
        catalog=catalog,
        top_k=int(top_k),
        exclude_ids=set(exclude_ids),
        picked=picked,
    )
    _sim_cache.put(key, measured)
    return measured


def measure_skeleton_retrieval(
    question: str,
    examples: list,
    *,
    draft_sql: str | None,
    catalog: dict | None,
    top_k: int,
    exclude_ids: set[int],
    picked: list,
) -> dict:
    """Replay the band filter only to count comparisons and score the picked rows.

    The picked list is the retriever's list. This function does not replace it.
    """
    from data_service.agent import fewshot_skeleton as skeletons

    draft = skeletons.sql_skeleton(draft_sql or "")
    trusted = bool(draft) and skeletons.draft_is_trusted(question, draft)
    q_skel = skeletons.question_skeleton(question, catalog)
    q_tables = set(skeletons.mentioned_tables(question, catalog))
    if draft is not None:
        q_tables |= set(draft.tables)
    if trusted and draft is not None:
        allowed = {draft.band}
    elif draft is not None and skeletons.hard_markers(question) and not trusted:
        allowed = {skeletons.MEDIUM, skeletons.HARD}
    else:
        allowed = None

    scanned_all = 0
    scanned_filtered = 0
    score_by_id: dict[int, list] = {}
    for ex in examples:
        ex_id = getattr(ex, "id", None)
        if ex_id is not None and ex_id in exclude_ids:
            continue
        sql = (getattr(ex, "ideal_sql", None) or "").strip()
        uq = (getattr(ex, "user_question", None) or "").strip()
        if not sql or not uq:
            continue
        skeleton = skeletons.sql_skeleton(sql)
        if skeleton is None:
            continue
        scanned_all += 1
        if allowed is not None and skeleton.band not in allowed:
            continue
        scanned_filtered += 1
        table_overlap = len(skeleton.tables & q_tables)
        same_pattern = bool(q_skel) and skeletons.question_skeleton(uq, catalog) == q_skel
        priority = int(getattr(ex, "priority", 100) or 100)
        feature = (
            skeletons._draft_feature_score(draft, skeleton) if trusted and draft else 0
        )
        if trusted:
            key = [table_overlap, feature, 1 if same_pattern else 0, -priority]
        else:
            key = [1 if same_pattern else 0, table_overlap, -priority]
        if ex_id is not None:
            score_by_id[int(ex_id)] = key

    ids = [
        int(ex.id)
        for ex in picked
        if getattr(ex, "id", None) is not None
    ]
    return {
        "predicted_class": draft.band if draft is not None else None,
        "skeleton_predicted": _skeleton_to_json(draft),
        "selected_fewshot_ids": ids,
        "similarity_scores": [score_by_id.get(ex_id) for ex_id in ids],
        "num_candidates_scanned": {
            "with_class_filter": scanned_filtered,
            "without_class_filter": scanned_all,
        },
    }


def run_log_path(strategy: str, model: str, when: datetime | None = None) -> Path:
    day = (when or datetime.now()).strftime("%Y%m%d")
    safe = re.compile(r"[^A-Za-z0-9._-]+")
    name = (
        f"run_{safe.sub('_', strategy)}_{safe.sub('_', model)}_{day}.jsonl"
    )
    return RESULTS_DIR / name


def append_question_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _log_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def ddl_schema(db_id: str) -> dict[str, list[str]]:
    global _ddl_schemas
    if _ddl_schemas is None:
        path = Path(load_repro_config()["paths"]["schema_cache"])
        _ddl_schemas = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key, ddl in payload.items():
                    if isinstance(ddl, str):
                        _ddl_schemas[str(key)] = schema_from_create_sql(ddl)
    return _ddl_schemas.get(str(db_id), {})


def build_record(
    *,
    strategy: str,
    model: str,
    item: dict,
    row: dict,
    schema: dict[str, list[str]] | None,
    clock: StageClock | None,
) -> dict:
    gold_sql = str(item.get("query") or item.get("gold_sql") or row.get("gold_sql") or "")
    predicted_sql = str(row.get("pred_sql") or "")
    score = row.get("score")
    ex = 1 if score == 1 else 0
    hardness = None
    hardness_error = None
    parsed = item.get("sql") if isinstance(item.get("sql"), dict) else None
    schemas: list[dict[str, list[str]]] = []
    if schema:
        schemas.append(schema)
    ddl = ddl_schema(str(item.get("db_id") or row.get("db_id") or ""))
    if ddl and ddl not in schemas:
        schemas.append(ddl)
    try:
        if parsed and "select" in parsed:
            hardness = get_hardness(parsed)
        elif gold_sql:
            last_exc: Exception | None = None
            for candidate in schemas:
                try:
                    hardness = official_hardness(gold_sql, candidate)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if hardness is None and last_exc is not None:
                raise last_exc
            if hardness is None and not schemas:
                raise RuntimeError("no schema available for official get_sql")
    except Exception as exc:
        hardness_error = str(exc).splitlines()[0][:300]
    trace = current_trace() or {}
    skeleton = trace.get("skeleton_predicted")
    if skeleton is None and predicted_sql:
        from data_service.agent import fewshot_skeleton as skeletons

        skeleton = _skeleton_to_json(skeletons.sql_skeleton(predicted_sql))
    predicted_class = trace.get("predicted_class")
    if predicted_class is None:
        predicted_class = row.get("fewshot_draft_band")
    scanned = trace.get("num_candidates_scanned") or {
        "with_class_filter": 0,
        "without_class_filter": 0,
    }
    record = {
        "db_id": item.get("db_id") or row.get("db_id"),
        "question": item.get("question") or row.get("question"),
        "gold_sql": gold_sql,
        "predicted_sql": predicted_sql,
        "ex": ex,
        "official_hardness": hardness,
        "predicted_class": predicted_class,
        "skeleton_predicted": skeleton,
        "selected_fewshot_ids": trace.get("selected_fewshot_ids")
        if "selected_fewshot_ids" in trace
        else list(row.get("fewshot_ids") or []),
        "similarity_scores": trace.get("similarity_scores") or [],
        "num_candidates_scanned": scanned,
        "stage_latency_ms": dict(clock.latency_ms) if clock is not None else {},
        "total_latency_ms": clock.total_ms() if clock is not None else None,
        "llm_calls": list(getattr(_tls, "calls", None) or []),
        "model": model,
        "temperature": EVAL_TEMPERATURE,
        "seed": EVAL_SEED,
        "strategy": strategy,
        "selection_strategy": strategy,
        "exact_match": row.get("exact_match"),
        "official_ex": row.get("official_ex"),
        **public_config(),
    }
    if hardness_error:
        record["official_hardness_error"] = hardness_error
    if row.get("official_error"):
        record["official_error"] = row["official_error"]
    return record
