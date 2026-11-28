"""NL → plan → SQL → guard → execute → ground → chart pipeline.

Adapted from aras-ai-app Numbers agent (flow/runner/sql_guard/grounding/charts).
LLM calls go exclusively through aras-ai-router :8101.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from django.conf import settings

from data_service.agent.charts import build_chart_spec_from_table, wants_chart
from data_service.agent.config import normalize_dialect
from data_service.agent.fewshot_skeleton import draft_is_trusted, sql_skeleton
from data_service.agent.glossary import ColumnView
from data_service.agent.icrl import retrieve_similar
from data_service.agent.planner import plan_question
from data_service.agent.ranking_intent import question_rank_direction, wants_ranking
from data_service.agent.schema_slice import slice_schema
from data_service.agent.sql_guard import (
    SQL_GENERATION_RULES,
    _extract_from_tables,
    guard_sql_query,
    strip_default_limit_100,
)
from data_service.clients.base import DownstreamError
from data_service.clients.router import RouterClient
from data_service.models import ConnectionStatus, DataSource, DataSourceType, FewShotExample
from data_service.services.connections import (
    execute_readonly_sql,
    introspect_schema,
    probe_connection,
    source_to_db_config,
)

logger = logging.getLogger(__name__)


def notify(progress, *args, **kwargs) -> None:
    """Optional stage callback. No-op when the caller does not pass one."""
    if progress is None:
        return
    try:
        progress(*args, **kwargs)
    except Exception:
        logger.debug("progress callback failed", exc_info=True)


def safe_label(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_STATEMENT = re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL)
_PHRASE_MAX = 78


def _history_lines(history: list | None) -> str:
    if not history:
        return ""
    lines = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    raw = str(text or "").strip()
    fence = _SQL_FENCE.search(raw)
    if fence:
        return fence.group(1).strip()
    for line_group in re.split(r"\n{2,}", raw):
        candidate = line_group.strip()
        if candidate.upper().startswith("SELECT") or candidate.upper().startswith("WITH"):
            return candidate
    return raw


def merge_schema_description(user_notes: str, auto_desc: str) -> str:
    """Combine operator notes with introspected table list (aras-ai-app parity)."""
    user_notes = (user_notes or "").strip()
    auto_desc = (auto_desc or "").strip()
    marker = "Linked database tables:"
    if user_notes and marker in user_notes:
        user_notes = user_notes.split(marker, 1)[0].strip()
    if not user_notes:
        return auto_desc
    if not auto_desc:
        return user_notes
    return f"{user_notes}\n\n{auto_desc}".strip()


def _schema_context(source: DataSource, view: ColumnView | None = None) -> str:
    parts = []
    description = (source.description or "").strip()
    if description:
        parts.append(f"Business context:\n{description}")
    meta = source.meta or {}
    user_notes = str(meta.get("user_schema_description") or "").strip()
    schema_text = (source.schema_description or "").strip()
    if user_notes and user_notes not in schema_text:
        schema_text = merge_schema_description(user_notes, schema_text)
    if schema_text:
        parts.append(schema_text)
    glossary = (view or ColumnView.from_catalog(source.column_catalog)).prompt
    if glossary:
        parts.append(glossary)
    tables = meta.get("tables") or meta.get("created_tables") or []
    if tables:
        parts.append("Tables: " + json.dumps(tables, ensure_ascii=False)[:2000])
    if source.db_schema:
        parts.append(f"MySQL schema: {source.db_schema}")
    return "\n\n".join(parts) or "(no schema description)"


def _allowed_tables(source: DataSource) -> set[str] | None:
    """Restrict SQL to tables recorded for this source.

    Uploads always have this list after a successful import. Linked databases
    are restricted only when introspection already stored ``meta.tables`` or
    ``column_catalog``. A link with no catalog stays unrestricted so a query
    is not blocked before the schema has been read.
    """
    names: set[str] = set()
    meta = source.meta or {}
    for item in meta.get("tables") or []:
        if isinstance(item, dict) and item.get("table"):
            t = str(item["table"])
            names.add(t.split(".")[-1].lower())
        elif isinstance(item, str):
            names.add(item.split(".")[-1].lower())
    for key in (source.column_catalog or {}):
        names.add(str(key).split(".")[-1].lower())
    return names or None


def _sql_generation_prompt(
    *,
    question: str,
    schema: str,
    history: str,
    dialect: str,
    ranking_hint: str,
    plan_text: str,
    fewshot_block: str = "",
) -> str:
    return (
        "You are a careful analytics SQL assistant. "
        "Generate ONE read-only SQL query (SELECT or WITH…SELECT only). "
        "Never use INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE/GRANT/REVOKE. "
        f"Dialect: {dialect}.\n"
        f"{SQL_GENERATION_RULES}\n"
        f"{ranking_hint}"
        "Return ONLY the SQL, no explanation.\n\n"
        f"Plan:\n{plan_text or '(none)'}\n\n"
        f"Schema:\n{schema}\n\n"
        f"{fewshot_block}"
        f"Conversation history:\n{history or '(none)'}\n\n"
        f"Question:\n{question}\n"
    )


def _grounding_prompt(
    *,
    question: str,
    sql: str,
    columns: list,
    rows: list,
    units: dict | None = None,
) -> str:
    preview = json.dumps(
        {"columns": columns, "rows": rows[:30]}, ensure_ascii=False, default=str
    )
    unit_block = ""
    if units:
        unit_block = (
            "Column units (write the unit beside the number; do not convert the number):\n"
            + json.dumps(units, ensure_ascii=False)
            + "\n\n"
        )
    return (
        "You are an analytics answer grounding assistant. "
        "Answer the user question in English using ONLY the SQL result evidence below. "
        "Use the column labels from the result, not SQL identifiers. "
        "Do not change numeric values. "
        "Do not translate SQL filter literals or stored values. "
        "If the result is empty, say no matching data was found. "
        "Do not invent numbers.\n\n"
        f"Question: {question}\n"
        f"{unit_block}"
        f"SQL: {sql}\n"
        f"Result JSON: {preview}\n"
    )


def _fewshot_top_k() -> int:
    """Prompt few-shots. Cap and default come from configs/repro.yaml."""
    from repro_config import load as load_repro_config

    cfg = load_repro_config()
    return max(0, min(int(cfg["k_cap"]), int(getattr(settings, "FEWSHOT_RAG_TOP_K", cfg["k"]))))


def _list_active_fewshots(source: DataSource) -> list[FewShotExample]:
    if not getattr(source, "pk", None):
        return []
    return list(
        FewShotExample.objects.filter(data_source=source, is_active=True).order_by(
            "priority"
        )
    )


def _select_fewshots(
    source: DataSource,
    question: str,
    *,
    examples: list[FewShotExample] | None = None,
    draft_sql: str | None = None,
    exclude: FewShotExample | None = None,
) -> list[FewShotExample]:
    pool = examples if examples is not None else _list_active_fewshots(source)
    if not pool:
        return []
    catalog = source.column_catalog if isinstance(source.column_catalog, dict) else {}
    return retrieve_similar(
        question,
        pool,
        top_k=_fewshot_top_k(),
        exclude_ids={exclude.id} if exclude and exclude.id else set(),
        draft_sql=draft_sql,
        catalog=catalog,
    )


def _fewshot_tables_from(examples: list[FewShotExample]) -> set[str]:
    names: set[str] = set()
    for example in examples:
        names |= _extract_from_tables(example.ideal_sql or "")
    return names


def _format_fewshot_block(examples: list[FewShotExample]) -> tuple[str, int]:
    if not examples:
        return ("", 0)
    lines = ["Retrieved few-shot examples (follow their SQL style):"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"{i}. Q: {(ex.user_question or '').strip()}")
        sql = strip_default_limit_100((ex.ideal_sql or "").strip())
        lines.append(f"   SQL: {sql}")
    return ("\n".join(lines) + "\n\n", len(examples))


def _fewshot_match(source: DataSource, question: str) -> FewShotExample | None:
    q = (question or "").strip()
    if not q:
        return None
    for ex in FewShotExample.objects.filter(
        data_source=source, is_active=True
    ).order_by("priority"):
        if (ex.user_question or "").strip() == q and (ex.ideal_sql or "").strip():
            return ex
    return None


def _plan_text(plan) -> str:
    if not plan or getattr(plan, "used_heuristic", True):
        return ""
    return "\n".join(
        [
            *(f"step: {s}" for s in plan.plan_steps),
            *(f"assume: {a}" for a in plan.assumptions),
        ]
    )


def _complete_sql(
    router: RouterClient,
    *,
    question: str,
    schema: str,
    history: str,
    dialect: str,
    ranking_hint: str,
    plan_text: str,
    fewshot_block: str = "",
) -> str:
    prompt = _sql_generation_prompt(
        question=question,
        schema=schema,
        history=history,
        dialect=dialect,
        ranking_hint=ranking_hint,
        plan_text=plan_text,
        fewshot_block=fewshot_block,
    )
    from repro_config import load as load_repro_config

    return _extract_sql(
        router.complete(
            prompt,
            temperature=float(load_repro_config()["temperature"]["sql"]),
            max_tokens=800,
        )
    )


def _sql_repair_prompt(
    *,
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
    dialect: str,
) -> str:
    """Execution-guided repair: fix SQL that failed at the database."""
    return (
        "You are a careful analytics SQL assistant. "
        "The previous read-only SQL query failed when executed against the database. "
        "Produce a corrected ONE read-only SQL query (SELECT or WITH…SELECT only). "
        "Never use INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE/GRANT/REVOKE. "
        f"Dialect: {dialect}.\n"
        f"{SQL_GENERATION_RULES}\n"
        "Fix the exact execution error. Stay faithful to the user question and schema. "
        "Output ONLY the corrected SQL inside a markdown code block "
        "(```sql ... ```). No explanation.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question:\n{question}\n\n"
        f"Failed SQL:\n{failed_sql}\n\n"
        f"Database error:\n{error_message}\n"
    )


def _repair_sql(
    router: RouterClient,
    *,
    question: str,
    schema: str,
    failed_sql: str,
    error_message: str,
    dialect: str,
) -> str:
    prompt = _sql_repair_prompt(
        question=question,
        schema=schema,
        failed_sql=failed_sql,
        error_message=error_message,
        dialect=dialect,
    )
    from repro_config import load as load_repro_config

    return _extract_sql(
        router.complete(
            prompt,
            temperature=float(load_repro_config()["temperature"]["sql"]),
            max_tokens=800,
        )
    )


def _is_sqlite_source(source: DataSource, db_conf: dict | None = None) -> bool:
    conf = db_conf if db_conf is not None else source_to_db_config(source)
    if conf.get("_sqlite_path"):
        return True
    engine = str(getattr(source, "db_engine", "") or conf.get("dialect") or "").lower()
    return "sqlite" in engine


def _effective_row_limit(source: DataSource, db_conf: dict, options: dict) -> int:
    """Never inject a default LIMIT 100. Sqlite/benchmark runs stay uncapped."""
    if _is_sqlite_source(source, db_conf):
        return 0
    override = options.get("row_limit")
    if override is not None:
        try:
            return max(0, int(override))
        except (TypeError, ValueError):
            return 0
    try:
        return max(0, int(getattr(settings, "SQL_ROW_LIMIT", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _known_table_count(source: DataSource) -> int | None:
    catalog = source.column_catalog if isinstance(source.column_catalog, dict) else {}
    if catalog:
        return len(catalog)
    meta = source.meta or {}
    tables = meta.get("tables") or meta.get("created_tables") or []
    if isinstance(tables, list) and tables:
        return len(tables)
    return None


def _progress_phrase(text: str) -> str:
    """A short phrase safe to show. SQL and over-long text stay hidden."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned or _SQL_STATEMENT.search(cleaned):
        return ""
    if len(cleaned) > _PHRASE_MAX:
        cleaned = cleaned[: _PHRASE_MAX - 1].rstrip() + "…"
    return cleaned


def _announce_understanding(progress, question: str, plan) -> None:
    """Tell the chat what the question was taken to mean, and what to do next."""
    understood = (question or "").strip()
    steps: list[str] = []
    if plan is not None and not getattr(plan, "used_heuristic", True):
        resolved = str(getattr(plan, "resolved_question", "") or "").strip()
        if resolved:
            understood = resolved
        seen: set[str] = set()
        for step in getattr(plan, "plan_steps", None) or []:
            phrase = _progress_phrase(step)
            key = phrase.casefold()
            if not phrase or key in seen:
                continue
            seen.add(key)
            steps.append(phrase)
            if len(steps) == 4:
                break
    phrase = _progress_phrase(understood)
    if phrase:
        notify(progress, "question", "completed", detail=phrase)
    for step in steps:
        if step.casefold() == (phrase or "").casefold():
            continue
        notify(progress, "intent", "completed", detail=step)


def run_query(
    *,
    source: DataSource,
    question: str,
    history: list | None = None,
    options: dict | None = None,
    router: RouterClient | None = None,
    progress=None,
) -> dict[str, Any]:
    from data_service.clients.router import use_chat_model

    with use_chat_model((options or {}).get("model")):
        return _run_query(
            source=source,
            question=question,
            history=history,
            options=options,
            router=router,
            progress=progress,
        )


def _run_query(
    *,
    source: DataSource,
    question: str,
    history: list | None = None,
    options: dict | None = None,
    router: RouterClient | None = None,
    progress=None,
) -> dict[str, Any]:
    options = options or {}
    include_chart = bool(options.get("include_chart", True))
    skip_planner = bool(options.get("skip_planner", False))
    router = router or RouterClient()

    empty: dict[str, Any] = {
        "answer": "",
        "queries": [],
        "data": {"columns": [], "rows": []},
        "chart_spec": None,
        "chart": None,
        "metadata": {
            "data_source_id": source.id,
            "engine": source.db_engine or "mysql",
            "llm_path": "router:8101",
        },
    }

    q = (question or "").strip()
    if not q:
        empty["answer"] = "No question was received."
        empty["metadata"]["progress_reported"] = progress is not None
        return empty

    dialect = normalize_dialect(source.db_engine or "mysql")
    db_conf = source_to_db_config(source)
    sqlite_mode = _is_sqlite_source(source, db_conf)
    row_limit = _effective_row_limit(source, db_conf, options)
    title = safe_label(getattr(source, "title", ""))
    if title:
        notify(progress, "data_source", "completed", detail=title)
    view = ColumnView.from_catalog(source.column_catalog)
    full_schema = _schema_context(source, view)
    schema = slice_schema(source, q) or full_schema
    table_count = _known_table_count(source)
    if schema and schema != "(no schema description)":
        notify(progress, "data_schema", "running")
        notify(progress, "data_schema", "completed", count=table_count)
    hist = _history_lines(history)

    plan = None
    if not skip_planner:
        plan = plan_question(q, schema=schema, history=hist, router=router)
        empty["metadata"]["planner"] = {
            "resolved_question": plan.resolved_question,
            "used_heuristic": plan.used_heuristic,
            "corrections": plan.corrections,
            "plan_steps": [
                phrase
                for phrase in (_progress_phrase(step) for step in plan.plan_steps)
                if phrase
            ][:4],
        }

    _announce_understanding(progress, q, plan)

    rank = question_rank_direction(q, plan)
    ranking_hint = ""
    if wants_ranking(q, plan) and rank:
        ranking_hint = (
            f"Ranking intent detected: ORDER BY should be {rank.upper()}. "
            "Do not default to DESC unless the user asked for top/highest.\n"
        )

    few = _fewshot_match(source, q)
    sql_raw = few.ideal_sql if few else None
    llm_used = False
    retrieved_n = 0
    draft_sql = None
    draft_band = None
    draft_trusted = None
    fewshot_ids: list[int] = []
    plan_text = _plan_text(plan)

    notify(progress, "sql_generate", "running")
    if not sql_raw:
        llm_used = True
        examples = _list_active_fewshots(source)
        try:
            if not examples:
                sql_raw = _complete_sql(
                    router,
                    question=q,
                    schema=schema,
                    history=hist,
                    dialect=dialect,
                    ranking_hint=ranking_hint,
                    plan_text=plan_text,
                )
            else:
                try:
                    draft_sql = _complete_sql(
                        router,
                        question=q,
                        schema=schema,
                        history=hist,
                        dialect=dialect,
                        ranking_hint=ranking_hint,
                        plan_text=plan_text,
                    )
                except DownstreamError:
                    draft_sql = None
                draft_skel = sql_skeleton(draft_sql or "")
                if draft_skel is not None:
                    draft_band = draft_skel.band
                    draft_trusted = draft_is_trusted(q, draft_skel)
                elif draft_sql:
                    draft_trusted = False
                picked = _select_fewshots(
                    source,
                    q,
                    examples=examples,
                    draft_sql=draft_sql,
                    exclude=few,
                )
                if picked:
                    fewshot_ids = [
                        int(ex.id)
                        for ex in picked
                        if getattr(ex, "id", None) is not None
                    ]
                    extra_tables = _fewshot_tables_from(picked)
                    if extra_tables:
                        schema = (
                            slice_schema(source, q, extra_tables=extra_tables)
                            or full_schema
                        )
                    fewshot_block, retrieved_n = _format_fewshot_block(picked)
                    sql_raw = _complete_sql(
                        router,
                        question=q,
                        schema=schema,
                        history=hist,
                        dialect=dialect,
                        ranking_hint=ranking_hint,
                        plan_text=plan_text,
                        fewshot_block=fewshot_block,
                    )
                elif draft_sql:
                    sql_raw = draft_sql
                else:
                    sql_raw = _complete_sql(
                        router,
                        question=q,
                        schema=schema,
                        history=hist,
                        dialect=dialect,
                        ranking_hint=ranking_hint,
                        plan_text=plan_text,
                    )
        except DownstreamError as exc:
            notify(progress, "sql_generate", "failed")
            empty["answer"] = f"Router connection error: {exc.message}"
            empty["metadata"]["error"] = exc.message
            empty["metadata"]["progress_reported"] = progress is not None
            return empty

    notify(progress, "sql_generate", "completed")
    notify(progress, "sql_validate", "running")

    guarded, err = guard_sql_query(
        sql_raw,
        dialect=dialect,
        limit=row_limit,
        allowed_tables=_allowed_tables(source),
        sqlite_mode=sqlite_mode,
        question=q,
    )
    queries: list[dict[str, Any]] = []
    if err:
        queries.append(
            {"sql": guarded or sql_raw, "status": "blocked", "error": err}
        )
        notify(progress, "sql_validate", "failed")
        empty["answer"] = f"Query blocked: {err}"
        empty["queries"] = queries
        empty["metadata"]["progress_reported"] = progress is not None
        return empty

    notify(progress, "sql_validate", "completed")

    upload_schema = (
        source.db_schema if source.type == DataSourceType.UPLOAD else None
    )
    exec_timeout = settings.SQL_TIMEOUT_SECONDS
    sql_retry_attempted = False
    sql_retry_fixed = False

    connected = {"ok": False}

    def _connected() -> None:
        connected["ok"] = True
        notify(progress, "sql_execute", "running")

    def _run_sql(sql_text: str) -> tuple[list, list, str]:
        connected["ok"] = False
        notify(progress, "data_connect", "running")
        return execute_readonly_sql(
            db_conf,
            sql_text,
            schema=upload_schema,
            timeout=exec_timeout,
            on_connected=_connected,
        )

    def _record_exec_failure(sql_text: str, exc: Exception, *, attempt: int) -> None:
        logger.exception("SQL execution failed")
        if connected["ok"]:
            notify(progress, "sql_execute", "failed")
        else:
            notify(progress, "data_connect", "failed")
        queries.append(
            {
                "sql": sql_text,
                "status": "error",
                "error": str(exc),
                "attempt": attempt,
            }
        )

    try:
        columns, rows, preview = _run_sql(guarded)
        queries.append(
            {
                "sql": guarded,
                "status": "executed",
                "result_preview": preview[:4000],
                "attempt": 1,
            }
        )
    except Exception as exc:  # noqa: BLE001
        reached_engine = bool(connected["ok"])
        _record_exec_failure(guarded, exc, attempt=1)
        first_error = str(exc)
        # Only repair when MySQL actually ran the statement (status "error"),
        # not when the session never connected.
        if not reached_engine:
            empty["answer"] = f"Query execution error: {exc}"
            empty["queries"] = queries
            empty["metadata"]["error"] = first_error
            empty["metadata"]["sql_retry_attempted"] = False
            empty["metadata"]["sql_retry_fixed"] = False
            empty["metadata"]["progress_reported"] = progress is not None
            return empty

        sql_retry_attempted = True
        notify(progress, "sql_repair", "running")
        try:
            repaired_raw = _repair_sql(
                router,
                question=q,
                schema=schema,
                failed_sql=guarded,
                error_message=first_error,
                dialect=dialect,
            )
            llm_used = True
        except DownstreamError as repair_exc:
            notify(progress, "sql_repair", "failed")
            empty["answer"] = f"Query execution error: {first_error}"
            empty["queries"] = queries
            empty["metadata"]["error"] = first_error
            empty["metadata"]["sql_retry_attempted"] = True
            empty["metadata"]["sql_retry_fixed"] = False
            empty["metadata"]["sql_repair_error"] = repair_exc.message
            empty["metadata"]["progress_reported"] = progress is not None
            return empty

        repaired, repair_err = guard_sql_query(
            repaired_raw,
            dialect=dialect,
            limit=row_limit,
            allowed_tables=_allowed_tables(source),
            sqlite_mode=sqlite_mode,
            question=q,
        )
        if repair_err:
            notify(progress, "sql_repair", "failed")
            queries.append(
                {
                    "sql": repaired or repaired_raw,
                    "status": "blocked",
                    "error": repair_err,
                    "attempt": 2,
                }
            )
            empty["answer"] = f"Query execution error: {first_error}"
            empty["queries"] = queries
            empty["metadata"]["error"] = first_error
            empty["metadata"]["sql_retry_attempted"] = True
            empty["metadata"]["sql_retry_fixed"] = False
            empty["metadata"]["progress_reported"] = progress is not None
            return empty

        notify(progress, "sql_repair", "completed")
        try:
            columns, rows, preview = _run_sql(repaired)
            guarded = repaired
            sql_retry_fixed = True
            queries.append(
                {
                    "sql": guarded,
                    "status": "executed",
                    "result_preview": preview[:4000],
                    "attempt": 2,
                    "repaired": True,
                }
            )
        except Exception as retry_exc:  # noqa: BLE001
            _record_exec_failure(repaired, retry_exc, attempt=2)
            empty["answer"] = f"Query execution error: {retry_exc}"
            empty["queries"] = queries
            empty["metadata"]["error"] = str(retry_exc)
            empty["metadata"]["sql_retry_attempted"] = True
            empty["metadata"]["sql_retry_fixed"] = False
            empty["metadata"]["progress_reported"] = progress is not None
            return empty

    notify(progress, "sql_execute", "completed", count=len(rows))
    labels = view.labels_for(columns)
    column_units = view.units_aligned(columns, sql=guarded)
    units = view.units_for(columns, sql=guarded)
    if columns:
        notify(progress, "data_format", "completed", count=len(rows))

    answer = ""
    if few and few.ideal_answer and not llm_used:
        answer = few.ideal_answer
    else:
        notify(progress, "answer_generate", "running")
        try:
            answer = router.complete(
                _grounding_prompt(
                    question=q,
                    sql=guarded,
                    columns=labels,
                    rows=rows,
                    units=units,
                ),
                temperature=float(__import__("repro_config").load()["temperature"]["grounding"]),
                max_tokens=600,
            )
            llm_used = True
            notify(progress, "answer_generate", "completed")
        except DownstreamError as exc:
            notify(progress, "answer_generate", "failed")
            answer = f"Result: {len(rows)} rows. Columns: {', '.join(labels)}"
            empty["metadata"]["grounding_error"] = exc.message

    # Structured chart_spec ONLY — never PNG/base64 on the main query path.
    chart_spec = None
    if wants_chart(q, include_chart=include_chart):
        chart_spec = build_chart_spec_from_table(
            question=q,
            columns=labels,
            rows=rows,
            sql=guarded,
            units=column_units,
        )
        if chart_spec:
            notify(progress, "chart_prepare", "completed")

    return {
        "answer": answer,
        "queries": queries,
        "data": {"columns": labels, "rows": rows, "units": column_units},
        "chart_spec": chart_spec,
        "chart": chart_spec,
        "metadata": {
            "data_source_id": source.id,
            "engine": dialect,
            "llm_path": "router:8101",
            "fewshot": bool(few),
            "fewshot_retrieved": retrieved_n,
            "fewshot_ids": fewshot_ids,
            "fewshot_draft_band": draft_band,
            "fewshot_draft_trusted": draft_trusted,
            "rank_direction": rank,
            "planner": empty["metadata"].get("planner"),
            "sql_retry_attempted": sql_retry_attempted,
            "sql_retry_fixed": sql_retry_fixed,
            "progress_reported": progress is not None,
        },
    }


def verify_link_source(source: DataSource) -> None:
    db_conf = source_to_db_config(source)
    probe_connection(db_conf)
    try:
        desc, catalog, tables = introspect_schema(db_conf)
        meta = dict(source.meta or {})
        user_notes = str(meta.get("user_schema_description") or "").strip()
        if desc:
            source.schema_description = merge_schema_description(user_notes, desc)
        elif user_notes:
            source.schema_description = user_notes
        if catalog:
            source.column_catalog = ColumnView.merge(source.column_catalog, catalog)
        if tables:
            meta["tables"] = tables
        source.meta = meta
    except Exception as exc:  # noqa: BLE001
        logger.warning("schema introspect failed: %s", exc)
    source.last_status = ConnectionStatus.CONNECTED
    source.status_message = "connected"
    source.last_error = None
    source.save(
        update_fields=[
            "last_status",
            "status_message",
            "last_error",
            "schema_description",
            "column_catalog",
            "meta",
            "updated_at",
        ]
    )
