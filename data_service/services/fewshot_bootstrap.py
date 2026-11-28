"""Queue ICRL few-shot bootstrap after a source is ready.

Stage 1: enqueue without failing upload.
Stage 2: schema-graph traversals sized by join complexity.
Stage 3–4: synthetic question/SQL + in-context RL refine (skipped for easy).
Stage 5: same SQL guard + execute path as user queries.
Stage 6: persist validated pairs as FewShotExample (auto_icrl only).

Generation fills reserved easy / medium / hard quotas. Easy comes only from
single-table paths; medium from single-table groups or one-join pairs; hard
from multi-join / CTE paths. Leftover slots prefer tables not covered yet.
"""

from __future__ import annotations

import logging
from typing import Any

from django.conf import settings
from django.db import connection

from data_service.agent.config import normalize_dialect
from data_service.agent.fewshot_skeleton import (
    EASY,
    HARD,
    MEDIUM,
    fewshot_quota,
    skeleton_shape_key,
    sql_skeleton,
)
from data_service.agent.icrl import (
    AUTO_INTENT,
    build_traversals,
    complexity_reward,
    generate_pair,
    quote_mysql_schema_idents,
    schema_join_stats,
    tables_from_meta,
)
from data_service.agent.pipeline import _allowed_tables
from data_service.agent.sql_guard import SQL_GENERATION_RULES
from data_service.clients.router import RouterClient
from data_service.models import DataSource, FewShotExample
from data_service.services.connections import execute_readonly_sql, source_to_db_config

logger = logging.getLogger(__name__)

_BAND_STYLES = {
    EASY: (
        "Style band: easy. One COUNT(*) from a single table.",
        "Style band: easy. WHERE filter on a real profile value; no GROUP BY.",
        "Style band: easy. One SUM/AVG/MIN/MAX on a single table; no JOIN.",
    ),
    MEDIUM: (
        "Style band: medium. GROUP BY on one table, optionally HAVING.",
        "Style band: medium. Exactly one JOIN plus an aggregate; no WITH.",
        "Style band: medium. GROUP BY plus ORDER BY ranking on one measure.",
    ),
    HARD: (
        "Style band: hard. Two JOINs across the path plus an aggregate.",
        "Style band: hard. WITH/CTE wrapping a multi-table join.",
        "Style band: hard. Subquery or three-table join with a filter.",
    ),
}


def _ident(name: str) -> str:
    return quote_mysql_schema_idents(name, [name]) if name else name


def _diversity_extra(style: str, previous: list[str]) -> str:
    lines = [style, "Generate a NEW question, different from previous ones."]
    if previous:
        lines.append("Already generated questions (do not repeat):")
        lines.extend(f"- {q}" for q in previous[-8:])
    return "\n".join(lines)


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _column_kind(name: str, sql_type: str) -> str:
    blob = f"{sql_type} {name}".lower()
    if any(k in blob for k in ("date", "time")):
        return "date"
    if any(
        k in blob
        for k in (
            "int",
            "decimal",
            "float",
            "double",
            "numeric",
            "year",
            "real",
            "capacity",
            "age",
            "amount",
            "price",
            "qty",
            "highest",
            "lowest",
            "average",
        )
    ):
        return "number"
    if name.lower() == "id" or name.lower().endswith("_id"):
        return "id"
    return "category"


def _catalog_types(source: DataSource) -> dict[str, dict[str, str]]:
    catalog = source.column_catalog if isinstance(source.column_catalog, dict) else {}
    out: dict[str, dict[str, str]] = {}
    for raw_name, cols in catalog.items():
        table = str(raw_name).split(".")[-1].strip().lower()
        typed: dict[str, str] = {}
        if isinstance(cols, list):
            for entry in cols:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("english_name") or entry.get("name") or "").strip()
                if name:
                    typed[name] = str(entry.get("sql_type") or "")
        out[table] = typed
    return out


def collect_profiles(
    db_conf: dict[str, Any], source: DataSource, tables
) -> dict[str, dict[str, Any]]:
    """Row counts, types, min/max, and a few frequent values for the prompt."""
    types = _catalog_types(source)
    schema = source.db_schema if source.db_schema else None
    timeout = int(getattr(settings, "SQL_TIMEOUT_SECONDS", 30))
    profiles: dict[str, dict[str, Any]] = {}
    for table in tables:
        info: dict[str, Any] = {"row_count": None, "columns": {}}
        try:
            _cols, rows, _preview = execute_readonly_sql(
                db_conf,
                f"SELECT COUNT(*) AS n FROM {_ident(table.name)}",
                schema=schema,
                timeout=timeout,
            )
            if rows:
                info["row_count"] = rows[0][0]
        except Exception:
            logger.info("profile count failed for %s", table.name)
        for col in list(table.columns)[:8]:
            kind = _column_kind(col, (types.get(table.name) or {}).get(col, ""))
            if kind == "id":
                continue
            fact: dict[str, Any] = {
                "kind": kind,
                "sql_type": (types.get(table.name) or {}).get(col, ""),
            }
            csql, tsql = _ident(col), _ident(table.name)
            try:
                if kind in {"number", "date"}:
                    _cols, rows, _preview = execute_readonly_sql(
                        db_conf,
                        f"SELECT MIN({csql}) AS lo, MAX({csql}) AS hi FROM {tsql}",
                        schema=schema,
                        timeout=timeout,
                    )
                    if rows and len(rows[0]) >= 2:
                        fact["min"], fact["max"] = rows[0][0], rows[0][1]
                else:
                    _cols, rows, _preview = execute_readonly_sql(
                        db_conf,
                        f"SELECT {csql} AS v, COUNT(*) AS n FROM {tsql} "
                        f"GROUP BY {csql} ORDER BY n DESC LIMIT 3",
                        schema=schema,
                        timeout=timeout,
                    )
                    fact["top"] = [row[0] for row in rows if row and row[0] not in (None, "")]
            except Exception:
                logger.info("profile column failed for %s.%s", table.name, col)
            info["columns"][col] = fact
        profiles[table.name] = info
    return profiles


def profile_block(trav, profiles: dict[str, dict[str, Any]]) -> str:
    lines = ["Measured column facts (use these real values, do not invent them):"]
    for name in trav.tables:
        info = profiles.get(name) or {}
        if info.get("row_count") is not None:
            lines.append(f"- {name} row_count: {info['row_count']}")
        for col, fact in (info.get("columns") or {}).items():
            if fact.get("sql_type"):
                lines.append(f"- {name}.{col} type: {fact['sql_type']}")
            if "min" in fact:
                lines.append(f"- {name}.{col} min: {fact['min']} max: {fact['max']}")
            if fact.get("top"):
                shown = ", ".join(str(v) for v in fact["top"][:3])
                lines.append(f"- {name}.{col} frequent: {shown}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _template_pairs(
    trav, profiles: dict[str, dict[str, Any]] | None = None, *, band: str | None = None
) -> list[dict[str, Any]]:
    table = str(trav.tables[0]) if trav.tables else ""
    if not table:
        return []
    cols = list(trav.columns_by_table.get(table) or ())
    tsql = _ident(table)
    lower = [(c, c.lower()) for c in cols]
    num = next(
        (
            c
            for c, lc in lower
            if any(k in lc for k in ("qty", "amount", "price", "sum", "count", "sales", "total"))
        ),
        None,
    )
    cat = next(
        (
            c
            for c, lc in lower
            if any(k in lc for k in ("group", "name", "title", "city", "state", "province", "store", "product"))
        ),
        None,
    )
    datec = next(
        (c for c, lc in lower if "date" in lc or lc.endswith("_at")),
        None,
    )
    drafts: list[tuple[str, str]] = [
        (f"How many rows are recorded in table {table}?", f"SELECT COUNT(*) AS n FROM {tsql}"),
    ]
    if num:
        nsql = _ident(num)
        drafts.append(
            (
                f"What is the sum of \"{num}\" in table {table}?",
                f"SELECT SUM({nsql}) AS total FROM {tsql}",
            )
        )
    if cat:
        csql = _ident(cat)
        drafts.append(
            (
                f"How many rows are there grouped by \"{cat}\"?",
                f"SELECT {csql} AS grp, COUNT(*) AS n FROM {tsql} GROUP BY {csql}",
            )
        )
        top = (((profiles or {}).get(table) or {}).get("columns") or {}).get(cat, {}).get("top") or []
        if top:
            drafts.append(
                (
                    f"How many rows in table {table} have \"{cat}\" equal to {top[0]}?",
                    f"SELECT COUNT(*) AS n FROM {tsql} WHERE {csql} = {_sql_literal(top[0])}",
                )
            )
        drafts.append(
            (
                f"Which \"{cat}\" value in table {table} has more than one row?",
                f"SELECT {csql} AS grp, COUNT(*) AS n FROM {tsql} GROUP BY {csql} HAVING n > 1",
            )
        )
    if datec:
        dsql = _ident(datec)
        drafts.append(
            (
                f"What are the oldest and newest values of \"{datec}\"?",
                f"SELECT MIN({dsql}) AS first_d, MAX({dsql}) AS last_d FROM {tsql}",
            )
        )
    if cat and num:
        csql, nsql = _ident(cat), _ident(num)
        drafts.append(
            (
                f"What are the top five \"{cat}\" values by sum of \"{num}\"?",
                f"SELECT {csql} AS grp, SUM({nsql}) AS total FROM {tsql} "
                f"GROUP BY {csql} ORDER BY total DESC",
            )
        )
    facts = ((profiles or {}).get(table) or {}).get("columns") or {}
    for col, fact in facts.items():
        if fact.get("kind") == "number" and fact.get("min") not in (None, "") and fact.get("max") not in (None, ""):
            if str(fact["min"]) == str(fact["max"]):
                continue
            csql = _ident(col)
            drafts.append(
                (
                    f"What are the minimum and maximum of \"{col}\" in table {table}?",
                    f"SELECT MIN({csql}) AS lo, MAX({csql}) AS hi FROM {tsql}",
                )
            )
            break
    for src, src_col, dst, dst_col in trav.foreign_keys:
        if src not in trav.tables or dst not in trav.tables:
            continue
        drafts.append(
            (
                f"How many related rows come from joining \"{src}\" and \"{dst}\"?",
                f"SELECT COUNT(*) AS n FROM {_ident(src)} "
                f"JOIN {_ident(dst)} ON {_ident(src)}.{_ident(src_col)} = "
                f"{_ident(dst)}.{_ident(dst_col)}",
            )
        )
    if len(trav.tables) >= 3 and trav.foreign_keys:
        src, src_col, dst, dst_col = trav.foreign_keys[0]
        drafts.append(
            (
                f"With a CTE, how many rows join \"{src}\" and \"{dst}\"?",
                "WITH linked AS ("
                f"SELECT 1 AS x FROM {_ident(src)} JOIN {_ident(dst)} "
                f"ON {_ident(src)}.{_ident(src_col)} = {_ident(dst)}.{_ident(dst_col)}"
                ") SELECT COUNT(*) AS n FROM linked",
            )
        )
    out: list[dict[str, Any]] = []
    for question, sql in drafts:
        skeleton = sql_skeleton(sql)
        if band and (skeleton is None or skeleton.band != band):
            continue
        out.append(
            {
                "question": question,
                "sql": sql,
                "tables": list(trav.tables),
                "reward": complexity_reward(sql),
                "style": _style_of(sql),
                "band": skeleton.band if skeleton else "",
            }
        )
    return out


def _style_of(sql: str) -> str:
    shape = sql_shape(sql, []).split(":", 1)[-1]
    if "WITH" in shape:
        return "with"
    if "JOIN" in shape:
        return "join"
    if "HAVING" in shape:
        return "having"
    if "WHERE" in shape:
        return "filter"
    if "MIN" in shape or "MAX" in shape:
        return "range"
    if "ORDER BY" in shape:
        return "rank"
    if "GROUP BY" in shape:
        return "group"
    return "count"


def sql_shape(sql: str, tables: list[str] | tuple[str, ...]) -> str:
    """Legacy marker string. Band dedupe uses ``skeleton_shape_key``."""
    blob = f" {(sql or '').upper()} "
    markers = (
        ("WITH", " WITH "),
        ("JOIN", " JOIN "),
        ("WHERE", " WHERE "),
        ("GROUP BY", " GROUP BY "),
        ("HAVING", " HAVING "),
        ("ORDER BY", " ORDER BY "),
        ("COUNT", "COUNT("),
        ("SUM", "SUM("),
        ("AVG", "AVG("),
        ("MIN", "MIN("),
        ("MAX", "MAX("),
    )
    found = [name for name, needle in markers if needle in blob]
    return ",".join(tables) + ":" + "+".join(found)


def _accept_pair(
    *,
    pair: dict[str, Any] | None,
    seen_questions: set[str],
    seen_sql: set[str],
    seen_shapes: set[str],
    db_conf: dict[str, Any],
    source: DataSource,
    band: str | None = None,
) -> dict[str, Any] | None:
    if not pair:
        return None
    qkey = pair["question"].strip().lower()
    skey = " ".join((pair.get("sql") or "").split()).lower()
    skeleton = sql_skeleton(pair.get("sql") or "")
    shape = skeleton_shape_key(pair.get("sql") or "", pair.get("tables") or [])
    if not qkey or qkey in seen_questions or not skey or skey in seen_sql:
        return None
    if skeleton is None or shape is None:
        logger.info("skip ICRL SQL with no skeleton")
        return None
    if band and skeleton.band != band:
        logger.info(
            "skip ICRL SQL outside band %s (got %s)", band, skeleton.band
        )
        return None
    if band == EASY and len(pair.get("tables") or []) != 1:
        logger.info("skip easy ICRL SQL on a multi-table path")
        return None
    if shape in seen_shapes:
        return None
    try:
        _columns, rows, _preview = execute_readonly_sql(
            db_conf,
            pair["sql"],
            schema=source.db_schema if source.db_schema else None,
            timeout=int(getattr(settings, "SQL_TIMEOUT_SECONDS", 30)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("skip ICRL SQL that did not execute: %s", exc)
        return None
    upper = f" {pair['sql'].upper()} "
    if not rows:
        logger.info("skip ICRL SQL with no rows")
        return None
    if (
        " WHERE " in upper
        and len(rows) == 1
        and len(rows[0]) == 1
        and rows[0][0] in (0, 0.0, None)
    ):
        logger.info("skip ICRL SQL whose filter matched nothing")
        return None
    if len(pair.get("tables") or []) > 1 and " JOIN " not in upper and " WITH " not in upper:
        logger.info("skip simple SQL on a multi-table path")
        return None
    pair["style"] = pair.get("style") or _style_of(pair["sql"])
    pair["band"] = skeleton.band
    seen_questions.add(qkey)
    seen_sql.add(skey)
    seen_shapes.add(shape)
    return pair


def _band_style(band: str, trav, index: int) -> str:
    styles = _BAND_STYLES.get(band) or _BAND_STYLES[EASY]
    base = styles[index % len(styles)]
    table_n = len(trav.tables)
    if band == MEDIUM and table_n >= 2:
        return (
            "Style band: medium. Exactly one JOIN on the given ON columns "
            "plus an aggregate. No WITH and no second JOIN."
        )
    if band == MEDIUM and table_n == 1:
        return (
            "Style band: medium. GROUP BY on this one table, optionally with "
            "HAVING or ORDER BY. No JOIN and no WITH."
        )
    if band == HARD and table_n >= 3:
        return (
            "Style band: hard. Join all tables in the path, or wrap that join "
            "in WITH."
        )
    if band == HARD:
        return (
            "Style band: hard. WITH/CTE around a join, or a subquery over the "
            "given tables."
        )
    return base


def _paths_for_band(band: str, traversals: list) -> list:
    if band == EASY:
        return [t for t in traversals if len(t.tables) == 1]
    if band == MEDIUM:
        singles = [t for t in traversals if len(t.tables) == 1]
        pairs = [t for t in traversals if len(t.tables) == 2]
        return singles + pairs
    return [t for t in traversals if len(t.tables) >= 2]


def _sort_paths(paths: list, covered: set[str]) -> list:
    def key(trav) -> tuple:
        names = {str(n).split(".")[-1].casefold() for n in trav.tables}
        uncovered = len(names - covered)
        return (-uncovered, len(names), ",".join(sorted(names)))

    return sorted(paths, key=key)


def _covered_tables(candidates: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for pair in candidates:
        for table in pair.get("tables") or []:
            bare = str(table).split(".")[-1].casefold()
            if bare:
                names.add(bare)
    return names


def _executor(db_conf: dict[str, Any], source: DataSource):
    timeout = int(getattr(settings, "SQL_TIMEOUT_SECONDS", 30))
    schema = source.db_schema if source.db_schema else None

    def _ran(sql: str) -> bool:
        try:
            execute_readonly_sql(db_conf, sql, schema=schema, timeout=timeout)
        except Exception:
            return False
        return True

    return _ran


def _fill_band(
    *,
    band: str,
    need: int,
    paths: list,
    candidates: list[dict[str, Any]],
    seen_questions: set[str],
    seen_sql: set[str],
    seen_shapes: set[str],
    db_conf: dict[str, Any],
    source: DataSource,
    profiles: dict[str, dict[str, Any]],
    client: RouterClient,
    dialect: str,
    row_limit: int,
    iterations: int,
    source_allowed: set[str] | None,
    source_id: int,
) -> int:
    """Append up to ``need`` accepted pairs for one difficulty band. Returns added count."""
    if need <= 0 or not paths:
        return 0
    added = 0
    max_attempts = max(
        need * 3,
        int(getattr(settings, "FEWSHOT_ICRL_MAX_ATTEMPTS", 20)),
    )
    attempts = 0
    consecutive_miss = 0
    run = _executor(db_conf, source)

    while added < need and attempts < max_attempts:
        covered = _covered_tables(candidates)
        ordered = _sort_paths(paths, covered)
        progressed = False
        for trav in ordered:
            if added >= need or attempts >= max_attempts:
                break
            trav_names = {n.lower() for n in trav.tables}
            if source_allowed is not None and not trav_names <= source_allowed:
                continue
            style = _band_style(band, trav, added + attempts)
            extra = "\n".join(
                part
                for part in (
                    "The question must be natural English. "
                    + SQL_GENERATION_RULES,
                    profile_block(trav, profiles),
                    _diversity_extra(
                        style,
                        [p["question"] for p in candidates if p.get("band") == band],
                    ),
                )
                if part
            )
            pair = None
            try:
                pair = generate_pair(
                    trav,
                    dialect=dialect,
                    router=client,
                    allowed_tables=trav_names,
                    row_limit=row_limit,
                    iterations=iterations,
                    extra=extra,
                    execute=run,
                    band=band,
                )
            except Exception:
                logger.exception(
                    "ICRL generate failed for source %s path %s band %s",
                    source_id,
                    trav.tables,
                    band,
                )
            attempts += 1
            accepted = _accept_pair(
                pair=pair,
                seen_questions=seen_questions,
                seen_sql=seen_sql,
                seen_shapes=seen_shapes,
                db_conf=db_conf,
                source=source,
                band=band,
            )
            if accepted:
                candidates.append(accepted)
                added += 1
                consecutive_miss = 0
                progressed = True
                break
            consecutive_miss += 1
            if consecutive_miss >= 3 and pair is None:
                break
        if not progressed:
            break

    if added < need:
        for trav in _sort_paths(paths, _covered_tables(candidates)):
            if added >= need:
                break
            trav_names = {n.lower() for n in trav.tables}
            if source_allowed is not None and not trav_names <= source_allowed:
                continue
            for tpl in _template_pairs(trav, profiles, band=band):
                if added >= need:
                    break
                sql = quote_mysql_schema_idents(
                    tpl["sql"],
                    list(trav.tables)
                    + [c for cols in (trav.columns_by_table or {}).values() for c in cols],
                )
                tpl = {**tpl, "sql": sql}
                accepted = _accept_pair(
                    pair=tpl,
                    seen_questions=seen_questions,
                    seen_sql=seen_sql,
                    seen_shapes=seen_shapes,
                    db_conf=db_conf,
                    source=source,
                    band=band,
                )
                if accepted:
                    candidates.append(accepted)
                    added += 1
    return added


def enqueue_fewshot_bootstrap(source_id: int) -> None:
    if not getattr(settings, "FEWSHOT_ICRL_ON_UPLOAD", True):
        return
    try:
        bootstrap_fewshots_for_source(source_id)
    except Exception:
        logger.exception("few-shot bootstrap failed for source %s", source_id)


def icrl_generation_params(*, iterations: int, quota: dict[str, Any]) -> dict[str, Any]:
    """Params written on every accepted row of one bootstrap run."""
    from repro_config import load as load_repro_config

    cfg = load_repro_config()
    configured_max = int(getattr(settings, "FEWSHOT_ICRL_MAX_ATTEMPTS", 20))
    return {
        "iterations": int(iterations),
        "temperature": float(cfg["temperature"]["icrl"]),
        "max_attempts": configured_max,
        "max_attempts_by_band": {
            band: max(int(quota.get(band) or 0) * 3, configured_max)
            for band in (EASY, MEDIUM, HARD)
        },
        "band_quotas": {
            band: int(quota.get(band) or 0) for band in (EASY, MEDIUM, HARD, "total")
        },
        "config_hash": str(cfg["config_hash"]),
    }


def bootstrap_fewshots_for_source(
    source_id: int,
    *,
    router: RouterClient | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Build the auto_icrl pool for one data source.

    Reward: ``complexity_reward`` on the candidate SQL only. The split is
    ``none`` (not Spider train, not Spider dev). See that function's docstring
    for the keyword sum. If ``icrl.reward_split`` is ``dev``, a loud warning
    is printed.

    Add: a candidate is kept only when the question and SQL are new, the
    skeleton band matches the slot, the shape key is new, the SQL executes on
    this source and returns rows, and a WHERE that yields a single 0/NULL
    cell is dropped. Easy slots also require exactly one table. Multi-table
    paths require JOIN or WITH. Shape, question, and SQL dedup sets are one
    set for this source and are shared across bands.

    Remove: immediately before insert, every existing FewShotExample on this
    source with intent ``auto_icrl`` is deleted. Other intents are left in
    place. The new rows are bulk-created with
    ``priority = max(1, 200 - int(reward * 10))`` and a ``generation_params``
    object (iterations, temperature, max attempts, band quotas, config hash).
    Those params are also logged once for the source.
    """
    from data_service.agent.icrl import warn_if_reward_on_dev

    warn_if_reward_on_dev()
    connection.ensure_connection()
    try:
        source = DataSource.objects.get(pk=source_id)
    except DataSource.DoesNotExist:
        return {"status": "skipped", "reason": "missing_source", "created": 0}

    tables = tables_from_meta(source.meta, source.column_catalog)
    if not tables:
        logger.info("few-shot bootstrap skipped for source %s: no tables", source_id)
        return {
            "status": "skipped",
            "reason": "no_tables",
            "data_source_id": source_id,
            "created": 0,
            "traversals": 0,
        }

    dialect = normalize_dialect(source.db_engine or "mysql")
    iterations = int(getattr(settings, "FEWSHOT_ICRL_ITERATIONS", 1))
    # 0 disables LIMIT injection so synthetic few-shots do not teach LIMIT 100.
    row_limit = 0
    table_count, pair_count = schema_join_stats(tables)
    quota = fewshot_quota(table_count=table_count, pair_count=pair_count)
    max_trav = max(
        int(getattr(settings, "FEWSHOT_ICRL_MAX_TRAVERSALS", 8)),
        table_count + pair_count + min(table_count, max(pair_count, 1)),
        quota["total"],
    )
    traversals = build_traversals(tables, max_traversals=max_trav)
    paths = [list(t.tables) for t in traversals]
    source_allowed = _allowed_tables(source)
    client = router or RouterClient()
    db_conf = source_to_db_config(source)
    profiles = collect_profiles(db_conf, source, tables)

    candidates: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_sql: set[str] = set()
    seen_shapes: set[str] = set()
    band_counts = {EASY: 0, MEDIUM: 0, HARD: 0}
    generation_params = icrl_generation_params(iterations=iterations, quota=quota)
    logger.info(
        "few-shot ICRL generation params source %s: %s",
        source_id,
        generation_params,
    )
    for band in (EASY, MEDIUM, HARD):
        need = int(quota.get(band) or 0)
        got = _fill_band(
            band=band,
            need=need,
            paths=_paths_for_band(band, traversals),
            candidates=candidates,
            seen_questions=seen_questions,
            seen_sql=seen_sql,
            seen_shapes=seen_shapes,
            db_conf=db_conf,
            source=source,
            profiles=profiles,
            client=client,
            dialect=dialect,
            row_limit=row_limit,
            iterations=iterations,
            source_allowed=source_allowed,
            source_id=source_id,
        )
        band_counts[band] = got

    created = 0
    if candidates:
        try:
            rows = [
                FewShotExample(
                    data_source=source,
                    title=(
                        "icrl:"
                        + str(c.get("style") or _style_of(c["sql"]))
                        + ":"
                        + ",".join(c["tables"])
                    )[:240],
                    user_question=c["question"].strip(),
                    ideal_sql=c["sql"],
                    ideal_answer="",
                    intent=AUTO_INTENT,
                    is_active=True,
                    priority=max(1, 200 - int(float(c["reward"]) * 10)),
                    generation_params=dict(generation_params),
                )
                for c in candidates
            ]
            FewShotExample.objects.filter(
                data_source=source, intent=AUTO_INTENT
            ).delete()
            FewShotExample.objects.bulk_create(rows)
            created = len(rows)
        except Exception:
            logger.exception(
                "few-shot persist failed for source %s", source_id
            )

    logger.info(
        "few-shot ICRL for source %s: %s created / %s validated / "
        "quota=%s bands=%s / %s traversals (intent=%s)",
        source_id,
        created,
        len(candidates),
        quota,
        band_counts,
        len(traversals),
        AUTO_INTENT,
    )
    return {
        "status": "ok",
        "reason": "persisted" if created else "validated",
        "data_source_id": source_id,
        "created": created,
        "traversals": len(traversals),
        "paths": paths,
        "quota": quota,
        "bands": band_counts,
        "validated": len(candidates),
        "candidates": [
            {
                "question": c["question"],
                "sql": c["sql"],
                "tables": c["tables"],
                "reward": c["reward"],
                "band": c.get("band") or "",
            }
            for c in candidates
        ],
    }
