"""DB connection helpers for linked and upload sources."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from django.conf import settings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from data_service.agent.config import build_db_config, build_db_uri
from data_service.models import DataSource, DataSourceType


def source_to_db_config(source: DataSource) -> dict:
    if source.type == DataSourceType.UPLOAD:
        # Uploaded tables live in the service's own MySQL.
        db = settings.DATABASES["default"]
        if db["ENGINE"].endswith("sqlite3"):
            # Tests: use in-memory sqlite via SQLAlchemy
            return {
                "dialect": "mysql",  # guard dialect; execution uses sqlite URI override
                "host": "localhost",
                "port": 0,
                "user": "sqlite",
                "password": "",
                "database": str(db["NAME"]),
                "_sqlite_path": str(db["NAME"]),
            }
        return build_db_config(
            {
                "engine": "mysql",
                "host": db.get("HOST") or "127.0.0.1",
                "port": int(db.get("PORT") or 3306),
                "user": db.get("USER") or "",
                "password": db.get("PASSWORD") or "",
                "database": db.get("NAME") or "",
            }
        )

    return build_db_config(
        {
            "engine": source.db_engine or "mysql",
            "host": source.host,
            "port": source.port,
            "user": source.username,
            "password": source.password,
            "database": source.db_name,
        }
    )


def _engine_for(db_conf: dict) -> Engine:
    if db_conf.get("_sqlite_path") is not None:
        path = db_conf["_sqlite_path"]
        uri = f"sqlite:///{path}"
        return create_engine(uri, pool_pre_ping=True)
    uri = build_db_uri(db_conf)
    return create_engine(uri, pool_pre_ping=True, pool_size=2, max_overflow=0)


def probe_connection(db_conf: dict) -> None:
    engine = _engine_for(db_conf)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


_MYSQL_FOREIGN_KEYS = (
    "SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
    "FROM information_schema.KEY_COLUMN_USAGE "
    "WHERE TABLE_SCHEMA = DATABASE() "
    "AND REFERENCED_TABLE_SCHEMA = DATABASE() "
    "AND REFERENCED_TABLE_NAME IS NOT NULL "
    "ORDER BY TABLE_NAME, ORDINAL_POSITION"
)

_POSTGRES_FOREIGN_KEYS = (
    "SELECT kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name "
    "FROM information_schema.table_constraints AS tc "
    "JOIN information_schema.key_column_usage AS kcu "
    "ON tc.constraint_name = kcu.constraint_name "
    "AND tc.table_schema = kcu.table_schema "
    "JOIN information_schema.constraint_column_usage AS ccu "
    "ON ccu.constraint_name = tc.constraint_name "
    "AND ccu.table_schema = tc.table_schema "
    "WHERE tc.constraint_type = 'FOREIGN KEY' "
    "AND tc.table_schema = 'public' "
    "ORDER BY kcu.table_name, kcu.ordinal_position"
)


def _attach_foreign_keys(tables_meta: list[dict], rows) -> None:
    by_table = {str(item.get("table")): item for item in tables_meta}
    for table, column, ref_table, ref_column in rows:
        column_name = str(column or "").strip()
        referenced_table = str(ref_table or "").strip()
        referenced_column = str(ref_column or "").strip()
        target = by_table.get(str(table))
        if not target or not column_name or not referenced_table or not referenced_column:
            continue
        entry = {
            "column": column_name,
            "references_table": referenced_table,
            "references_column": referenced_column,
        }
        keys = target.setdefault("foreign_keys", [])
        if entry not in keys:
            keys.append(entry)


def introspect_schema(db_conf: dict) -> tuple[str, dict, list[dict]]:
    """Best-effort schema description for linked DBs (MySQL/Postgres)."""
    dialect = db_conf.get("dialect") or "mysql"
    engine = _engine_for(db_conf)
    catalog: dict = {}
    tables_meta: list[dict] = []
    lines = ["Linked database tables:"]
    with engine.connect() as conn:
        if dialect == "mysql":
            rows = conn.execute(
                text(
                    "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                    "FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() "
                    "ORDER BY TABLE_NAME, ORDINAL_POSITION"
                )
            ).fetchall()
            fk_sql = _MYSQL_FOREIGN_KEYS
        elif dialect == "postgresql":
            rows = conn.execute(
                text(
                    "SELECT table_name, column_name, data_type "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' "
                    "ORDER BY table_name, ordinal_position"
                )
            ).fetchall()
            fk_sql = _POSTGRES_FOREIGN_KEYS
        else:
            return "", {}, []

        by_table: dict[str, list[str]] = {}
        for table, column, dtype in rows:
            t = str(table)
            by_table.setdefault(t, []).append(str(column))
            catalog.setdefault(t, []).append(
                {
                    "english_name": str(column),
                    "persian_name": str(column),
                    "unit": "",
                    "format": "",
                    "sql_type": str(dtype or ""),
                }
            )
        for t, cols in by_table.items():
            tables_meta.append({"table": t, "columns": cols})
            lines.append(f"- {t}: {', '.join(cols)}")
        try:
            fk_rows = conn.execute(text(fk_sql)).fetchall()
        except Exception:
            logger.warning("foreign key introspect failed", exc_info=True)
            fk_rows = []
        _attach_foreign_keys(tables_meta, fk_rows)
    return "\n".join(lines), catalog, tables_meta


def execute_readonly_sql(
    db_conf: dict,
    sql: str,
    *,
    schema: str | None = None,
    timeout: int = 30,
    on_connected=None,
) -> tuple[list[str], list[list[Any]], str]:
    engine = _engine_for(db_conf)
    with engine.connect() as conn:
        if on_connected is not None:
            try:
                on_connected()
            except Exception:
                logger.debug("sql progress callback failed", exc_info=True)
        if schema and not db_conf.get("_sqlite_path"):
            # Restrict session to upload schema when possible (MySQL).
            conn.execute(text(f"USE `{schema}`"))
        if timeout and not db_conf.get("_sqlite_path"):
            try:
                conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={int(timeout) * 1000}"))
            except Exception:
                pass
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [list(r) for r in result.fetchall()]
    preview = json.dumps(
        {"columns": columns, "rows": rows[:20]},
        ensure_ascii=False,
        default=str,
    )
    return columns, rows, preview
