"""Database connection config — ported from Numbers agent config."""

from __future__ import annotations

import os
from urllib.parse import quote_plus

SUPPORTED_ANALYTICS_DIALECTS = {"postgresql", "mysql", "sqlserver"}


class DatabaseConfigError(ValueError):
    """Raised when request database config is missing or invalid."""


def normalize_dialect(raw_dialect: str | None) -> str:
    if raw_dialect is None:
        return "mysql"
    dialect = str(raw_dialect).strip().lower()
    if dialect == "postgres":
        dialect = "postgresql"
    if dialect in {"mssql", "sql_server", "sql server"}:
        dialect = "sqlserver"
    if dialect not in SUPPORTED_ANALYTICS_DIALECTS:
        raise DatabaseConfigError(
            f"Unsupported database dialect: {dialect}. Supported: postgresql, mysql, sqlserver"
        )
    return dialect


def default_port_for_dialect(dialect: str) -> int:
    if dialect == "postgresql":
        return 5432
    if dialect == "mysql":
        return 3306
    return 1433


def build_db_config(config: dict | None) -> dict:
    if config is not None and not isinstance(config, dict):
        raise DatabaseConfigError("config must be a JSON object when provided.")

    merged: dict = {}
    if config:
        merged.update(config)

    if merged.get("dialect") is None and merged.get("engine"):
        merged["dialect"] = merged["engine"]

    merged["dialect"] = normalize_dialect(merged.get("dialect"))
    merged["port"] = int(merged.get("port") or default_port_for_dialect(merged["dialect"]))

    missing = [
        field
        for field in ("host", "user", "database")
        if not str(merged.get(field, "")).strip()
    ]
    if missing:
        raise DatabaseConfigError(
            "Missing database config field(s): " + ", ".join(missing)
        )

    return merged


def _mysql_socket_timeout_seconds() -> int:
    """Align pymysql read/write timeouts with SQL_TIMEOUT_SECONDS (not a fixed 30s)."""
    try:
        value = int(os.getenv("SQL_TIMEOUT_SECONDS", "300") or "300")
    except (TypeError, ValueError):
        value = 300
    return max(30, value)


def build_db_uri(db_conf: dict) -> str:
    dialect = db_conf["dialect"]
    user = quote_plus(str(db_conf["user"]))
    password = quote_plus(str(db_conf.get("password", "")))
    host = str(db_conf["host"]).strip()
    port = int(db_conf["port"])
    database = quote_plus(str(db_conf["database"]))

    if dialect == "postgresql":
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    if dialect == "mysql":
        sock_timeout = _mysql_socket_timeout_seconds()
        return (
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            f"?connect_timeout=10&read_timeout={sock_timeout}&write_timeout={sock_timeout}"
        )

    driver = (
        db_conf.get("driver")
        or os.getenv("SQLSERVER_ODBC_DRIVER")
        or "ODBC Driver 18 for SQL Server"
    ).strip()
    driver_q = quote_plus(driver)
    trust = str(db_conf.get("trust_server_certificate", "yes")).strip().lower()
    trust_val = "yes" if trust in {"1", "true", "yes", "y"} else "no"
    return (
        f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
        f"?driver={driver_q}&TrustServerCertificate={trust_val}"
    )
