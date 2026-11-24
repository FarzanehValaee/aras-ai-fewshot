"""SQL safety helpers — ported from aras-ai-app Numbers agent."""

from __future__ import annotations

import json
import re

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DELETE|UPDATE|INSERT|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)
_FILE_EXPORT = re.compile(r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b", re.IGNORECASE)
_MULTI_STATEMENT = re.compile(r";\s*\S")
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_HAS_TOP = re.compile(r"\bSELECT\s+TOP\s+\d+", re.IGNORECASE)
_SELECT_PREFIX = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_COUNT_STAR = re.compile(r"^\s*SELECT\s+COUNT\s*\(", re.IGNORECASE)
_SELECT_BODY = re.compile(r"(?is)\b(?:WITH\b.+?\bSELECT\b|SELECT\b).+")
_ACTION_NOISE_SPLIT = re.compile(
    r"\b(?:Observation|Action\s*\d*\s*:|Action\s*\d*\s*Input\s*\d*\s*:|Final\s*Answer\s*:)\b",
    re.IGNORECASE,
)
_JSON_QUERY_RE = re.compile(
    r"""["']query["']\s*:\s*["']((?:\\.|[^"\\])*)["']""",
    re.IGNORECASE | re.DOTALL,
)


def _unwrap_json_query(text: str) -> str:
    s = str(text or "").strip()
    if not s.startswith("{"):
        return s
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            for key in ("query", "sql", "input"):
                if data.get(key):
                    return str(data[key]).strip()
    except json.JSONDecodeError:
        pass
    match = _JSON_QUERY_RE.search(s)
    if match:
        raw = match.group(1)
        try:
            return json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            return raw.replace('\\"', '"').replace("\\n", "\n").strip()
    return s


def _strip_sql_suffix_junk(sql: str) -> str:
    s = str(sql or "").strip()
    s = re.sub(r';+\s*["\'`].*$', "", s, flags=re.DOTALL).strip()
    # Only strip markdown fence closers (```). Never strip a single trailing `
    # — that often closes a MySQL identifier like `u5_1__dani1_sample`.
    s = re.sub(r"\s*```+\s*$", "", s).strip()
    s = s.rstrip(";").strip()
    s = re.sub(r'["\']\s*\}\s*$', "", s).strip()
    # Strip a wrapping quote around the whole statement only. Do not strip a
    # trailing quote that closes a string literal (e.g. WHERE x = 'France').
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"" and s.count(s[0]) == 2:
        s = s[1:-1].strip()
    # If an identifier backtick was left open, close it (LLM / prior sanitizers).
    if s.count("`") % 2 == 1:
        s = s + "`"
    return s


def _normalize_sql_text(query: str) -> str:
    cleaned = " ".join(str(query or "").split()).strip()
    cleaned = _unwrap_json_query(cleaned)
    cleaned = _ACTION_NOISE_SPLIT.split(cleaned, maxsplit=1)[0].strip()
    match = _SELECT_BODY.search(cleaned)
    if match:
        cleaned = match.group(0).strip()
    cleaned = _strip_sql_suffix_junk(cleaned)
    return cleaned


def sanitize_agent_sql_input(query: str) -> str:
    cleaned = str(query or "").strip()
    if cleaned.startswith("```sql"):
        cleaned = cleaned[5:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = " ".join(cleaned.split()).strip()
    return _normalize_sql_text(cleaned)


def escape_sql_like_fragment(value: str) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return text.replace("'", "''")



_CTE_DEF = re.compile(
    r"(?:WITH\s+(?:RECURSIVE\s+)?|,)\s*([`\"\[]?)([A-Za-z_][\w]*)\1\s+AS\s*\(",
    re.IGNORECASE,
)


def _extract_cte_names(sql: str) -> set[str]:
    """Names defined by WITH … AS (, including comma-separated CTEs."""
    return {match.group(2).lower() for match in _CTE_DEF.finditer(sql or "")}


def _extract_from_tables(sql: str) -> set[str]:
    """Best-effort table names from FROM/JOIN (unqualified + last part of schema.table)."""
    names: set[str] = set()
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+"
        r"([`\"\[]?)([A-Za-z_][\w]*)\1"
        r"(?:\s*\.\s*([`\"\[]?)([A-Za-z_][\w]*)\3)?",
        sql,
        flags=re.IGNORECASE,
    ):
        first = match.group(2).lower()
        second = (match.group(4) or "").lower()
        if second:
            # schema.table → compare on table name (allowlist stores bare table)
            names.add(second)
        else:
            names.add(first)
    return names - _extract_cte_names(sql)


def validate_sql(
    query: str,
    *,
    dialect: str,
    allowed_tables: set[str] | None = None,
) -> tuple[bool, str]:
    cleaned = _normalize_sql_text(query)
    if not cleaned:
        return False, "Query is empty."
    if _FORBIDDEN_KEYWORDS.search(cleaned):
        return False, "Only read-only SELECT queries are allowed."
    if _FILE_EXPORT.search(cleaned):
        return False, "File export statements are not allowed."
    if _MULTI_STATEMENT.search(cleaned):
        return False, "Multiple SQL statements are not allowed."
    if not _SELECT_PREFIX.match(cleaned) and not cleaned.upper().startswith("WITH "):
        return False, "Only SELECT (or WITH … SELECT) queries are allowed."
    if allowed_tables:
        allowed = {t.lower().split(".")[-1] for t in allowed_tables}
        referenced = _extract_from_tables(cleaned)
        # Ignore CTE names loosely; only enforce when we see physical refs
        bad = {t for t in referenced if t not in allowed}
        if bad:
            return False, f"Table(s) not allowed for this data source: {', '.join(sorted(bad))}"
    return True, ""


def enforce_row_limit(query: str, *, dialect: str, limit: int) -> str:
    cleaned = _normalize_sql_text(query)
    if not cleaned:
        return cleaned
    if _COUNT_STAR.match(cleaned):
        return cleaned
    if dialect == "sqlserver":
        if _HAS_TOP.search(cleaned):
            return cleaned
        return re.sub(r"(?i)^\s*SELECT\b", f"SELECT TOP {int(limit)} ", cleaned, count=1)
    if _HAS_LIMIT.search(cleaned):
        return cleaned
    return f"{cleaned} LIMIT {int(limit)}"


def guard_sql_query(
    query: str,
    *,
    dialect: str,
    limit: int,
    allowed_tables: set[str] | None = None,
) -> tuple[str, str | None]:
    normalized = sanitize_agent_sql_input(query)
    ok, err = validate_sql(
        normalized, dialect=dialect, allowed_tables=allowed_tables
    )
    if not ok:
        return normalized, err
    return enforce_row_limit(normalized, dialect=dialect, limit=limit), None
