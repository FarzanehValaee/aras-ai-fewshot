"""Schema text limited to tables and columns that share words with the question.

The caller keeps the user's sentence unchanged. This module only decides which
catalog rows sit beside that sentence. When nothing overlaps, it returns None
and the caller sends the full schema.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_TABLES = 5
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u0600-\u06FF]+")
_TRANS = str.maketrans(
    {
        "\u200c": "",
        "\u200d": "",
    }
)
# Question fillers. Domain words (amount, sales, month) stay.
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "to",
        "in",
        "for",
        "is",
        "on",
        "by",
        "with",
        "from",
        "what",
        "which",
        "how",
        "that",
        "this",
        "please",
        "show",
        "give",
        "tell",
        "are",
        "was",
        "were",
        "be",
        "or",
        "at",
        "as",
        "it",
        "we",
        "you",
        "me",
        "do",
        "does",
        "did",
        "can",
    }
)
_LINKED_MARKER = "Linked database tables:"


def slice_schema(
    source: Any,
    question: str,
    *,
    extra_tables: set[str] | None = None,
) -> str | None:
    """Return a short schema, or None when the question matches no column or table."""
    wanted = _tokens(question)
    if not wanted:
        return None

    catalog = getattr(source, "column_catalog", None)
    tables = _catalog_tables(catalog)
    if not tables:
        return None

    keys = _key_names(source)
    extra = {str(name).split(".")[-1].casefold() for name in (extra_tables or set()) if str(name).strip()}

    ranked: list[tuple[int, str, list[_Column]]] = []
    by_name: dict[str, tuple[int, str, list[_Column]]] = {}
    for bare, columns in tables:
        table_hit = len(wanted & _tokens(bare))
        scored: list[_Column] = []
        column_hit = 0
        for column in columns:
            score = len(wanted & _column_tokens(column))
            column.score = score
            column_hit += score
            scored.append(column)
        rank = table_hit + column_hit
        if rank <= 0 and bare.casefold() not in extra:
            continue
        row = (rank, bare, scored)
        ranked.append(row)
        by_name[bare.casefold()] = row

    matched = [row for row in ranked if row[0] > 0]
    matched.sort(key=lambda row: (-row[0], row[1].casefold()))
    chosen = matched[:_MAX_TABLES]
    chosen_names = {bare.casefold() for _, bare, _ in chosen}
    for name in sorted(extra):
        if name in chosen_names or name not in by_name:
            continue
        chosen.append(by_name[name])
        chosen_names.add(name)
    if not any(rank > 0 for rank, _, _ in chosen):
        return None

    selected_tables: list[tuple[str, list[_Column]]] = []
    selected_names: set[str] = set()
    for _rank, bare, columns in chosen:
        table_hit = len(wanted & _tokens(bare))
        own_hit = any(column.score > 0 for column in columns)
        include_all = bare.casefold() in extra or (table_hit > 0 and not own_hit)
        table_keys = keys.get(bare.casefold(), set())
        kept = [
            column
            for column in columns
            if include_all or column.score > 0 or _is_key(column.sql, table_keys)
        ]
        if not kept:
            continue
        selected_tables.append((bare, kept))
        selected_names.add(bare.casefold())
    if not selected_tables:
        return None

    lines: list[str] = []
    notes = _context_notes(source)
    if notes:
        lines.append(notes)
    lines.append("Relevant schema:")
    for bare, columns in selected_tables:
        lines.append(f"- {bare}: {', '.join(column.sql for column in columns)}")
    glossary = _glossary(selected_tables)
    if glossary:
        lines.append(glossary)
    joins = _join_lines(source, selected_names)
    if joins:
        lines.append("Joins:\n" + "\n".join(joins))
    return "\n".join(lines)


class _Column:
    def __init__(self, sql: str, persian: str, source: str, unit: str):
        self.sql = sql
        self.persian = persian
        self.source = source
        self.unit = unit
        self.score = 0


def _tokens(text: str) -> set[str]:
    raw = str(text or "").translate(_TRANS)
    found: set[str] = set()
    for match in _TOKEN.finditer(raw):
        piece = match.group(0)
        parts = [piece]
        if "_" in piece:
            parts.extend(part for part in piece.split("_") if part)
        for part in parts:
            token = part.casefold()
            if len(token) <= 1 or token in _STOP:
                continue
            found.add(token)
    return found


def _column_tokens(column: _Column) -> set[str]:
    return _tokens(" ".join((column.sql, column.persian, column.source, column.unit)))


def _catalog_tables(catalog: Any) -> list[tuple[str, list[_Column]]]:
    if not isinstance(catalog, dict):
        return []
    tables: list[tuple[str, list[_Column]]] = []
    for raw_name, raw_columns in catalog.items():
        bare = str(raw_name).split(".")[-1].strip()
        if not bare or not isinstance(raw_columns, list):
            continue
        columns: list[_Column] = []
        for raw in raw_columns:
            column = _column(raw)
            if column is not None:
                columns.append(column)
        if columns:
            tables.append((bare, columns))
    return tables


def _column(raw: Any) -> _Column | None:
    if isinstance(raw, str):
        sql = raw.strip()
        if not sql:
            return None
        return _Column(sql, "", "", "")
    if not isinstance(raw, dict):
        return None
    sql = str(raw.get("english_name") or raw.get("name") or "").strip()
    if not sql:
        return None
    return _Column(
        sql,
        str(raw.get("persian_name") or "").strip(),
        str(raw.get("source_name") or "").strip(),
        str(raw.get("unit") or "").strip(),
    )


def _is_key(sql: str, table_keys: set[str]) -> bool:
    name = sql.casefold()
    return name == "id" or name.endswith("_id") or name in table_keys


def _key_names(source: Any) -> dict[str, set[str]]:
    """FK columns and the columns they point at, keyed by bare table name."""
    meta = getattr(source, "meta", None) or {}
    names: dict[str, set[str]] = {}
    for item in meta.get("tables") or []:
        if not isinstance(item, dict):
            continue
        owner = str(item.get("table") or "").split(".")[-1].strip().casefold()
        if not owner:
            continue
        for fk in item.get("foreign_keys") or []:
            if not isinstance(fk, dict):
                continue
            column = str(fk.get("column") or "").strip().casefold()
            ref_table = str(fk.get("references_table") or "").split(".")[-1].strip().casefold()
            ref_column = str(fk.get("references_column") or "").strip().casefold()
            if column:
                names.setdefault(owner, set()).add(column)
            if ref_table and ref_column:
                names.setdefault(ref_table, set()).add(ref_column)
    return names


def _context_notes(source: Any) -> str:
    parts: list[str] = []
    description = str(getattr(source, "description", "") or "").strip()
    if description:
        parts.append(f"Business context:\n{description}")
    meta = getattr(source, "meta", None) or {}
    notes = str(meta.get("user_schema_description") or "").strip()
    if _LINKED_MARKER in notes:
        notes = notes.split(_LINKED_MARKER, 1)[0].strip()
    if notes:
        parts.append(notes)
    return "\n\n".join(parts)


def _glossary(tables: list[tuple[str, list[_Column]]]) -> str:
    lines: list[str] = []
    for bare, columns in tables:
        for column in columns:
            line = _glossary_line(bare, column)
            if line:
                lines.append(line)
    if not lines:
        return ""
    return (
        "Column glossary. In SQL use only the english_name in backticks.\n"
        + "\n".join(lines)
    )


def _glossary_line(table: str, column: _Column) -> str:
    sql = column.sql
    persian = column.persian if column.persian and column.persian != sql else ""
    source = column.source if column.source and column.source != sql else ""
    if persian and source and persian != source:
        label = f"{persian} ({source})"
    else:
        label = persian or source
    name = f"`{table}`.`{sql}`"
    if label and column.unit:
        return f"{name} = {label}, unit: {column.unit}"
    if label:
        return f"{name} = {label}"
    if column.unit:
        return f"{name} | unit: {column.unit}"
    return ""


def _join_lines(source: Any, selected: set[str]) -> list[str]:
    meta = getattr(source, "meta", None) or {}
    lines: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in meta.get("tables") or []:
        if not isinstance(item, dict):
            continue
        owner = str(item.get("table") or "").split(".")[-1].strip()
        if owner.casefold() not in selected:
            continue
        for fk in item.get("foreign_keys") or []:
            if not isinstance(fk, dict):
                continue
            column = str(fk.get("column") or "").strip()
            ref_table = str(fk.get("references_table") or "").split(".")[-1].strip()
            ref_column = str(fk.get("references_column") or "").strip()
            if not column or not ref_table or not ref_column:
                continue
            if ref_table.casefold() not in selected:
                continue
            key = (owner.casefold(), column.casefold(), ref_table.casefold(), ref_column.casefold())
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {owner}.{column} = {ref_table}.{ref_column}")
    return lines
