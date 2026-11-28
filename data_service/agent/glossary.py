"""Column labels and units for any data source, read from its column catalog.

``english_name`` is the SQL identifier. ``persian_name`` is a legacy JSON key
for an optional display label (often equal to the English identifier).
``unit`` is the unit shown next to numbers. A column with no distinct label
and no unit is left unchanged, so the same code works for every database.
"""

from __future__ import annotations

import re
from typing import Any

_AGGREGATE = re.compile(
    r"^(?:sum|avg|count|min|max)\s*\(\s*(?:distinct\s+)?(.+?)\s*\)$",
    re.IGNORECASE,
)
_SELECT_BODY = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
_AS_ALIAS = re.compile(
    r"^(?P<expr>[\s\S]+?)\s+as\s+(?P<alias>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w]*)\s*$",
    re.IGNORECASE,
)
_BARE_ALIAS = re.compile(
    r"^(?P<expr>[\s\S]*\))\s+(?P<alias>`[^`]+`|\"[^\"]+\"|[A-Za-z_][\w]*)\s*$",
    re.IGNORECASE,
)
_FROM_TABLE = re.compile(
    r"\b(?:from|join)\s+(`[^`]+`|\"[^\"]+\"|[A-Za-z_][\w]*)",
    re.IGNORECASE,
)
_COUNT_STAR = re.compile(r"^count\s*\(\s*(?:\*|1)\s*\)$", re.IGNORECASE)


class ColumnView:
    def __init__(
        self,
        lines: list[str],
        by_name: dict[str, tuple[str, str]],
        units_by_table: dict[str, set[str]] | None = None,
    ):
        self._lines = lines
        self._by_name = by_name
        self._units_by_table = units_by_table or {}

    @classmethod
    def from_catalog(cls, catalog: Any) -> ColumnView:
        lines: list[str] = []
        grouped: dict[str, list[tuple[str, str]]] = {}
        units_by_table: dict[str, set[str]] = {}
        for table, column in _columns(catalog):
            sql = column["english_name"]
            label, unit = _label_and_unit(column)
            text = _glossary_line(table, sql, label, unit)
            if text:
                lines.append(text)
            grouped.setdefault(sql.casefold(), []).append((label, unit))
            if unit:
                units_by_table.setdefault(_table_key(table), set()).add(unit)

        by_name: dict[str, tuple[str, str]] = {}
        for key, pairs in grouped.items():
            labels = {label for label, _unit in pairs}
            units = {unit for _label, unit in pairs if unit}
            if len(labels) == 1 and len(units) <= 1:
                label = next(iter(labels))
                unit = next(iter(units)) if units else ""
                by_name[key] = (label, unit)
        return cls(lines, by_name, units_by_table)

    @property
    def prompt(self) -> str:
        if not self._lines:
            return ""
        return (
            "Column glossary. In SQL use only the english_name in backticks.\n"
            + "\n".join(self._lines)
        )

    def labels_for(self, columns: list[Any]) -> list[str]:
        return [self._lookup(column)[0] for column in columns]

    def units_aligned(self, columns: list[Any], sql: str = "") -> list[str]:
        """One unit string per result column. Empty when that column has none."""
        return [unit for _label, unit in self._resolved(columns, sql)]

    def units_for(self, columns: list[Any], sql: str = "") -> dict[str, str]:
        units: dict[str, str] = {}
        blocked: set[str] = set()
        for label, unit in self._resolved(columns, sql):
            if not unit or label in blocked:
                continue
            previous = units.get(label)
            if previous is None:
                units[label] = unit
            elif previous != unit:
                units.pop(label, None)
                blocked.add(label)
        return units

    def _resolved(self, columns: list[Any], sql: str) -> list[tuple[str, str]]:
        alias_map = _select_alias_map(sql)
        count_unit = self._count_star_unit(sql)
        resolved: list[tuple[str, str]] = []
        for column in columns:
            key = _column_key(column)
            expression = alias_map.get(key, str(column)) if key else str(column)
            label, unit = self._lookup(expression)
            if not unit and _is_count_star(expression):
                unit = count_unit
            resolved.append((label, unit))
        return resolved

    def _count_star_unit(self, sql: str) -> str:
        """Unit for COUNT(*) when every saved unit on the queried tables is the same."""
        found: set[str] = set()
        for table in _from_tables(sql):
            found.update(self._units_by_table.get(_table_key(table), ()))
        if len(found) == 1:
            return next(iter(found))
        return ""

    def _lookup(self, column: Any) -> tuple[str, str]:
        raw = str(column)
        key = _column_key(raw)
        if key and key in self._by_name:
            return self._by_name[key]
        embedded = self._embedded(raw)
        if embedded is not None:
            return embedded
        return (raw, "")

    def _embedded(self, raw: str) -> tuple[str, str] | None:
        """A result expression such as SUM(Sales) inherits that column's unit."""
        text = str(raw or "").casefold()
        hits: list[tuple[str, str]] = []
        for name, pair in self._by_name.items():
            if name and re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", text):
                hits.append(pair)
        if len(hits) == 1:
            return hits[0]
        units = {unit for _label, unit in hits if unit}
        if hits and len(units) == 1:
            return (hits[0][0], next(iter(units)))
        return None

    @staticmethod
    def merge(old: Any, new: Any) -> dict:
        """Keep saved display labels and units when introspection refreshes columns."""
        saved: dict[tuple[str, str], dict[str, str]] = {}
        for table, column in _columns(old):
            key = (_table_key(table), column["english_name"].casefold())
            current = saved.setdefault(key, {"persian_name": "", "unit": ""})
            if column["persian_name"]:
                current["persian_name"] = column["persian_name"]
            if column["unit"]:
                current["unit"] = column["unit"]

        if not isinstance(new, dict):
            return _copy_catalog(old)

        merged: dict[str, list[dict]] = {}
        for table, columns in new.items():
            if not isinstance(columns, list):
                continue
            copied: list[dict] = []
            for column in columns:
                entry = _normalize_entry(column)
                if entry is None:
                    continue
                kept = saved.get((_table_key(table), entry["english_name"].casefold()))
                if kept:
                    if kept["persian_name"]:
                        entry["persian_name"] = kept["persian_name"]
                    if kept["unit"]:
                        entry["unit"] = kept["unit"]
                copied.append(entry)
            merged[str(table)] = copied
        return merged

    @staticmethod
    def apply_labels(catalog: Any, edits: Any) -> dict:
        """Write display label (``persian_name`` key) and ``unit``. SQL names stay."""
        if not isinstance(edits, dict):
            raise ValueError("column_catalog must be an object")
        updated = _copy_catalog(catalog)
        index: dict[tuple[str, str], list[dict]] = {}
        for table, columns in updated.items():
            for column in columns:
                key = (_table_key(table), column["english_name"].casefold())
                index.setdefault(key, []).append(column)

        for table, columns in edits.items():
            if not isinstance(columns, list):
                continue
            for column in columns:
                if not isinstance(column, dict):
                    continue
                sql = str(column.get("english_name") or "").strip()
                if not sql:
                    continue
                targets = index.get((_table_key(table), sql.casefold()))
                if not targets:
                    continue
                for target in targets:
                    if "persian_name" in column:
                        target["persian_name"] = str(column.get("persian_name") or "").strip()
                    if "unit" in column:
                        target["unit"] = str(column.get("unit") or "").strip()
        return updated


def _columns(catalog: Any):
    if not isinstance(catalog, dict):
        return
    for table, columns in catalog.items():
        if not isinstance(columns, list):
            continue
        for column in columns:
            entry = _normalize_entry(column)
            if entry is not None:
                yield str(table), entry


def _copy_catalog(catalog: Any) -> dict[str, list[dict]]:
    copied: dict[str, list[dict]] = {}
    for table, column in _columns(catalog):
        copied.setdefault(table, []).append(dict(column))
    return copied


def _normalize_entry(column: Any) -> dict | None:
    if isinstance(column, str):
        sql = column.strip()
        if not sql:
            return None
        return {"english_name": sql, "persian_name": "", "unit": ""}
    if not isinstance(column, dict):
        return None
    sql = str(column.get("english_name") or column.get("name") or "").strip()
    if not sql:
        return None
    entry = dict(column)
    entry["english_name"] = sql
    entry["persian_name"] = str(column.get("persian_name") or "").strip()
    entry["unit"] = str(column.get("unit") or "").strip()
    return entry


def _label_and_unit(column: dict) -> tuple[str, str]:
    sql = column["english_name"]
    display = column["persian_name"]
    label = display if display and display != sql else sql
    return label, column["unit"]


def _glossary_line(table: str, sql: str, label: str, unit: str) -> str:
    if label == sql and not unit:
        return ""
    short = str(table).split(".")[-1].strip() or str(table)
    name = f"`{short}`.`{sql}`"
    if label != sql and unit:
        return f"{name} = {label}, unit: {unit}"
    if label != sql:
        return f"{name} = {label}"
    return f"{name} | unit: {unit}"


def _table_key(table: Any) -> str:
    return _strip_ident(str(table or "").split(".")[-1]).casefold()


def _strip_ident(name: str) -> str:
    return str(name or "").strip().strip("`").strip('"').strip("[]")


def _column_key(name: str) -> str:
    text = _strip_ident(name)
    match = _AGGREGATE.match(text)
    if match:
        text = _strip_ident(match.group(1))
        if text in {"*", "1"}:
            return ""
    if "." in text:
        text = _strip_ident(text.split(".")[-1])
    if text in {"*", "1"}:
        return ""
    return text.casefold()


def _is_count_star(expression: str) -> bool:
    return bool(_COUNT_STAR.match(str(expression or "").strip()))


def _select_alias_map(sql: str) -> dict[str, str]:
    """Map a SELECT output name to the expression that produced it."""
    match = _SELECT_BODY.search(str(sql or ""))
    if not match:
        return {}
    mapping: dict[str, str] = {}
    for item in _split_select_items(match.group(1)):
        text = item.strip()
        if not text:
            continue
        found = _AS_ALIAS.match(text) or _BARE_ALIAS.match(text)
        if not found:
            continue
        alias = _column_key(found.group("alias"))
        if alias:
            mapping[alias] = found.group("expr").strip()
    return mapping


def _split_select_items(body: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(buf))
            buf = []
            continue
        buf.append(char)
    if buf:
        items.append("".join(buf))
    return items


def _from_tables(sql: str) -> list[str]:
    return [match.group(1) for match in _FROM_TABLE.finditer(str(sql or ""))]
