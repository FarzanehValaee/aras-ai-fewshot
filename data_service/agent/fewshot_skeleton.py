"""Pure few-shot skeletons. No database and no model calls.

Retrieval and bootstrap both use these labels. SQL text is reduced to a clause
skeleton and a difficulty band. A question is reduced to analytical roles, with
catalog names removed so the same pattern on two tables still matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

_TRANS = str.maketrans(
    {
        "\u200c": "",
        "\u200d": "",
    }
)
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u0600-\u06FF]+")
_PLURAL_SUFFIXES = ("ies", "es", "s")
_KEYWORDS = frozenset(
    """
    where group order having limit offset join left right inner outer cross
    full on union select from as natural with recursive window qualify and or
    not null true false distinct all case when then else end asc desc by using
    lateral straight_join
    """.split()
)
# Same priority as the old question style hint: first match wins for style_hint.
_HARD_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("join", ("join",)),
    ("having", ("more than", "greater than")),
    ("range", ("minimum", "maximum")),
    ("filter", ("equal to", "equals")),
    ("group", ("grouped by", "group by")),
)
_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("count", ("count", "how many", "number of")),
    ("sum", ("sum", "total")),
    ("avg", ("average", "avg", "mean")),
    ("min", ("minimum", "lowest", "min")),
    ("max", ("maximum", "highest", "max")),
    ("group", ("for each", "grouped by", "group by", "per", "each")),
    ("join", ("join",)),
    ("filter", ("equal to", "equals", "greater than", "less than", "more than")),
)


def _normalize(text: str) -> str:
    return str(text or "").translate(_TRANS).casefold()


def _boundary(phrase: str) -> re.Pattern[str]:
    body = re.escape(_normalize(phrase))
    return re.compile(rf"(?<![\w\u0600-\u06FF]){body}(?![\w\u0600-\u06FF])", re.I)


_ROLE_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (role, tuple(_boundary(phrase) for phrase in phrases)) for role, phrases in _ROLES
)
_ROLE_ATOMS = frozenset(
    _normalize(phrase)
    for _role, phrases in _ROLES
    for phrase in phrases
    if " " not in phrase and "\u200c" not in phrase
)


@dataclass(frozen=True)
class SqlSkeleton:
    """Clause skeleton of one read-only SQL statement.

    ``band`` is easy for a single table with no join, grouping, having, CTE, or
    subquery, and at most one aggregate call. A WHERE and an ORDER BY still
    count as easy. Medium is grouping, having, exactly one join, two tables, or
    two aggregate calls. Hard is two joins, three tables, a CTE, or a subquery.
    """

    tables: frozenset[str] = frozenset()
    join_count: int = 0
    aggs: frozenset[str] = frozenset()
    agg_count: int = 0
    where: bool = False
    group_by: bool = False
    having: bool = False
    order_by: bool = False
    cte: bool = False
    subquery: bool = False

    @property
    def band(self) -> str:
        if self.cte or self.subquery or self.join_count >= 2 or len(self.tables) >= 3:
            return HARD
        if (
            self.join_count == 1
            or self.group_by
            or self.having
            or len(self.tables) == 2
            or self.agg_count >= 2
        ):
            return MEDIUM
        return EASY

    def covers(self, marker: str) -> bool:
        if marker == "join":
            return self.join_count >= 1
        if marker == "having":
            return self.having
        if marker == "range":
            return bool({"min", "max"} & self.aggs)
        if marker == "filter":
            return self.where
        if marker == "group":
            return self.group_by
        return False


def sql_skeleton(sql: str) -> SqlSkeleton | None:
    """Parse a SQL skeleton, or None when the text is not a SELECT or WITH."""
    if not str(sql or "").strip():
        return None
    prepared = _prepare_sql(sql)
    if not re.search(r"\b(?:SELECT|WITH)\b", prepared, re.I):
        return None
    tables: set[str] = set()
    _collect_tables(prepared, tables)
    cte_names, without_cte = _split_ctes(prepared)
    tables -= cte_names
    aggs = tuple(
        match.group(1).casefold()
        for match in re.finditer(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", prepared, re.I)
    )
    return SqlSkeleton(
        tables=frozenset(tables),
        join_count=len(re.findall(r"\bJOIN\b", prepared, re.I)),
        aggs=frozenset(aggs),
        agg_count=len(aggs),
        where=bool(re.search(r"\bWHERE\b", prepared, re.I)),
        group_by=bool(re.search(r"\bGROUP\s+BY\b", prepared, re.I)),
        having=bool(re.search(r"\bHAVING\b", prepared, re.I)),
        order_by=bool(re.search(r"\bORDER\s+BY\b", prepared, re.I)),
        cte=bool(cte_names) or bool(re.search(r"\bWITH\b", prepared, re.I)),
        subquery=bool(re.search(r"\(\s*SELECT\b", without_cte, re.I)),
    )


def question_skeleton(question: str, catalog: dict | None = None) -> str:
    """Analytical roles left after catalog names are removed.

    ``count of orders for each customer`` and ``count of invoices for each
    product`` both become ``count+group`` when those entity names are in the
    catalog.
    """
    text = _strip_schema(_normalize(question), _schema_tokens(catalog))
    found = [
        role
        for role, patterns in _ROLE_PATTERNS
        if any(pattern.search(text) for pattern in patterns)
    ]
    return "+".join(found)


def hard_markers(question: str) -> tuple[str, ...]:
    """Hard-word markers present in the question, in style-hint priority.

    Each needle is matched as a whole word on the casefolded question
    (``\\bjoin\\b``), so a column such as ``date_joined`` does not count as ``join``.
    """
    text = _normalize(question)
    found: list[str] = []
    for name, needles in _HARD_MARKERS:
        if any(
            re.search(rf"\b{re.escape(_normalize(needle))}\b", text)
            for needle in needles
        ):
            found.append(name)
    return tuple(found)


def style_hint(question: str) -> str:
    """First hard marker, matching the historical single style hint."""
    markers = hard_markers(question)
    return markers[0] if markers else ""


def draft_is_trusted(question: str, skeleton: SqlSkeleton | None) -> bool:
    """False when the draft SQL is missing a piece the question wording requires.

    A question with no hard marker trusts any real skeleton. ``join`` requires
    a join, ``more than`` requires HAVING, ``minimum``/``maximum`` require MIN
    or MAX, ``equal to`` requires WHERE, and ``grouped by`` requires GROUP BY.
    """
    if skeleton is None:
        return False
    return all(skeleton.covers(marker) for marker in hard_markers(question))


def fewshot_quota(*, table_count: int, pair_count: int) -> dict[str, int]:
    """How many easy/medium/hard examples a schema graph should produce.

    Sizes come from configs/repro.yaml ``pool``. No join edges keep hard at
    ``no_join_hard`` and the total at ``no_join_easy + no_join_medium``.
    With edges the total is
    ``clamp(base + per_extra_table*(tables-1) + per_join_pair*pairs, min_total, max_total)``.
    The first ``reserved_per_band`` of each open band are reserved; leftover
    slots go hard, then medium, then easy.
    """
    from repro_config import load as load_repro_config

    pool = load_repro_config()["pool"]
    tables = max(0, int(table_count))
    pairs = max(0, int(pair_count))
    if tables <= 0:
        return {EASY: 0, MEDIUM: 0, HARD: 0, "total": 0}
    if pairs <= 0:
        easy = int(pool["no_join_easy"])
        medium = int(pool["no_join_medium"])
        hard = int(pool["no_join_hard"])
        return {EASY: easy, MEDIUM: medium, HARD: hard, "total": easy + medium + hard}

    total = min(
        int(pool["max_total"]),
        max(
            int(pool["min_total"]),
            int(pool["base"])
            + int(pool["per_extra_table"]) * max(0, tables - 1)
            + int(pool["per_join_pair"]) * pairs,
        ),
    )
    reserved = int(pool["reserved_per_band"])
    easy, medium, hard = reserved, reserved, reserved
    remaining = total - (easy + medium + hard)
    while remaining > 0:
        hard += 1
        remaining -= 1
        if remaining <= 0:
            break
        medium += 1
        remaining -= 1
        if remaining <= 0:
            break
        easy += 1
        remaining -= 1
    return {EASY: easy, MEDIUM: medium, HARD: hard, "total": easy + medium + hard}


def skeleton_shape_key(
    sql: str, tables: list[str] | tuple[str, ...] | None = None
) -> str | None:
    """Dedupe key: band + tables + aggregate set. Ignores WHERE literals."""
    skeleton = sql_skeleton(sql)
    if skeleton is None:
        return None
    names = set(skeleton.tables)
    if not names and tables:
        names = {
            str(name).split(".")[-1].strip().casefold()
            for name in tables
            if str(name).strip()
        }
    aggs = ",".join(sorted(skeleton.aggs)) if skeleton.aggs else "none"
    return f"{skeleton.band}:{','.join(sorted(names))}:{aggs}"


def mentioned_tables(question: str, catalog: dict | None = None) -> frozenset[str]:
    """Bare catalog table names that appear in the question (plurals allowed)."""
    if not isinstance(catalog, dict):
        return frozenset()
    text = _normalize(question)
    found: set[str] = set()
    for raw_name in catalog:
        bare = str(raw_name).split(".")[-1].strip()
        if not bare:
            continue
        for token in {bare.casefold()} | _name_tokens(bare):
            if not token:
                continue
            if _boundary(token).search(text) or _has_plural(text, token):
                found.add(bare.casefold())
                break
    return frozenset(found)


def _has_plural(text: str, stem: str) -> bool:
    for suffix in _PLURAL_SUFFIXES:
        if _boundary(stem + suffix).search(text):
            return True
    return False


def _draft_feature_score(draft: SqlSkeleton, example: SqlSkeleton) -> int:
    from repro_config import load as load_repro_config

    weights = load_repro_config()["similarity"]["feature_weights"]
    score = 0
    if draft.join_count == example.join_count:
        score += int(weights["join_count"])
    if draft.group_by == example.group_by:
        score += int(weights["group_by"])
    if draft.having == example.having:
        score += int(weights["having"])
    if draft.where == example.where:
        score += int(weights["where"])
    if draft.cte == example.cte:
        score += int(weights["cte"])
    if draft.subquery == example.subquery:
        score += int(weights["subquery"])
    score += int(weights["agg_overlap"]) * len(draft.aggs & example.aggs)
    score += int(weights["table_overlap"]) * len(draft.tables & example.tables)
    return score


def retrieve_for_prompt(
    question: str,
    examples: list,
    *,
    draft_sql: str | None = None,
    catalog: dict | None = None,
    top_k: int | None = None,
    exclude_ids: set[int] | None = None,
) -> list:
    """Pick few-shots by SQL/question skeletons, not raw word Jaccard.

    Trusted draft → same difficulty band, then table overlap, then question
    skeleton. Untrusted draft (hard wording, weak SQL) → medium/hard only,
    ranked by question skeleton and tables. No draft → all bands with that
    same question/table ranking.

    Returns at most five examples, all one band. Same-pattern examples on
    other tables are used only when nothing overlaps the question tables,
    and then only one.
    """
    from data_service.agent.selection_strategy import select_fewshots

    return select_fewshots(
        question,
        examples,
        strategy="full",
        k=top_k,
        draft_sql=draft_sql,
        catalog=catalog,
        exclude_ids=exclude_ids,
    )


def _schema_tokens(catalog: dict | None) -> set[str]:
    if not isinstance(catalog, dict):
        return set()
    tokens: set[str] = set()
    for raw_name, columns in catalog.items():
        tokens |= _name_tokens(str(raw_name).split(".")[-1])
        if not isinstance(columns, list):
            continue
        for column in columns:
            if isinstance(column, str):
                tokens |= _name_tokens(column)
                continue
            if not isinstance(column, dict):
                continue
            for key in ("english_name", "name", "persian_name", "source_name"):
                tokens |= _name_tokens(str(column.get(key) or ""))
    return {token for token in tokens if token and token not in _ROLE_ATOMS}


def _name_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for match in _TOKEN.finditer(_normalize(text)):
        piece = match.group(0).casefold()
        if not piece:
            continue
        found.add(piece)
        if "_" in piece:
            found.update(part for part in piece.split("_") if part)
    return found


def _strip_schema(text: str, names: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        if _is_schema_token(match.group(0).casefold(), names):
            return " "
        return match.group(0)

    return _TOKEN.sub(replace, text)


def _is_schema_token(token: str, names: set[str]) -> bool:
    if token in names:
        return True
    for suffix in _PLURAL_SUFFIXES:
        if len(token) <= len(suffix) + 1 or not token.endswith(suffix):
            continue
        if token[: -len(suffix)] in names:
            return True
    return False


def _prepare_sql(sql: str) -> str:
    return _blank_function_from(_mask_literals(sql))


def _mask_literals(sql: str) -> str:
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            out.append(" ")
            i = n if end < 0 else end + 2
            continue
        if sql.startswith("--", i) or sql[i] == "#":
            end = sql.find("\n", i)
            out.append(" ")
            i = n if end < 0 else end
            continue
        if sql[i] == "'":
            i += 1
            while i < n:
                if sql.startswith("''", i):
                    i += 2
                    continue
                if sql[i] == "'":
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(sql[i])
        i += 1
    return "".join(out)


def _blank_function_from(sql: str) -> str:
    pattern = re.compile(
        r"\b(?:EXTRACT|SUBSTRING|SUBSTR|TRIM)\s*\([^()]*\bFROM\b",
        re.I,
    )

    def replace(match: re.Match[str]) -> str:
        return re.sub(r"\bFROM\b", "    ", match.group(0), count=1, flags=re.I)

    return pattern.sub(replace, sql)


def _collect_tables(sql: str, tables: set[str]) -> None:
    i = 0
    n = len(sql)
    while i < n:
        match = re.search(r"\b(?:FROM|JOIN)\b", sql[i:], re.I)
        if not match:
            return
        kind = match.group(0).upper()
        i += match.end()
        if kind == "JOIN":
            i = _read_join_target(sql, i, tables)
        else:
            i = _read_from_list(sql, i, tables)


def _read_from_list(sql: str, i: int, tables: set[str]) -> int:
    while True:
        i = _skip_ws(sql, i)
        if i >= len(sql):
            return i
        if sql[i] == "(":
            end = _skip_parens(sql, i)
            _collect_tables(sql[i + 1 : end - 1], tables)
            i = _skip_alias(sql, end)
        else:
            name, nxt = _read_table(sql, i)
            if not name:
                return i
            tables.add(name)
            i = _skip_alias(sql, nxt)
        nxt = _skip_ws(sql, i)
        if nxt < len(sql) and sql[nxt] == ",":
            i = nxt + 1
            continue
        return i


def _read_join_target(sql: str, i: int, tables: set[str]) -> int:
    i = _skip_ws(sql, i)
    if i < len(sql) and sql[i] == "(":
        end = _skip_parens(sql, i)
        _collect_tables(sql[i + 1 : end - 1], tables)
        return _skip_alias(sql, end)
    name, nxt = _read_table(sql, i)
    if not name:
        return i
    tables.add(name)
    return _skip_alias(sql, nxt)


def _read_table(sql: str, i: int) -> tuple[str | None, int]:
    name, nxt = _read_ident(sql, i)
    if not name:
        return None, i
    while True:
        dot = _skip_ws(sql, nxt)
        if dot >= len(sql) or sql[dot] != ".":
            break
        part, after = _read_ident(sql, dot + 1)
        if not part:
            break
        name, nxt = part, after
    bare = _bare(name)
    if not bare or bare in _KEYWORDS:
        return None, i
    return bare, nxt


def _read_ident(sql: str, i: int) -> tuple[str | None, int]:
    i = _skip_ws(sql, i)
    if i >= len(sql):
        return None, i
    if sql[i] in "`\"":
        quote = sql[i]
        end = sql.find(quote, i + 1)
        if end < 0:
            return None, len(sql)
        return sql[i + 1 : end], end + 1
    match = re.match(r"[A-Za-z_\u0600-\u06FF][\w\u0600-\u06FF]*", sql[i:])
    if not match:
        return None, i
    word = match.group(0)
    if word.casefold() in _KEYWORDS:
        return None, i
    return word, i + len(word)


def _bare(name: str) -> str:
    parts = [part.strip("`\" ").casefold() for part in name.split(".")]
    parts = [part for part in parts if part]
    return parts[-1] if parts else ""


def _skip_alias(sql: str, i: int) -> int:
    pos = _skip_ws(sql, i)
    marker = re.match(r"AS\b", sql[pos:], re.I)
    if marker:
        pos += marker.end()
        name, after = _read_ident(sql, pos)
        return after if name else pos
    name, after = _read_ident(sql, pos)
    if name is None:
        return i
    return after


def _skip_ws(sql: str, i: int) -> int:
    while i < len(sql) and sql[i].isspace():
        i += 1
    return i


def _skip_parens(sql: str, i: int) -> int:
    depth = 0
    while i < len(sql):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(sql)


def _split_ctes(sql: str) -> tuple[set[str], str]:
    """CTE names, and the statement with those CTE bodies removed."""
    match = re.search(r"\bWITH\b", sql, re.I)
    if not match:
        return set(), sql
    names: set[str] = set()
    i = match.end()
    recursive = re.match(r"\s*RECURSIVE\b", sql[i:], re.I)
    if recursive:
        i += recursive.end()
    kept = [sql[: match.start()]]
    while i < len(sql):
        name, after_name = _read_ident(sql, i)
        if not name:
            kept.append(sql[i:])
            break
        pos = _skip_ws(sql, after_name)
        if pos < len(sql) and sql[pos] == "(":
            pos = _skip_parens(sql, pos)
            pos = _skip_ws(sql, pos)
        marker = re.match(r"AS\b", sql[pos:], re.I)
        if not marker:
            kept.append(sql[i:])
            break
        names.add(_bare(name))
        pos = _skip_ws(sql, pos + marker.end())
        if pos >= len(sql) or sql[pos] != "(":
            kept.append(sql[pos:])
            break
        i = _skip_parens(sql, pos)
        kept.append(" ")
        i = _skip_ws(sql, i)
        if i < len(sql) and sql[i] == ",":
            i += 1
            continue
        kept.append(sql[i:])
        break
    return names, "".join(kept)
