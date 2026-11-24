#!/usr/bin/env python3
"""Decide whether the few-shot pool is protocol-clean for Spider dev.

Read-only. Reads the pool, its English translation, Spider dev, and Spider
train. Writes results/pool_forensics.md and does not modify the pool.

A pool row's database is the Spider db_id whose table set equals the uploaded
data source's tables (from the source catalog, checked against tables named
in the row's SQL). Dev database ids come from dev.json. SQL overlap is an
exact match of case, whitespace, and alias-normalized SQL on that same db_id.
Question similarity is embedding cosine when sentence-transformers is
importable, otherwise content-word Jaccard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL = ROOT / "benchmark" / "fewshot_pool.json"
POOL_EN = ROOT / "benchmark" / "fewshot_pool_en.json"
REPORT = ROOT / "results" / "pool_forensics.md"

DEV_CANDIDATES = [
    Path(os.environ.get("SPIDER_DEV", "")),
    ROOT / "data" / "spider" / "spider_data" / "dev.json",
    ROOT / "benchmark" / "spider" / "dev.json",
    Path("/tmp/spider_dev.jsonl"),
    Path("/home/aras-lab/my-project/spider_dev_questions.json"),
]
TRAIN_CANDIDATES = [
    Path(os.environ.get("SPIDER_TRAIN", "")),
    ROOT / "data" / "spider" / "spider_data" / "train_spider.json",
    ROOT / "benchmark" / "spider" / "train_spider.json",
    Path("/tmp/train_spider.json"),
]
TABLES_CANDIDATES = [
    Path(os.environ.get("SPIDER_TABLES", "")),
    ROOT / "data" / "spider" / "spider_data" / "tables.json",
    ROOT / "benchmark" / "spider" / "tables.json",
    Path("/tmp/spider_tables.json"),
]

_SQL_KEYWORDS = frozenset(
    """
    select from where group by having order limit join inner left right full
    cross outer on and or not in is null as with union all distinct case when
    then else end asc desc between like exists intersect except using natural
    """.split()
)
_WS = re.compile(r"\s+")
_QUOTES = re.compile(r"[`\"\[\]]")
_AS_ALIAS = re.compile(r"\bas\s+[a-z_][a-z0-9_]*\b")
_QUALIFIER = re.compile(r"\b[a-z_][a-z0-9_]*\.")
_FROM_JOIN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", re.I)
_PUNCT = re.compile(r"[^a-z0-9]+")
_STOP = frozenset(
    """
    a an the of to and or in on for with by from is are was were be been being
    that which what who whom whose how when where why there this these those
    it its their do does did done not no nor than then so if but at into over
    under about across per each every all any some such me my we our you your
    please show list find give tell display return many much
    """.split()
)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path and str(path) not in {"", "."} and path.is_file():
            return path
    return None


def load_json_any(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl" or text[0] != "[":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    payload = json.loads(text)
    if isinstance(payload, dict):
        for key in ("data", "examples", "questions"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise SystemExit(f"{path} is a JSON object, not a list of examples")
    if not isinstance(payload, list):
        raise SystemExit(f"{path} is not a list")
    return payload


def spider_examples(path: Path) -> list[dict]:
    rows = []
    for index, raw in enumerate(load_json_any(path)):
        question = str(raw.get("question") or raw.get("q") or "").strip()
        sql = str(raw.get("query") or raw.get("SQL") or raw.get("sql") or "").strip()
        db_id = str(raw.get("db_id") or raw.get("db") or "").strip()
        if not question or not sql or not db_id:
            raise SystemExit(f"{path} row {index} is missing question, SQL, or db_id")
        rows.append({"db_id": db_id, "question": question, "query": sql})
    return rows


def normalize_sql(sql: str) -> str:
    """Case, whitespace, and alias normalization. Operators and star stay."""
    text = _QUOTES.sub("", sql.lower().replace(";", " "))
    text = _WS.sub(" ", text).strip()
    text = _AS_ALIAS.sub(" ", text)
    text = _QUALIFIER.sub("", text)
    tokens = text.split()
    out: list[str] = []
    expect_table = False
    saw_table = False
    for tok in tokens:
        if tok in {"from", "join"}:
            expect_table = True
            saw_table = False
            out.append(tok)
            continue
        if expect_table and not saw_table:
            out.append(tok)
            saw_table = True
            continue
        if expect_table and saw_table:
            if tok == ",":
                saw_table = False
                out.append(tok)
                continue
            if tok in _SQL_KEYWORDS or tok in {"inner", "outer", "left", "right", "cross", "join"}:
                expect_table = False
                out.append(tok)
                continue
            if re.fullmatch(r"t\d+|[a-z]", tok):
                continue
            expect_table = False
            out.append(tok)
            continue
        out.append(tok)
    text = " ".join(out)
    text = re.sub(r"\s*([(),])\s*", r"\1", text)
    return _WS.sub(" ", text).strip()


def sql_tables(sql: str) -> frozenset[str]:
    found = set()
    for match in _FROM_JOIN.finditer(sql):
        name = match.group(1).casefold()
        if name not in _SQL_KEYWORDS:
            found.add(name)
    return frozenset(found)


def content_tokens(text: str) -> frozenset[str]:
    words = _PUNCT.sub(" ", text.lower()).split()
    return frozenset(word for word in words if word not in _STOP and len(word) > 1)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_INTERNAL_TABLES = frozenset({"sqlite_sequence", "sqlite_stat1"})


def table_set(names) -> frozenset[str]:
    return frozenset(
        str(name).casefold()
        for name in names
        if str(name).casefold() not in _INTERNAL_TABLES
    )


def load_table_signatures(path: Path) -> dict[frozenset[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path} is not the official Spider tables list")
    signatures: dict[frozenset[str], str] = {}
    for item in payload:
        names = table_set(item.get("table_names_original") or [])
        db_id = str(item.get("db_id") or "")
        if not names or not db_id:
            continue
        previous = signatures.get(names)
        if previous and previous != db_id:
            raise SystemExit(f"table set {sorted(names)} is both {previous} and {db_id}")
        signatures[names] = db_id
    return signatures


def load_catalog_from_django(source_ids: list[int]) -> dict[str, dict]:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aras_ai_data.settings")
    import django

    django.setup()
    from data_service.models import DataSource

    catalog: dict[str, dict] = {}
    for source in DataSource.objects.filter(id__in=source_ids):
        raw = source.column_catalog or {}
        if isinstance(raw, dict):
            tables = [str(name) for name in raw.keys()]
        else:
            tables = []
        catalog[str(source.id)] = {"title": source.title or "", "tables": tables}
    return catalog


def load_catalog(path: Path | None, source_ids: list[int]) -> dict[str, dict]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"{path} is not a catalog object")
        return payload
    return load_catalog_from_django(source_ids)


def resolve_rows(
    pool: list[dict],
    signatures: dict[frozenset[str], str],
    catalog: dict[str, dict],
) -> list[dict]:
    by_source: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        by_source[int(row["data_source_id"])].append(row)
    source_db: dict[int, str] = {}
    source_tables: dict[int, frozenset[str]] = {}
    for source_id, rows in by_source.items():
        info = catalog.get(str(source_id)) or {}
        tables = table_set(info.get("tables") or [])
        if not tables:
            tables = frozenset().union(*(sql_tables(row["ideal_sql"]) for row in rows))
        source_tables[source_id] = tables
        # Exact table-set equality only. A shared table name is not a database.
        source_db[source_id] = signatures.get(tables, "")
    resolved = []
    for row in pool:
        source_id = int(row["data_source_id"])
        mentioned = sql_tables(row.get("ideal_sql") or "")
        db_id = source_db[source_id]
        note = "catalog table set"
        if db_id and mentioned and not mentioned <= source_tables[source_id]:
            note = "sql tables are outside the source catalog"
        title = (catalog.get(str(source_id)) or {}).get("title") or ""
        label = db_id or f"unresolved:{source_id}"
        resolved.append(
            {
                "id": row["id"],
                "data_source_id": source_id,
                "title": title,
                "db_id": db_id,
                "label": label,
                "note": note,
                "sql_tables": sorted(mentioned),
            }
        )
    return resolved


def sql_hits(pool_rows: list[dict], gold: list[dict], db_ids: set[str]) -> list[dict]:
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in gold:
        if item["db_id"] not in db_ids:
            continue
        index[(item["db_id"], normalize_sql(item["query"]))].append(item)
    hits = []
    for row in pool_rows:
        if row["db_id"] not in db_ids:
            continue
        key = (row["db_id"], normalize_sql(row["ideal_sql"]))
        matches = index.get(key) or []
        if matches:
            hits.append({"pool": row, "gold": matches[0], "n_gold": len(matches)})
    return hits


def embed_scores(left: list[str], right: list[str]) -> tuple[list[tuple[int, float]], str] | None:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def pack(texts: list[str]):
        vectors = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    left_v = pack(left)
    right_v = pack(right)
    scores = left_v @ right_v.T
    best = scores.argmax(axis=1)
    return [(int(index), float(scores[row, index])) for row, index in enumerate(best)], (
        "embedding cosine, sentence-transformers/all-mpnet-base-v2"
    )


def jaccard_scores(left: list[str], right: list[str]) -> tuple[list[tuple[int, float]], str]:
    right_sets = [content_tokens(text) for text in right]
    chosen: list[tuple[int, float]] = []
    for text in left:
        tokens = content_tokens(text)
        best_i = 0
        best_s = 0.0
        for index, other in enumerate(right_sets):
            score = jaccard(tokens, other)
            if score > best_s:
                best_i = index
                best_s = score
        chosen.append((best_i, best_s))
    return chosen, "content-word Jaccard (sentence-transformers is not installed)"


def best_matches(left: list[str], right: list[str]) -> tuple[list[tuple[int, float]], str]:
    embedded = embed_scores(left, right)
    if embedded is not None:
        return embedded
    return jaccard_scores(left, right)


def distribution(scores: list[float]) -> list[tuple[str, int]]:
    bins = [(index / 10, (index + 1) / 10) for index in range(10)]
    counts = []
    for low, high in bins:
        if high == 1.0:
            count = sum(1 for score in scores if low <= score <= high or score > 1)
        else:
            count = sum(1 for score in scores if low <= score < high)
        counts.append((f"{low:.1f}–{high:.1f}", count))
    return counts


def md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def samples_md(hits: list[dict], questions_en: dict[int, str], limit: int) -> str:
    lines = []
    for number, hit in enumerate(hits[:limit], start=1):
        pool = hit["pool"]
        gold = hit["gold"]
        question = questions_en.get(pool["id"], pool["user_question"])
        lines.append(f"#### {number}. pool {pool['id']} / `{pool['db_id']}`")
        lines.append("")
        lines.append(f"- pool question: {md(question)}")
        lines.append(f"- dev/train question: {md(gold['question'])}")
        lines.append("")
        lines.append("```sql")
        lines.append(pool["ideal_sql"].strip())
        lines.append("```")
        lines.append("")
        lines.append("```sql")
        lines.append(gold["query"].strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def high_pairs_md(
    pool: list[dict],
    other: list[dict],
    matches: list[tuple[int, float]],
    questions_en: dict[int, str],
    threshold: float,
) -> str:
    lines = []
    for row_index, (other_index, score) in enumerate(matches):
        if score <= threshold:
            continue
        row = pool[row_index]
        gold = other[other_index]
        question = questions_en.get(row["id"], row["user_question"])
        lines.append(
            f"- {score:.4f} pool {row['id']} db `{row.get('db_id') or row.get('label')}`: "
            f"{md(question)}"
        )
        lines.append(f"  - matched `{gold['db_id']}`: {md(gold['question'])}")
    if not lines:
        return "None.\n"
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--pool-en", type=Path, default=POOL_EN)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--fail-on-dev",
        action="store_true",
        help="Exit 1 when any dev-database row, dev-gold SQL match, or question score above the threshold remains.",
    )
    args = parser.parse_args()

    dev_path = args.dev or first_existing(DEV_CANDIDATES)
    train_path = args.train or first_existing(TRAIN_CANDIDATES)
    tables_path = args.tables or first_existing(TABLES_CANDIDATES)
    if dev_path is None or train_path is None or tables_path is None:
        raise SystemExit(
            "Need dev.json, train_spider.json, and Spider tables.json. "
            f"dev={dev_path} train={train_path} tables={tables_path}"
        )

    pool = load_json_any(args.pool)
    pool_en = load_json_any(args.pool_en)
    en_by_id = {int(row["id"]): str(row.get("user_question") or "") for row in pool_en}
    dev = spider_examples(dev_path)
    train = spider_examples(train_path)
    dev_ids = {row["db_id"] for row in dev}
    train_ids = {row["db_id"] for row in train}
    signatures = load_table_signatures(tables_path)
    source_ids = sorted({int(row["data_source_id"]) for row in pool})
    catalog = load_catalog(args.catalog, source_ids)
    resolved = resolve_rows(pool, signatures, catalog)
    by_id = {row["id"]: row for row in resolved}
    for row in pool:
        info = by_id[row["id"]]
        row["db_id"] = info["db_id"]
        row["label"] = info["label"]

    dev_rows = [row for row in pool if row["db_id"] in dev_ids]
    train_rows = [row for row in pool if row["db_id"] in train_ids]
    dev_sql = sql_hits(pool, dev, dev_ids)
    train_sql = sql_hits(pool, train, train_ids)

    translated = [en_by_id.get(int(row["id"]), row["user_question"]) for row in pool]
    dev_match, dev_method = best_matches(translated, [row["question"] for row in dev])
    train_match, train_method = best_matches(translated, [row["question"] for row in train])
    dev_high = sum(1 for _index, score in dev_match if score > args.threshold)
    train_high = sum(1 for _index, score in train_match if score > args.threshold)

    counts = Counter(row["label"] for row in resolved)
    dev_counts = Counter(row["db_id"] for row in dev_rows)
    census = ["| database | rows | data_source_ids | split |", "|---|---:|---|---|"]
    sources_for: dict[str, list[int]] = defaultdict(list)
    for info in resolved:
        if info["data_source_id"] not in sources_for[info["label"]]:
            sources_for[info["label"]].append(info["data_source_id"])
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if label in dev_ids:
            split = "dev"
        elif label in train_ids:
            split = "train"
        else:
            split = "other"
        ids = ", ".join(str(source_id) for source_id in sources_for[label])
        census.append(f"| `{label}` | {count} | {ids} | {split} |")

    dev_score_values = [score for _index, score in dev_match]
    train_score_values = [score for _index, score in train_match]

    def dist_table(scores: list[float]) -> str:
        lines = ["| best score | rows |", "|---|---:|"]
        for label, count in distribution(scores):
            lines.append(f"| {label} | {count} |")
        ordered = sorted(scores)
        if ordered:
            mid = ordered[len(ordered) // 2]
            lines.append("")
            lines.append(
                f"min {ordered[0]:.4f} / median {mid:.4f} / "
                f"mean {sum(ordered) / len(ordered):.4f} / max {ordered[-1]:.4f}"
            )
        return "\n".join(lines)

    dropped = len(pool) - len(dev_rows)
    leaked = bool(dev_rows or dev_sql or dev_high)
    verdict = (
        "NOT protocol-clean for Spider dev cross-domain evaluation."
        if leaked
        else "No pool row sits on a dev database, matches dev gold SQL, or scores above the question threshold."
    )
    flagged = ["| db_id | rows | pool ids |", "|---|---:|---|"]
    ids_for: dict[str, list[int]] = defaultdict(list)
    for row in dev_rows:
        ids_for[row["db_id"]].append(int(row["id"]))
    for db_id, count in sorted(dev_counts.items(), key=lambda item: (-item[1], item[0])):
        flagged.append(f"| `{db_id}` | {count} | {', '.join(str(i) for i in ids_for[db_id])} |")

    body = f"""# Pool forensics

{verdict}

The pool file was not modified. Question similarity method: {dev_method}.
Train used the same method: {train_method}.

| input | path | rows |
|---|---|---:|
| pool | `{args.pool}` | {len(pool)} |
| English questions | `{args.pool_en}` | {len(pool_en)} |
| dev | `{dev_path}` | {len(dev)} |
| train | `{train_path}` | {len(train)} |
| Spider tables | `{tables_path}` | {len(signatures)} databases |

Dev databases ({len(dev_ids)}): {", ".join(f"`{db_id}`" for db_id in sorted(dev_ids))}.
Train databases: {len(train_ids)}. Databases in both splits: {len(dev_ids & train_ids)}.

## 1. Database census

Each row inherits its `data_source_id`. That source is a Spider database only when its catalog table set equals the official table set (`sqlite_sequence` ignored). SQL is used to confirm the row's tables sit inside that set, not to guess a database from one shared table name.

N_dev_db_rows = **{len(dev_rows)}**.

{chr(10).join(census)}

### Rows on a Spider dev database

{chr(10).join(flagged) if dev_rows else "None."}

## 2. SQL overlap with dev gold

Normalized SQL: lowercase, quotes removed, whitespace collapsed, `AS` aliases removed, `alias.column` reduced to `column`, and a bare `T1`/`a` alias after a table name removed. Comparison is only against gold SQL on the same `db_id`.

Exact-match rows: **{len(dev_sql)}** of {len(dev_rows)} rows on dev databases.

{samples_md(dev_sql, en_by_id, args.samples) if dev_sql else "No exact SQL matches."}

## 3. Question laundering against dev

Each of the {len(pool)} translated pool questions against all {len(dev)} dev questions. The score kept is that pool row's best match.

{dist_table(dev_score_values)}

Pairs with similarity > {args.threshold:.2f}: **{dev_high}**.

{high_pairs_md(pool, dev, dev_match, en_by_id, args.threshold)}

## 4. Train provenance

Rows whose resolved database is a train database: **{len(train_rows)}**.
Exact normalized SQL against train gold on the same `db_id`: **{len(train_sql)}**.
Translated questions with train similarity > {args.threshold:.2f}: **{train_high}**.

{dist_table(train_score_values)}

### Train SQL matches

{samples_md(train_sql, en_by_id, args.samples) if train_sql else "No exact SQL matches."}

### Train question pairs above {args.threshold:.2f}

{high_pairs_md(pool, train, train_match, en_by_id, args.threshold)}

## 5. Verdict

| | dev | train |
|---|---:|---:|
| rows on that split's databases | {len(dev_rows)} | {len(train_rows)} |
| rows with gold-SQL exact match | {len(dev_sql)} | {len(train_sql)} |
| rows with question similarity > {args.threshold:.2f} | {dev_high} | {train_high} |

Pool size if every dev-database row is dropped: **{dropped}** (from {len(pool)}).

{verdict}
"""
    write_report(args.report, body)
    print(f"N_dev_db_rows={len(dev_rows)}")
    print(f"dev_sql={len(dev_sql)} train_sql={len(train_sql)}")
    print(f"dev_q>{args.threshold}={dev_high} train_q>{args.threshold}={train_high}")
    print(f"remaining_if_dev_dbs_dropped={dropped}")
    print(f"method={dev_method}")
    print(f"wrote {args.report}")
    if args.fail_on_dev and (dev_rows or dev_sql or dev_high):
        raise SystemExit(
            "CLEAN CHECK FAILED "
            f"dev_db_rows={len(dev_rows)} dev_sql={len(dev_sql)} "
            f"laundering>{args.threshold}={dev_high}"
        )


if __name__ == "__main__":
    main()
