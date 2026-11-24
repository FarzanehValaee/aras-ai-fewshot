#!/usr/bin/env python3
"""Prove a few-shot pool contains nothing from the Spider dev split.

Four independent hash-set checks (pool vs Spider-dev), plus a source scan
for full Spider-dev questions embedded in prompts/templates. Exit 1 on any
hit; exit 0 with a clean "no leakage" report otherwise.

Examples:
  python check_leakage.py \\
      --pool fewshot_pool.json \\
      --train /path/to/train_spider.json \\
      --dev /path/to/dev.json

  python check_leakage.py --from-db --dev /tmp/spider_dev.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parent
HIT_CAP = 20
MIN_EMBEDDED_Q_LEN = 16
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

QUESTION_KEYS = ("question", "user_question", "q", "nl", "utterance")
SQL_KEYS = ("query", "gold_sql", "ideal_sql", "SQL", "sql", "gold")
DB_KEYS = ("db_id", "db", "database")

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "site-packages",
    "staticfiles",
    "media",
    "export_to_windows",
    "paper_result",
    "paper_runs",
    "paper_runs_flash",
    "bird_runs",
    "bird_runs_reasoning",
    "bird_runs_reasoning_no_evidence",
}
SKIP_FILE_NAMES = {"check_leakage.py"}
SOURCE_SUFFIXES = {".py", ".jinja", ".j2", ".txt", ".md"}

POOL_CANDIDATES = [
    ROOT / "fewshot_pool.json",
    ROOT / "fewshot_pool.jsonl",
    ROOT / "benchmark" / "fewshot_pool.json",
    ROOT / "benchmark" / "fewshot_pool.jsonl",
    Path("/home/aras-lab/my-project/bird_pool.json"),
]
TRAIN_CANDIDATES = [
    ROOT / "data" / "spider" / "spider_data" / "train_spider.json",
    ROOT / "benchmark" / "spider" / "train_spider.json",
    ROOT / "spider" / "train_spider.json",
    Path(os.environ.get("SPIDER_TRAIN", "")),
    Path(os.environ.get("SPIDER_DIR", "")) / "train_spider.json",
]
DEV_CANDIDATES = [
    ROOT / "data" / "spider" / "spider_data" / "dev.json",
    ROOT / "benchmark" / "spider" / "dev.json",
    ROOT / "spider" / "dev.json",
    Path(os.environ.get("SPIDER_DEV", "")),
    Path(os.environ.get("SPIDER_DIR", "")) / "dev.json",
    Path("/tmp/spider_dev.jsonl"),
    Path("/home/aras-lab/my-project/spider_dev_questions.json"),
]


@dataclass(frozen=True)
class Example:
    origin: str
    index: int
    question: str
    db_id: str
    sql: str
    source_id: int | None = None


@dataclass
class CheckResult:
    name: str
    hits: list[str] = field(default_factory=list)
    count: int = 0

    @property
    def leaked(self) -> bool:
        return self.count > 0


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path and str(path) not in (".", "") and path.is_file():
            return path
    return None


def load_json_or_jsonl(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    rows: list[Any] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSONL ({exc})") from exc
    return rows


def pick_str(obj: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def as_example(obj: dict, *, origin: str, index: int) -> Example | None:
    question = pick_str(obj, QUESTION_KEYS)
    if not question:
        return None
    source_id = None
    raw_source = obj.get("data_source_id")
    if raw_source is not None and str(raw_source).strip().lstrip("-").isdigit():
        source_id = int(raw_source)
    return Example(
        origin=origin,
        index=index,
        question=question,
        db_id=pick_str(obj, DB_KEYS),
        sql=pick_str(obj, SQL_KEYS),
        source_id=source_id,
    )


def walk_records(payload: Any, origin: str) -> list[Example]:
    found: list[Example] = []
    seen: set[tuple[str, str, str, str]] = set()
    counter = 0

    def consider(obj: dict) -> None:
        nonlocal counter
        example = as_example(obj, origin=origin, index=counter)
        if example is None:
            return
        key = (
            example.db_id,
            example.question,
            example.sql,
            example.origin,
            example.source_id,
        )
        if key in seen:
            return
        seen.add(key)
        found.append(example)
        counter += 1

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            consider(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def load_examples(path: Path, *, origin: str | None = None) -> list[Example]:
    return walk_records(load_json_or_jsonl(path), origin or str(path))


def load_pool_from_db() -> list[Example]:
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aras_ai_data.settings")
    try:
        import django
    except ImportError as exc:
        raise SystemExit(
            "--from-db requires Django. Pass --pool PATH instead."
        ) from exc
    django.setup()
    from data_service.models import FewShotExample

    rows = FewShotExample.objects.all().select_related("data_source")
    examples: list[Example] = []
    for i, row in enumerate(rows):
        question = (row.user_question or "").strip()
        if not question:
            continue
        db_id = ""
        source = getattr(row, "data_source", None)
        if source is not None:
            db_id = (getattr(source, "title", None) or str(source.id) or "").strip()
        examples.append(
            Example(
                origin=f"db:FewShotExample:{row.id}",
                index=i,
                question=question,
                db_id=db_id,
                sql=(row.ideal_sql or "").strip(),
                source_id=row.data_source_id,
            )
        )
    return examples


def const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    return None


def dict_str_map(node: ast.Dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key_node, value_node in zip(node.keys, node.values):
        key = const_str(key_node)
        value = const_str(value_node)
        if key is None or value is None:
            continue
        out[key] = value
    return out


def extract_hardcoded(path: Path) -> list[Example]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    found: list[Example] = []
    seen: set[tuple[str, str, str]] = set()
    origin = str(path)

    def consider(mapping: dict[str, str], lineno: int) -> None:
        example = as_example(mapping, origin=f"{origin}:{lineno}", index=lineno)
        if example is None:
            return
        key = (example.db_id, example.question, example.sql)
        if key in seen:
            return
        seen.add(key)
        found.append(example)

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            mapping = dict_str_map(node)
            if any(k in mapping for k in QUESTION_KEYS):
                consider(mapping, getattr(node, "lineno", 0))
        elif isinstance(node, ast.Call) and node.keywords:
            mapping = {}
            for kw in node.keywords:
                if not kw.arg:
                    continue
                value = const_str(kw.value)
                if value is not None:
                    mapping[kw.arg] = value
            if any(k in mapping for k in QUESTION_KEYS):
                consider(mapping, getattr(node, "lineno", 0))
    return found


def iter_source_files(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            if root.name not in SKIP_FILE_NAMES:
                yield root
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_DIR_NAMES and not name.startswith(".")
            ]
            for name in filenames:
                if name in SKIP_FILE_NAMES:
                    continue
                path = Path(dirpath) / name
                if path.suffix.lower() in SOURCE_SUFFIXES:
                    yield path


def normalize_question(text: str) -> str:
    lowered = text.lower().translate(PUNCT_TABLE)
    return " ".join(lowered.split())


_SQL_WS = re.compile(r"\s+")
_SQL_COMMA = re.compile(r"\s*,\s*")
_SQL_QUOTES = re.compile(r"[`\"\[\]]")


def normalize_sql(text: str) -> str:
    """Case/whitespace/quote-insensitive SQL; keeps operators and star."""
    lowered = text.lower().replace(";", " ")
    lowered = _SQL_QUOTES.sub("", lowered)
    lowered = _SQL_COMMA.sub(",", lowered)
    return _SQL_WS.sub(" ", lowered).strip()


def index_by_hash(items: list[Example], key_fn) -> dict[str, list[Example]]:
    buckets: dict[str, list[Example]] = {}
    for item in items:
        key = key_fn(item)
        if not key:
            continue
        buckets.setdefault(sha(key), []).append(item)
    return buckets


def collect_hits(
    name: str,
    pool: list[Example],
    index: dict[str, list[Example]],
    key_fn,
) -> CheckResult:
    result = CheckResult(name=name)
    seen_pair: set[tuple[str, str, str, str]] = set()
    for item in pool:
        key = key_fn(item)
        if not key:
            continue
        matches = index.get(sha(key), [])
        for other in matches:
            stamp = (item.origin, item.question, other.origin, other.question)
            if stamp in seen_pair:
                continue
            seen_pair.add(stamp)
            result.count += 1
            if len(result.hits) < HIT_CAP:
                extra = ""
                if "sql" in name and item.sql:
                    extra = f" sql={item.sql!r}"
                result.hits.append(
                    f"pool[{item.origin}#{item.index}] db={item.db_id!r} "
                    f"q={item.question!r}{extra}  <->  "
                    f"dev[{other.origin}#{other.index}] db={other.db_id!r} "
                    f"q={other.question!r}"
                )
    return result


def scan_embedded_questions(
    source_files: list[Path],
    dev: list[Example],
    *,
    min_len: int,
) -> CheckResult:
    result = CheckResult(name="source_embedded_dev_question")
    unique: dict[str, Example] = {}
    for item in dev:
        if len(item.question) < min_len:
            continue
        unique.setdefault(item.question, item)

    corpus: list[tuple[Path, str]] = []
    for path in source_files:
        try:
            corpus.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue

    for question, item in unique.items():
        for path, text in corpus:
            if question not in text:
                continue
            result.count += 1
            if len(result.hits) < HIT_CAP:
                line_no = 0
                for i, line in enumerate(text.splitlines(), start=1):
                    if question in line:
                        line_no = i
                        break
                loc = f"{path}:{line_no}" if line_no else str(path)
                result.hits.append(
                    f"{loc} contains dev[{item.origin}#{item.index}] "
                    f"db={item.db_id!r} q={question!r}"
                )
            break
    return result


def print_check(result: CheckResult) -> None:
    status = "HIT" if result.leaked else "clean"
    print(f"\n[{result.name}] {status}  count={result.count}")
    for line in result.hits:
        print(f"  - {line}")
    if result.count > len(result.hits):
        print(f"  ... {result.count - len(result.hits)} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="Few-shot pool JSON/JSONL (array, object, or nested maps).",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load the pool from Django FewShotExample rows.",
    )
    parser.add_argument("--train", type=Path, default=None, help="Spider train_spider.json")
    parser.add_argument("--dev", type=Path, default=None, help="Spider dev.json or JSONL")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="Source root to scan (repeatable). Defaults to data_service, "
        "benchmark/*.py, scripts.",
    )
    parser.add_argument(
        "--min-embedded-len",
        type=int,
        default=MIN_EMBEDDED_Q_LEN,
        help="Ignore shorter Spider-dev questions in the source substring scan.",
    )
    return parser.parse_args()


def default_source_roots() -> list[Path]:
    roots = [ROOT / "data_service", ROOT / "scripts"]
    bench = ROOT / "benchmark"
    if bench.is_dir():
        roots.extend(sorted(bench.glob("*.py")))
    return [path for path in roots if path.exists()]


def resolve_inputs(args: argparse.Namespace) -> tuple[list[Example], list[Example], list[Example], list[Path]]:
    pool_path = args.pool or (
        None if args.from_db else first_existing(POOL_CANDIDATES)
    )
    train_path = args.train or first_existing(TRAIN_CANDIDATES)
    dev_path = args.dev or first_existing(DEV_CANDIDATES)

    if not args.from_db and pool_path is None:
        raise SystemExit(
            "few-shot pool not found. Pass --pool PATH or --from-db."
        )
    if train_path is None:
        raise SystemExit(
            "Spider train_spider.json not found. Pass --train PATH "
            "(or set SPIDER_DIR / SPIDER_TRAIN)."
        )
    if dev_path is None:
        raise SystemExit(
            "Spider dev.json not found. Pass --dev PATH (or set SPIDER_DIR / SPIDER_DEV)."
        )

    pool: list[Example] = []
    if args.from_db:
        pool.extend(load_pool_from_db())
    if pool_path is not None:
        pool.extend(load_examples(pool_path))
    train = load_examples(train_path, origin=str(train_path))
    dev = load_examples(dev_path, origin=str(dev_path))
    if not pool:
        raise SystemExit("few-shot pool is empty after parsing.")
    if not train:
        raise SystemExit(f"train split is empty: {train_path}")
    if not dev:
        raise SystemExit(f"dev split is empty: {dev_path}")

    source_roots = args.source or default_source_roots()
    source_files = list(iter_source_files(source_roots))
    hardcoded: list[Example] = []
    for path in source_files:
        if path.suffix.lower() != ".py":
            continue
        hardcoded.extend(extract_hardcoded(path))

    print("sanity")
    print(f"  pool_file          = {pool_path or '(django FewShotExample)'}")
    print(f"  train_file         = {train_path}")
    print(f"  dev_file           = {dev_path}")
    print(f"  pool_size          = {len(pool)}")
    print(f"  train_size         = {len(train)}")
    print(f"  dev_size           = {len(dev)}")
    print(f"  hardcoded_examples = {len(hardcoded)}")
    print(f"  source_files       = {len(source_files)}")
    return pool + hardcoded, train, dev, source_files


def _membership_keys(item: Example) -> list[str]:
    keys = [item.question, normalize_question(item.question)]
    if item.db_id and item.question:
        keys.append(f"{item.db_id}\0{item.question}")
    if item.sql:
        keys.append(normalize_sql(item.sql))
    return [key for key in keys if key]


def target_database_census(
    pool: list[Example], dev_ids: set[str]
) -> tuple[list[tuple[str, int]], list[tuple[Example, str]]]:
    """Resolve each pool row with the same table-set match as pool_forensics.

    A row is on a dev database only when its data-source catalog equals that
    database's official tables (sqlite_sequence ignored). Returns the census
    and the contaminated rows.
    """
    from pool_forensics import (
        TABLES_CANDIDATES,
        first_existing,
        load_catalog_from_django,
        load_table_signatures,
        resolve_rows,
    )

    tables_path = first_existing(TABLES_CANDIDATES)
    if tables_path is None:
        raise SystemExit(
            "target-database check needs Spider tables.json. "
            "Set SPIDER_TABLES or place tables.json where pool_forensics looks."
        )
    signatures = load_table_signatures(tables_path)
    source_ids = sorted({item.source_id for item in pool if item.source_id is not None})
    catalog: dict = {}
    if source_ids:
        catalog = load_catalog_from_django(source_ids)
    records = []
    for index, item in enumerate(pool):
        source_id = item.source_id if item.source_id is not None else -(index + 1)
        records.append(
            {
                "id": item.index,
                "data_source_id": source_id,
                "ideal_sql": item.sql,
            }
        )
    resolved = resolve_rows(records, signatures, catalog)
    counts: dict[str, int] = {}
    hits: list[tuple[Example, str]] = []
    for item, info in zip(pool, resolved):
        label = str(info["label"])
        counts[label] = counts.get(label, 0) + 1
        db_id = str(info["db_id"] or "")
        if db_id and db_id in dev_ids:
            hits.append((item, db_id))
    census = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return census, hits


def print_target_census(
    census: list[tuple[str, int]], hits: list[tuple[Example, str]]
) -> None:
    print("\n[target_database] census")
    for label, count in census:
        print(f"  {label} {count}")
    print(f"  dev-database rows = {len(hits)}")


def assert_pool_subset_of_train(
    pool: list[Example],
    train: list[Example],
    dev: list[Example] | None = None,
) -> None:
    """Fail unless every pool item matches the train split and none match dev.

    A pool item is in a split when any check_leakage key matches: exact
    question, normalized question, (db_id, question), or normalized SQL.
    Dev hits are reported with the same collectors as the CLI.
    """
    train_keys: set[str] = set()
    for item in train:
        for key in _membership_keys(item):
            train_keys.add(sha(key))
    outside: list[Example] = []
    for item in pool:
        if not any(sha(key) in train_keys for key in _membership_keys(item)):
            outside.append(item)
    dev_hits: list[CheckResult] = []
    if dev:
        exact_index = index_by_hash(dev, lambda item: item.question)
        norm_index = index_by_hash(dev, lambda item: normalize_question(item.question))
        pair_index = index_by_hash(
            dev,
            lambda item: f"{item.db_id}\0{item.question}" if item.db_id and item.question else "",
        )
        sql_index = index_by_hash(
            [item for item in dev if item.sql], lambda item: normalize_sql(item.sql)
        )
        dev_hits = [
            collect_hits("exact_question", pool, exact_index, lambda item: item.question),
            collect_hits(
                "normalized_question",
                pool,
                norm_index,
                lambda item: normalize_question(item.question),
            ),
            collect_hits(
                "db_id_question_pair",
                pool,
                pair_index,
                lambda item: f"{item.db_id}\0{item.question}" if item.db_id and item.question else "",
            ),
            collect_hits(
                "normalized_gold_sql",
                [item for item in pool if item.sql],
                sql_index,
                lambda item: normalize_sql(item.sql),
            ),
        ]
    leaked = [check for check in dev_hits if check.leaked]
    dev_ids = {item.db_id for item in dev or [] if item.db_id}
    census, dev_db_hits = target_database_census(pool, dev_ids)
    print_target_census(census, dev_db_hits)
    if not outside and not leaked and not dev_db_hits:
        print(
            f"pool ⊆ train  pool={len(pool)} train={len(train)} "
            f"dev={0 if dev is None else len(dev)}"
        )
        return
    print("POOL IS NOT A SUBSET OF TRAIN", file=sys.stderr)
    if outside:
        print(f"outside train: {len(outside)}", file=sys.stderr)
        for item in outside[:20]:
            print(
                f"  - {item.origin}#{item.index} db={item.db_id!r} q={item.question!r}",
                file=sys.stderr,
            )
    for check in leaked:
        print(f"dev hit [{check.name}] count={check.count}", file=sys.stderr)
        for line in check.hits:
            print(f"  - {line}", file=sys.stderr)
    if dev_db_hits:
        print(
            f"dev database [{len(dev_db_hits)} rows whose target schema is a Spider dev database]",
            file=sys.stderr,
        )
        for item, db_id in dev_db_hits[:20]:
            print(
                f"  - {item.origin}#{item.index} target={db_id} q={item.question!r}",
                file=sys.stderr,
            )
    raise SystemExit(1)


def main() -> int:
    args = parse_args()
    pool, _train, dev, source_files = resolve_inputs(args)

    exact_index = index_by_hash(dev, lambda x: x.question)
    norm_index = index_by_hash(dev, lambda x: normalize_question(x.question))
    pair_index = index_by_hash(
        dev, lambda x: f"{x.db_id}\0{x.question}" if x.db_id and x.question else ""
    )
    sql_index = index_by_hash(
        [item for item in dev if item.sql], lambda x: normalize_sql(x.sql)
    )

    checks = [
        collect_hits("exact_question", pool, exact_index, lambda x: x.question),
        collect_hits(
            "normalized_question",
            pool,
            norm_index,
            lambda x: normalize_question(x.question),
        ),
        collect_hits(
            "db_id_question_pair",
            pool,
            pair_index,
            lambda x: f"{x.db_id}\0{x.question}" if x.db_id and x.question else "",
        ),
        collect_hits(
            "normalized_gold_sql",
            [item for item in pool if item.sql],
            sql_index,
            lambda x: normalize_sql(x.sql),
        ),
        scan_embedded_questions(
            source_files, dev, min_len=max(1, args.min_embedded_len)
        ),
    ]

    for check in checks:
        print_check(check)

    dev_ids = {item.db_id for item in dev if item.db_id}
    census, dev_db_hits = target_database_census(pool, dev_ids)
    print_target_census(census, dev_db_hits)
    if dev_db_hits:
        checks.append(
            CheckResult(
                name="dev_database",
                hits=[
                    f"{item.origin}#{item.index} target={db_id} q={item.question}"
                    for item, db_id in dev_db_hits[:HIT_CAP]
                ],
                count=len(dev_db_hits),
            )
        )
        print_check(checks[-1])

    leaked = [check.name for check in checks if check.leaked]
    print()
    if leaked:
        print("LEAKAGE DETECTED in: " + ", ".join(leaked))
        return 1
    print("no leakage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
