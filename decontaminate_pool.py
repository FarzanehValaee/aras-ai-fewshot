#!/usr/bin/env python3
"""Drop Spider-dev material from the few-shot pool.

Writes new pool files and the drop lists. Does not modify
benchmark/fewshot_pool.json or benchmark/fewshot_pool_en.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pool_forensics import (  # noqa: E402
    best_matches,
    load_json_any,
    load_table_signatures,
    resolve_rows,
    spider_examples,
    table_set,
)

POOL = ROOT / "benchmark" / "fewshot_pool.json"
POOL_EN = ROOT / "benchmark" / "fewshot_pool_en.json"
CLEAN = ROOT / "benchmark" / "fewshot_pool_clean.json"
CLEAN_EN = ROOT / "benchmark" / "fewshot_pool_clean_en.json"
LAUNDER = ROOT / "results" / "extra_drop_laundering.txt"
NEAR = ROOT / "results" / "near_miss_schema_drops.md"
CATALOG = Path("/tmp/pool_catalog.json")
TABLES = ROOT / "data" / "spider" / "spider_data" / "tables.json"
DEV = ROOT / "data" / "spider" / "spider_data" / "dev.json"
BIRD_DEV = ROOT / "benchmark" / "bird" / "dev.json"
THRESHOLD = 0.85
SCHEMA_OVERLAP = 0.80


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def band_of(sql: str) -> str:
    from data_service.agent.fewshot_skeleton import sql_skeleton

    skeleton = sql_skeleton(sql or "")
    if skeleton is None:
        return "unparsed"
    return skeleton.band


def loud(text: str) -> None:
    bar = "!" * 60
    print(f"\n{bar}\nWARNING: {text}\n{bar}\n", flush=True)


def main() -> None:
    pool = load_json_any(POOL)
    pool_en = load_json_any(POOL_EN)
    if [row["id"] for row in pool] != [row["id"] for row in pool_en]:
        raise SystemExit("pool and English pool ids differ; refusing to drop")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    signatures = load_table_signatures(TABLES)
    dev = spider_examples(DEV)
    dev_ids = {row["db_id"] for row in dev}
    dev_sets = {db_id: tables for tables, db_id in signatures.items() if db_id in dev_ids}
    resolved = resolve_rows(pool, signatures, catalog)
    by_id = {info["id"]: info for info in resolved}

    dev_drop = {info["id"] for info in resolved if info["db_id"] in dev_ids}
    translated = [row["user_question"] for row in pool_en]
    matches, method = best_matches(translated, [row["question"] for row in dev])
    launder: list[tuple[int, float, str]] = []
    for row, (dev_index, score) in zip(pool, matches):
        if score <= THRESHOLD or row["id"] in dev_drop:
            continue
        launder.append((int(row["id"]), score, dev[dev_index]["question"]))
    launder.sort()
    LAUNDER.parent.mkdir(parents=True, exist_ok=True)
    LAUNDER.write_text(
        "".join(f"{row_id}\n" for row_id, _score, _question in launder),
        encoding="utf-8",
    )
    print(f"laundering extras={len(launder)} method={method} wrote {LAUNDER}")

    launder_ids = {row_id for row_id, _score, _question in launder}
    surviving = [row for row in pool if row["id"] not in dev_drop and row["id"] not in launder_ids]
    by_source: dict[int, list[dict]] = defaultdict(list)
    for row in surviving:
        by_source[int(row["data_source_id"])].append(row)

    near_ids: set[int] = set()
    near_blocks: list[str] = [
        "# Near-miss schema drops",
        "",
        "Overlap is `|schema ∩ dev| / |schema ∪ dev|` after dropping `sqlite_sequence`.",
        f"A schema at or above {SCHEMA_OVERLAP:.0%} against any Spider dev database is dropped.",
        "Dev-database schemas and laundering-only rows are already out of this pass.",
        "",
    ]
    for source_id, rows in sorted(by_source.items()):
        tables = table_set((catalog.get(str(source_id)) or {}).get("tables") or [])
        title = (catalog.get(str(source_id)) or {}).get("title") or ""
        best_db = ""
        best_score = 0.0
        best_shared: list[str] = []
        for db_id, dev_tables in dev_sets.items():
            score = overlap(tables, dev_tables)
            if score > best_score:
                best_db = db_id
                best_score = score
                best_shared = sorted(tables & dev_tables)
        if best_score < SCHEMA_OVERLAP:
            continue
        ids = [int(row["id"]) for row in rows]
        near_ids.update(ids)
        near_blocks.append(f"## source {source_id} `{title}`")
        near_blocks.append("")
        near_blocks.append(f"- overlap {best_score:.3f} with dev `{best_db}`")
        near_blocks.append(f"- shared tables: {', '.join(best_shared) or '(none)'}")
        near_blocks.append(f"- schema tables: {', '.join(sorted(tables))}")
        near_blocks.append(f"- dropped rows ({len(ids)}): {', '.join(str(i) for i in ids)}")
        near_blocks.append("")
    if not near_ids:
        near_blocks.append("No remaining schema met the 80% overlap rule. Nothing dropped.")
        near_blocks.append("")
    NEAR.write_text("\n".join(near_blocks), encoding="utf-8")
    print(f"near-miss rows={len(near_ids)} wrote {NEAR}")

    drop = dev_drop | launder_ids | near_ids
    clean = [row for row in pool if row["id"] not in drop]
    clean_en = [row for row in pool_en if row["id"] not in drop]
    if [row["id"] for row in clean] != [row["id"] for row in clean_en]:
        raise SystemExit("clean files diverged")
    dump(CLEAN, clean)
    dump(CLEAN_EN, clean_en)
    print(
        f"dropped dev_db={len(dev_drop)} launder={len(launder_ids)} "
        f"near={len(near_ids)} union={len(drop)} kept={len(clean)}"
    )
    print(f"wrote {CLEAN}")
    print(f"wrote {CLEAN_EN}")

    before = Counter(band_of(row["ideal_sql"]) for row in pool)
    after = Counter(band_of(row["ideal_sql"]) for row in clean)
    print("\nclass coverage (skeleton band)")
    print(f"{'class':<12} {'before':>8} {'after':>8}")
    for name in ("easy", "medium", "hard", "unparsed"):
        print(f"{name:<12} {before[name]:8d} {after[name]:8d}")
        if name != "unparsed" and after[name] < 50:
            loud(f"class {name} ends at {after[name]} rows, below 50.")

    bird_dev_ids = {row["db_id"] for row in load_json_any(BIRD_DEV)}
    bird_counts: dict[tuple[str, str], int] = Counter()
    for row in clean:
        info = by_id[row["id"]]
        title = info["title"] or ""
        schema = ""
        if title.startswith("bird:"):
            schema = title.split(":", 1)[1]
        elif info["db_id"] in bird_dev_ids:
            schema = info["db_id"]
        if not schema:
            continue
        split = "BIRD-dev" if schema in bird_dev_ids else "BIRD-train"
        bird_counts[(schema, split)] += 1
    print("\nBIRD provenance among remaining rows")
    if not bird_counts:
        print("no remaining row is on a BIRD schema")
    dev_bird_rows = 0
    for (schema, split), count in sorted(bird_counts.items()):
        print(f"  {schema} {split} {count}")
        if split == "BIRD-dev":
            dev_bird_rows += count
    if dev_bird_rows:
        loud(
            f"{dev_bird_rows} remaining rows sit on a BIRD-dev schema. "
            "This pool cannot be reused for a BIRD-dev evaluation without another purge."
        )


if __name__ == "__main__":
    main()
