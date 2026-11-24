#!/usr/bin/env python3
"""Build a Spider-train demonstration pool. This entry point is the dry run.

The dry run does not call a model and does not write a pool. It checks that
the official Spider files are present, labels every train gold SQL with the
runtime ``sql_skeleton`` band, and reports whether the configured quotas can
be filled after the excluded databases, normalized-question dedup, and the
per-database cap.

``question_skeleton`` is imported so the preflight uses the same function
selection applies to ``user_question``. The band labels come from
``sql_skeleton``, which is what the router uses on a draft SQL.

This script does not import or call the Spider eval runner.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_leakage import normalize_question
from data_service.agent.fewshot_skeleton import question_skeleton, sql_skeleton
from repro_config import load as load_repro_config

BANDS = ("easy", "medium", "hard")
OFFICIAL = (
    "https://yale-lily.github.io/spider "
    "(Drive file 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J, folder spider_data/)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Census and quota check only. No pool file and no model calls.",
    )
    args = parser.parse_args()
    cfg = load_repro_config()
    build = cfg["pool_build"]
    preflight(build)
    if not args.dry_run:
        raise SystemExit(
            "preflight passed. Re-run with --dry-run. "
            "The accept/execute build is not started by this step, and the "
            "Spider dev evaluation is not run."
        )
    report = dry_run(build)
    out = Path(cfg["paths"]["results_dir"]) / "pool_build_dryrun.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"wrote {out}")


def preflight(build: dict) -> None:
    """Stop if a required official artifact or the question skeleton is missing."""
    paths = build["paths"]
    missing: list[str] = []
    train = Path(paths["train_json"])
    tables = Path(paths["tables_json"])
    sqlite_dir = Path(paths["sqlite_dir"])
    if not train.is_file():
        missing.append(f"train_spider.json at {train} — official source {OFFICIAL}")
    if not tables.is_file():
        missing.append(f"tables.json at {tables} — official source {OFFICIAL}")
    sample = sqlite_dir / "department_management" / "department_management.sqlite"
    if not sample.is_file():
        missing.append(
            "Spider train sqlite directory "
            f"{sqlite_dir}/<db_id>/<db_id>.sqlite — official source {OFFICIAL}"
        )
    if not callable(question_skeleton):
        missing.append("question_skeleton in data_service/agent/fewshot_skeleton.py")
    if missing:
        raise SystemExit("PREFLIGHT FAILED\n" + "\n".join(missing))
    print("preflight ok")
    print(f"  question_skeleton = {inspect.getfile(question_skeleton)}")
    print(f"  train_json        = {train}")
    print(f"  tables_json       = {tables}")
    print(f"  sqlite_dir        = {sqlite_dir}")
    print("  eval runner       = not imported")


def dry_run(build: dict) -> str:
    train = json.loads(Path(build["paths"]["train_json"]).read_text(encoding="utf-8"))
    tables = json.loads(Path(build["paths"]["tables_json"]).read_text(encoding="utf-8"))
    excluded = set(build["excluded_databases"])
    cap = int(build["per_database_cap"])
    quotas = {band: int(build["quotas"][band]) for band in BANDS}

    census: Counter[str] = Counter()
    kept: list[tuple[str, str]] = []
    seen_questions: set[str] = set()
    dropped_excluded = 0
    dropped_dup = 0
    dropped_empty = 0
    for row in train:
        db_id = str(row["db_id"])
        question = str(row.get("question") or "")
        skeleton = sql_skeleton(str(row.get("query") or ""))
        band = skeleton.band if skeleton is not None else "unparsed"
        census[band] += 1
        if db_id in excluded:
            dropped_excluded += 1
            continue
        key = normalize_question(question)
        if not key:
            dropped_empty += 1
            continue
        if key in seen_questions:
            dropped_dup += 1
            continue
        seen_questions.add(key)
        if band in BANDS:
            kept.append((db_id, band))

    by_db: dict[str, Counter[str]] = defaultdict(Counter)
    for db_id, band in kept:
        by_db[db_id][band] += 1

    lines = [
        "Spider train pool dry run",
        "llm_calls = 0",
        "generate_sql = false",
        f"train_rows = {len(train)}",
        f"tables_databases = {len(tables)}",
        f"excluded_databases = {', '.join(sorted(excluded)) or '(none)'}",
        f"per_database_cap = {cap}",
        "",
        "band census (all train gold SQL, before exclusions)",
    ]
    for band in (*BANDS, "unparsed"):
        lines.append(f"  {band:8} {census[band]}")
    lines.append(f"  dropped_excluded = {dropped_excluded}")
    lines.append(f"  dropped_duplicate_question = {dropped_dup}")
    lines.append(f"  dropped_empty_question = {dropped_empty}")
    lines.append(f"  rows_after_exclude_and_dedup = {len(kept)}")
    lines.append("")
    lines.append("per-band counts after excluded DBs and normalized-question dedup")
    after = Counter(band for _db, band in kept)
    for band in BANDS:
        lines.append(f"  {band:8} {after[band]}")
    lines.append("")
    lines.append(
        "per-band upper bound if this band is filled alone "
        f"(at most {cap} rows from one database)"
    )
    solo = {}
    for band in BANDS:
        solo[band] = sum(min(cap, counts[band]) for counts in by_db.values())
        short = max(0, quotas[band] - solo[band])
        status = "reachable" if short == 0 else f"shortfall {short}"
        lines.append(f"  {band:8} available {solo[band]:4}  quota {quotas[band]:4}  {status}")
    joint = joint_flow(by_db, quotas, cap)
    lines.append("")
    lines.append(
        "joint reachability (one database contributes at most "
        f"{cap} rows across all bands)"
    )
    lines.append(f"  flow {joint} / required {sum(quotas.values())}")
    if joint == sum(quotas.values()):
        lines.append("  all quotas reachable together")
    else:
        lines.append(f"  joint shortfall {sum(quotas.values()) - joint}")
    lines.append("")
    lines.append("rows available per database after exclude and dedup")
    for db_id, counts in sorted(by_db.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        total = sum(counts.values())
        lines.append(
            f"  {db_id:32} {total:4}  "
            f"easy {counts['easy']:3} medium {counts['medium']:3} hard {counts['hard']:3}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def joint_flow(by_db: dict[str, Counter[str]], quotas: dict[str, int], cap: int) -> int:
    """Max rows placeable when each database is limited to ``cap`` rows total."""
    graph: dict[str, dict[str, int]] = defaultdict(dict)

    def add(src: str, dst: str, capacity: int) -> None:
        if capacity <= 0:
            return
        graph[src][dst] = graph[src].get(dst, 0) + capacity
        graph[dst].setdefault(src, 0)

    for db_id, counts in by_db.items():
        add("S", f"db:{db_id}", cap)
        for band in BANDS:
            add(f"db:{db_id}", f"band:{band}", int(counts[band]))
    for band in BANDS:
        add(f"band:{band}", "T", int(quotas[band]))

    total = 0
    while True:
        parent: dict[str, str | None] = {"S": None}
        queue: deque[str] = deque(["S"])
        while queue and "T" not in parent:
            node = queue.popleft()
            for nxt, capacity in graph[node].items():
                if capacity > 0 and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if "T" not in parent:
            return total
        path = []
        node = "T"
        while node != "S":
            prev = parent[node]
            path.append((prev, node))
            node = prev
        flow = min(graph[src][dst] for src, dst in path)
        for src, dst in path:
            graph[src][dst] -= flow
            graph[dst][src] += flow
        total += flow


if __name__ == "__main__":
    main()
