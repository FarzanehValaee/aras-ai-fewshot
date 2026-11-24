#!/usr/bin/env python3
"""Aggregate execution-guided SQL retry stats from Spider paper_runs jsonl files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/var/www/aras-ai-data/benchmark/paper_runs")
MODES = ("nofew", "jaccard", "skeleton")
MODE_LABELS = {
    "nofew": "No-Few (Baseline)",
    "jaccard": "Jaccard Few-Shot",
    "skeleton": "Skeleton Few-Shot (Ours)",
}
SEEDS = (1, 2, 3)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    scored = sum(1 for row in rows if row.get("score") is not None)
    attempted = sum(1 for row in rows if row.get("sql_retry_attempted"))
    fixed = sum(1 for row in rows if row.get("sql_retry_fixed"))
    rate = (100.0 * fixed / attempted) if attempted else 0.0
    return {
        "total": total,
        "scored": scored,
        "attempted": attempted,
        "fixed": fixed,
        "success_pct": round(rate, 2),
        "complete": total >= 1034,
    }


def discover_files(root: Path) -> list[tuple[str, int, Path]]:
    found: list[tuple[str, int, Path]] = []
    for mode in MODES:
        for seed in SEEDS:
            path = root / f"{mode}_seed{seed}.jsonl"
            if path.exists() and path.stat().st_size > 0:
                found.append((mode, seed, path))
    return found


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Directory containing *_seed*.jsonl files",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        default=True,
        help="Include incomplete runs (<1034 rows). Default: yes.",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Only aggregate runs with >=1034 rows.",
    )
    args = parser.parse_args()
    include_partial = not args.completed_only

    files = discover_files(args.root)
    if not files:
        raise SystemExit(f"no *_seed*.jsonl files found under {args.root}")

    per_seed: dict[tuple[str, int], dict] = {}
    per_mode: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "scored": 0, "attempted": 0, "fixed": 0, "runs": 0}
    )

    print(f"Root: `{args.root}`\n")
    seed_rows: list[list[str]] = []
    for mode, seed, path in files:
        stats = summarize(load_rows(path))
        if not include_partial and not stats["complete"]:
            print(
                f"- skipped partial `{path.name}` "
                f"({stats['total']}/1034 rows)",
                flush=True,
            )
            continue
        per_seed[(mode, seed)] = stats
        agg = per_mode[mode]
        agg["total"] += stats["total"]
        agg["scored"] += stats["scored"]
        agg["attempted"] += stats["attempted"]
        agg["fixed"] += stats["fixed"]
        agg["runs"] += 1
        status = "complete" if stats["complete"] else f"partial {stats['total']}/1034"
        seed_rows.append(
            [
                MODE_LABELS[mode],
                str(seed),
                status,
                str(stats["total"]),
                str(stats["attempted"]),
                str(stats["fixed"]),
                f"{stats['success_pct']:.2f}%",
            ]
        )

    if not seed_rows:
        raise SystemExit("no runs selected after filters")

    print("### Per-seed retry statistics\n")
    print(
        markdown_table(
            seed_rows,
            [
                "Method",
                "Seed",
                "Status",
                "Queries",
                "Retry attempted",
                "Retry fixed",
                "Success rate",
            ],
        )
    )
    print()

    method_rows: list[list[str]] = []
    grand = {"total": 0, "attempted": 0, "fixed": 0}
    for mode in MODES:
        agg = per_mode.get(mode)
        if not agg or agg["runs"] == 0:
            method_rows.append(
                [MODE_LABELS[mode], "—", "—", "—", "—", "—"]
            )
            continue
        rate = (100.0 * agg["fixed"] / agg["attempted"]) if agg["attempted"] else 0.0
        method_rows.append(
            [
                MODE_LABELS[mode],
                str(agg["runs"]),
                str(agg["total"]),
                str(agg["attempted"]),
                str(agg["fixed"]),
                f"{rate:.2f}%",
            ]
        )
        grand["total"] += agg["total"]
        grand["attempted"] += agg["attempted"]
        grand["fixed"] += agg["fixed"]

    overall_rate = (
        (100.0 * grand["fixed"] / grand["attempted"]) if grand["attempted"] else 0.0
    )
    method_rows.append(
        [
            "**Overall**",
            str(sum(1 for v in per_mode.values() if v["runs"])),
            str(grand["total"]),
            str(grand["attempted"]),
            str(grand["fixed"]),
            f"**{overall_rate:.2f}%**",
        ]
    )

    print("### Method-level retry efficiency\n")
    print(
        markdown_table(
            method_rows,
            [
                "Method",
                "Runs included",
                "Total queries",
                "Retry attempted",
                "Retry fixed",
                "Success rate (fixed/attempted)",
            ],
        )
    )
    print()
    print(
        f"_Success rate = sql_retry_fixed / sql_retry_attempted. "
        f"Overall: {grand['fixed']}/{grand['attempted']} "
        f"= {overall_rate:.2f}%._"
    )


if __name__ == "__main__":
    main()
