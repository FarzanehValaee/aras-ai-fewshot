#!/usr/bin/env python3
"""Accuracy by difficulty band + execution-status counts from Spider paper_runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/var/www/aras-ai-data")
PAPER_RUNS = ROOT / "benchmark" / "paper_runs"
MODES = ("nofew", "jaccard", "skeleton")
MODE_LABELS = {
    "nofew": "No-Few",
    "jaccard": "Jaccard",
    "skeleton": "Skeleton",
}
BANDS = ("easy", "medium", "hard")
BAND_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
STATUS_COLS = ("match", "result_mismatch", "error", "blocked", "no_sql")
SEEDS = (1, 2, 3)
MIN_COMPLETE = 1034


def _bootstrap_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aras_ai_data.settings")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import django

    django.setup()


def discover_completed(root: Path) -> list[tuple[str, int, Path]]:
    found: list[tuple[str, int, Path]] = []
    for mode in MODES:
        for seed in SEEDS:
            path = root / f"{mode}_seed{seed}.jsonl"
            if not path.exists():
                continue
            n = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            if n >= MIN_COMPLETE:
                found.append((mode, seed, path))
    return found


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    parser.add_argument("--root", type=Path, default=PAPER_RUNS)
    args = parser.parse_args()

    _bootstrap_django()
    from data_service.agent.fewshot_skeleton import sql_skeleton

    files = discover_completed(args.root)
    if not files:
        raise SystemExit(f"no completed *_seed*.jsonl under {args.root}")

    # band -> mode -> [seed_accuracy_pct, ...]
    band_seed_acc: dict[str, dict[str, list[float]]] = {
        band: {mode: [] for mode in MODES} for band in BANDS
    }
    # mode -> status -> total count across completed seeds
    status_totals: dict[str, dict[str, int]] = {
        mode: {status: 0 for status in STATUS_COLS} for mode in MODES
    }
    band_n: dict[str, dict[str, list[int]]] = {
        band: {mode: [] for mode in MODES} for band in BANDS
    }

    print(f"Completed runs under `{args.root}`:")
    for mode, seed, path in files:
        rows = load_rows(path)
        print(f"- {path.name} ({len(rows)} rows)")

        # Status counts (all rows; gold_error etc. ignored in the requested cols).
        for row in rows:
            reason = str(row.get("reason") or "")
            if reason in status_totals[mode]:
                status_totals[mode][reason] += 1

        # Band accuracy: only scorable rows (score is not None), band from gold SQL.
        by_band: dict[str, list[int]] = {band: [] for band in BANDS}
        for row in rows:
            if row.get("score") is None:
                continue
            gold = row.get("gold_sql") or ""
            skeleton = sql_skeleton(str(gold))
            if skeleton is None:
                continue
            band = skeleton.band
            if band not in by_band:
                continue
            by_band[band].append(int(row["score"]))

        for band in BANDS:
            scores = by_band[band]
            if not scores:
                continue
            acc = 100.0 * sum(scores) / len(scores)
            band_seed_acc[band][mode].append(acc)
            band_n[band][mode].append(len(scores))

    print()
    print("### 1) Average execution accuracy by method and difficulty band\n")
    print(
        "_Band = `sql_skeleton(gold_sql).band` "
        "(Easy / Medium / Hard). "
        "Accuracy averaged over completed seeds; only scorable rows "
        "(`score is not None`)._"
    )
    print()
    band_rows: list[list[str]] = []
    for mode in MODES:
        cells = [MODE_LABELS[mode]]
        for band in BANDS:
            vals = band_seed_acc[band][mode]
            ns = band_n[band][mode]
            if not vals:
                cells.append("—")
                continue
            mean = sum(vals) / len(vals)
            n_mean = sum(ns) / len(ns)
            cells.append(f"{mean:.2f}% (n≈{n_mean:.0f}/seed)")
        band_rows.append(cells)
    print(
        markdown_table(
            band_rows,
            ["Method", "Easy", "Medium", "Hard"],
        )
    )

    print()
    print("### 2) Execution status totals by method\n")
    print(
        "_Counts summed across all completed seeds. "
        "Columns requested for the paper error analysis; "
        "`gold_error` / other reasons are omitted from this table._"
    )
    print()
    status_rows: list[list[str]] = []
    for mode in MODES:
        cells = [MODE_LABELS[mode]]
        for status in STATUS_COLS:
            cells.append(str(status_totals[mode][status]))
        status_rows.append(cells)
    # Totals row
    status_rows.append(
        ["**Total**"]
        + [
            str(sum(status_totals[mode][status] for mode in MODES))
            for status in STATUS_COLS
        ]
    )
    print(
        markdown_table(
            status_rows,
            ["Method", *STATUS_COLS],
        )
    )


if __name__ == "__main__":
    main()
