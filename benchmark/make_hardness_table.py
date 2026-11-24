#!/usr/bin/env python3
"""Print execution accuracy by official Spider hardness.

Reads a per-question JSONL written by the eval instrumentation
(results/run_<strategy>_<model>_<YYYYMMDD>.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BANDS = ("easy", "medium", "hard", "extra")
ROOT = Path("/var/www/aras-ai-data")
RESULTS = ROOT / "results"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def latest_log() -> Path:
    files = sorted(RESULTS.glob("run_*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise SystemExit(f"no run_*.jsonl under {RESULTS}")
    return files[-1]


def _ex(row: dict) -> int | None:
    value = row.get("ex")
    if value in (0, 1):
        return int(value)
    score = row.get("score")
    if score in (0, 1):
        return int(score)
    return None


def summarize(rows: list[dict]) -> list[tuple[str, int, int, float | None]]:
    groups: dict[str, list[int]] = {band: [] for band in BANDS}
    overall: list[int] = []
    for row in rows:
        ex = _ex(row)
        if ex is None:
            continue
        overall.append(ex)
        band = str(row.get("official_hardness") or "")
        if band in groups:
            groups[band].append(ex)
    table: list[tuple[str, int, int, float | None]] = []
    for band in (*BANDS, "overall"):
        values = overall if band == "overall" else groups[band]
        correct = sum(values)
        count = len(values)
        accuracy = (correct / count) if count else None
        table.append((band, count, correct, accuracy))
    return table


def render(path: Path, rows: list[dict]) -> str:
    unparsed = sum(
        1
        for row in rows
        if _ex(row) is not None and row.get("official_hardness") not in BANDS
    )
    lines = [
        f"file: {path}",
        f"questions: {sum(1 for row in rows if _ex(row) is not None)}",
        "",
        f"{'hardness':<10} {'n':>8} {'correct':>8} {'EX':>8}",
    ]
    for band, count, correct, accuracy in summarize(rows):
        shown = f"{100.0 * accuracy:.2f}" if accuracy is not None else "—"
        lines.append(f"{band:<10} {count:8d} {correct:8d} {shown:>8}")
    if unparsed:
        lines.append("")
        lines.append(f"missing official_hardness: {unparsed}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        help="Per-question JSONL. Defaults to the newest results/run_*.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.jsonl or latest_log()
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    print(render(path, load_rows(path)))


if __name__ == "__main__":
    main()
