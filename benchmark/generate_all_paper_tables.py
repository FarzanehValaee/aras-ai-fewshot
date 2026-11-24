#!/usr/bin/env python3
"""Generate all final Markdown tables for the Spider paper from paper_runs/."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/var/www/aras-ai-data")
PAPER_RUNS = ROOT / "benchmark" / "paper_runs"
SUMMARY_PATH = PAPER_RUNS / "summary_3x3.json"
OUT_PATH = PAPER_RUNS / "final_paper_tables.md"

MODES = ("nofew", "jaccard", "skeleton")
MODE_LABELS = {
    "nofew": "No-Shot",
    "jaccard": "Jaccard",
    "skeleton": "Skeleton",
}
BANDS = ("easy", "medium", "hard")
STATUS_COLS = ("match", "result_mismatch", "error", "blocked", "no_sql")
SEEDS = (1, 2, 3)
MIN_COMPLETE = 1034


def _bootstrap_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aras_ai_data.settings")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import django

    django.setup()


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


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


def mean_stdev(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mean = sum(values) / len(values)
    if len(values) < 2:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return (mean, math.sqrt(var))


def safe_bool(row: dict, key: str) -> bool:
    val = row.get(key)
    if val is None:
        return False
    return bool(val)


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def table_overall_accuracy(summary: dict | None, seed_acc: dict[str, list[float]]) -> str:
    rows: list[list[str]] = []
    for mode in MODES:
        label = MODE_LABELS[mode]
        mean = stdev = None
        seed_vals: list[float] = []
        if summary and isinstance(summary.get("modes"), dict):
            block = summary["modes"].get(mode) or {}
            if block.get("mean") is not None:
                mean = float(block["mean"])
            if block.get("stdev") is not None:
                stdev = float(block["stdev"])
            seed_vals = [float(x) for x in (block.get("accuracies") or [])]
            if not seed_vals:
                seed_vals = [
                    float(run["accuracy_pct"])
                    for run in (block.get("runs") or [])
                    if run.get("accuracy_pct") is not None
                ]
        if mean is None or stdev is None:
            vals = seed_acc.get(mode) or seed_vals
            mean, stdev = mean_stdev([float(v) for v in vals])
        if not seed_vals:
            seed_vals = seed_acc.get(mode) or []
        seed_str = ", ".join(f"{v:.2f}" for v in seed_vals) if seed_vals else "—"
        if mean != mean:  # NaN
            rows.append([label, "—", "—", "—"])
        else:
            rows.append(
                [
                    label,
                    f"{mean:.2f} ± {stdev:.2f}",
                    seed_str,
                    str(len(seed_vals) or "—"),
                ]
            )
    body = markdown_table(
        rows,
        ["Method", "Mean ± Stdev (%)", "Per-seed accuracies (%)", "Seeds"],
    )
    return (
        "## Table 1. Overall Execution Accuracy\n\n"
        "_Spider dev execution accuracy on scorable instances; "
        "mean ± standard deviation across seeds._\n\n"
        f"{body}\n"
    )


def table_band_breakdown(
    band_seed_acc: dict[str, dict[str, list[float]]],
    band_n: dict[str, dict[str, list[int]]],
) -> str:
    rows: list[list[str]] = []
    for mode in MODES:
        cells = [MODE_LABELS[mode]]
        for band in BANDS:
            vals = band_seed_acc[band].get(mode) or []
            ns = band_n[band].get(mode) or []
            if not vals:
                cells.append("—")
                continue
            mean = sum(vals) / len(vals)
            n_mean = sum(ns) / len(ns) if ns else 0
            cells.append(f"{mean:.2f}% (n≈{n_mean:.0f}/seed)")
        rows.append(cells)
    body = markdown_table(rows, ["Method", "Easy", "Medium", "Hard"])
    return (
        "## Table 2. Difficulty Band Breakdown\n\n"
        "_Band from `sql_skeleton(gold_sql).band`. "
        "Accuracy averaged over completed seeds; only scorable rows "
        "(`score is not None`)._\n\n"
        f"{body}\n"
    )


def table_error_distribution(status_totals: dict[str, dict[str, int]]) -> str:
    rows: list[list[str]] = []
    for mode in MODES:
        cells = [MODE_LABELS[mode]]
        for status in STATUS_COLS:
            cells.append(str(status_totals[mode].get(status, 0)))
        rows.append(cells)
    rows.append(
        ["**Total**"]
        + [
            str(sum(status_totals[mode].get(status, 0) for mode in MODES))
            for status in STATUS_COLS
        ]
    )
    body = markdown_table(rows, ["Method", *STATUS_COLS])
    return (
        "## Table 3. Diagnostic Error Distribution\n\n"
        "_Counts summed across all completed seeds "
        "(3 × 1034 rows per method when fully complete). "
        "`gold_error` omitted from this diagnostic table._\n\n"
        f"{body}\n"
    )


def table_retrieval_coverage(
    summary: dict | None,
    fs_stats: dict[str, dict[str, list[float]]],
) -> str:
    rows: list[list[str]] = []
    for mode in ("jaccard", "skeleton"):
        fs_nonzero_vals: list[float] = []
        fs_avg_vals: list[float] = []
        if summary and isinstance(summary.get("modes"), dict):
            block = summary["modes"].get(mode) or {}
            for run in block.get("runs") or []:
                if run.get("fs_nonzero") is not None:
                    fs_nonzero_vals.append(float(run["fs_nonzero"]))
                if run.get("fs_avg") is not None:
                    fs_avg_vals.append(float(run["fs_avg"]))
        if not fs_nonzero_vals:
            fs_nonzero_vals = fs_stats[mode].get("fs_nonzero") or []
        if not fs_avg_vals:
            fs_avg_vals = fs_stats[mode].get("fs_avg") or []
        if not fs_nonzero_vals and not fs_avg_vals:
            rows.append([MODE_LABELS[mode], "—", "—", "—"])
            continue
        nz_mean = sum(fs_nonzero_vals) / len(fs_nonzero_vals) if fs_nonzero_vals else 0.0
        avg_mean = sum(fs_avg_vals) / len(fs_avg_vals) if fs_avg_vals else 0.0
        rows.append(
            [
                MODE_LABELS[mode],
                f"{nz_mean:.1f}",
                f"{avg_mean:.4f}",
                str(len(fs_nonzero_vals) or len(fs_avg_vals)),
            ]
        )
    # No-Shot row for completeness (coverage should be ~0).
    nofew_nz: list[float] = []
    nofew_avg: list[float] = []
    if summary and isinstance(summary.get("modes"), dict):
        block = summary["modes"].get("nofew") or {}
        for run in block.get("runs") or []:
            if run.get("fs_nonzero") is not None:
                nofew_nz.append(float(run["fs_nonzero"]))
            if run.get("fs_avg") is not None:
                nofew_avg.append(float(run["fs_avg"]))
    if not nofew_nz:
        nofew_nz = list(fs_stats["nofew"].get("fs_nonzero") or [])
    if not nofew_avg:
        nofew_avg = list(fs_stats["nofew"].get("fs_avg") or [])
    if nofew_nz or nofew_avg:
        rows.insert(
            0,
            [
                MODE_LABELS["nofew"],
                f"{(sum(nofew_nz)/len(nofew_nz)) if nofew_nz else 0:.1f}",
                f"{(sum(nofew_avg)/len(nofew_avg)) if nofew_avg else 0:.4f}",
                str(len(nofew_nz) or len(nofew_avg)),
            ],
        )
    body = markdown_table(
        rows,
        [
            "Method",
            "fs_nonzero (avg over seeds)",
            "fs_avg (avg over seeds)",
            "Seeds",
        ],
    )
    return (
        "## Table 4. Retrieval Coverage (`fs_nonzero` & `fs_avg`)\n\n"
        "_`fs_nonzero`: queries with ≥1 retrieved few-shot. "
        "`fs_avg`: mean `fewshot_retrieved` over scorable queries. "
        "Values averaged across seeds._\n\n"
        f"{body}\n"
    )


def table_retry_stats(
    summary: dict | None,
    retry_totals: dict[str, dict[str, int]],
) -> str:
    rows: list[list[str]] = []
    for mode in MODES:
        attempted = retry_totals[mode]["attempted"]
        fixed = retry_totals[mode]["fixed"]
        # Prefer summary totals when present and jsonl empty (older files).
        if summary and isinstance(summary.get("modes"), dict):
            block = summary["modes"].get(mode) or {}
            if attempted == 0 and block.get("sql_retry_attempted_total") is not None:
                attempted = int(block.get("sql_retry_attempted_total") or 0)
                fixed = int(block.get("sql_retry_fixed_total") or 0)
            elif attempted == 0:
                # Fall back to summing per-run fields if totals missing.
                for run in block.get("runs") or []:
                    attempted += int(run.get("sql_retry_attempted") or 0)
                    fixed += int(run.get("sql_retry_fixed") or 0)
        rate = (100.0 * fixed / attempted) if attempted else 0.0
        rows.append(
            [
                MODE_LABELS[mode],
                str(attempted),
                str(fixed),
                f"{rate:.2f}%",
            ]
        )
    tot_a = sum(int(r[1]) for r in rows)
    tot_f = sum(int(r[2]) for r in rows)
    tot_rate = (100.0 * tot_f / tot_a) if tot_a else 0.0
    rows.append(
        ["**Overall**", str(tot_a), str(tot_f), f"**{tot_rate:.2f}%**"]
    )
    body = markdown_table(
        rows,
        [
            "Method",
            "sql_retry_attempted",
            "sql_retry_fixed",
            "Success rate (fixed/attempted)",
        ],
    )
    return (
        "## Table 5. Execution-Guided Retry Statistics\n\n"
        "_Missing `sql_retry_*` keys are treated as false/0. "
        "Totals summed across completed seeds._\n\n"
        f"{body}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PAPER_RUNS)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    _bootstrap_django()
    from data_service.agent.fewshot_skeleton import sql_skeleton

    summary = load_summary(args.summary)
    files = discover_completed(args.root)
    if not files and summary is None:
        raise SystemExit(
            f"no completed jsonl under {args.root} and no summary at {args.summary}"
        )

    seed_acc: dict[str, list[float]] = {mode: [] for mode in MODES}
    band_seed_acc: dict[str, dict[str, list[float]]] = {
        band: {mode: [] for mode in MODES} for band in BANDS
    }
    band_n: dict[str, dict[str, list[int]]] = {
        band: {mode: [] for mode in MODES} for band in BANDS
    }
    status_totals: dict[str, dict[str, int]] = {
        mode: {status: 0 for status in STATUS_COLS} for mode in MODES
    }
    fs_stats: dict[str, dict[str, list[float]]] = {
        mode: {"fs_nonzero": [], "fs_avg": []} for mode in MODES
    }
    retry_totals: dict[str, dict[str, int]] = {
        mode: {"attempted": 0, "fixed": 0} for mode in MODES
    }

    print(f"Summary: {args.summary} ({'found' if summary else 'missing'})")
    print(f"Completed jsonl runs: {len(files)}")
    for mode, seed, path in files:
        rows = load_jsonl(path)
        print(f"- {path.name} ({len(rows)} rows)")

        scored = [row for row in rows if row.get("score") is not None]
        if scored:
            acc = 100.0 * sum(int(row["score"]) for row in scored) / len(scored)
            seed_acc[mode].append(acc)

        # Retrieval coverage over scorable rows.
        if scored:
            retrieved = [int(row.get("fewshot_retrieved") or 0) for row in scored]
            fs_stats[mode]["fs_nonzero"].append(
                float(sum(1 for x in retrieved if x > 0))
            )
            fs_stats[mode]["fs_avg"].append(
                float(sum(retrieved) / len(retrieved))
            )

        # Status + retry (graceful missing keys).
        for row in rows:
            reason = str(row.get("reason") or "")
            if reason in status_totals[mode]:
                status_totals[mode][reason] += 1
            if safe_bool(row, "sql_retry_attempted"):
                retry_totals[mode]["attempted"] += 1
            if safe_bool(row, "sql_retry_fixed"):
                retry_totals[mode]["fixed"] += 1

        # Band accuracy from gold SQL skeleton.
        by_band: dict[str, list[int]] = {band: [] for band in BANDS}
        for row in scored:
            gold = row.get("gold_sql") or ""
            skeleton = sql_skeleton(str(gold))
            if skeleton is None:
                continue
            band = skeleton.band
            if band in by_band:
                by_band[band].append(int(row["score"]))
        for band in BANDS:
            scores = by_band[band]
            if not scores:
                continue
            band_seed_acc[band][mode].append(100.0 * sum(scores) / len(scores))
            band_n[band][mode].append(len(scores))

    sections = [
        "# Final Paper Tables (Spider Dev)\n",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from `{args.root.name}/`"
        + (f" and `{args.summary.name}`" if summary else "")
        + "._\n",
        table_overall_accuracy(summary, seed_acc),
        table_band_breakdown(band_seed_acc, band_n),
        table_error_distribution(status_totals),
        table_retrieval_coverage(summary, fs_stats),
        table_retry_stats(summary, retry_totals),
    ]
    document = "\n".join(sections).rstrip() + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding="utf-8")
    print()
    print(document)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
