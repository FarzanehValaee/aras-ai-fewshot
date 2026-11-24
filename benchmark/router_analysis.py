#!/usr/bin/env python3
"""Quantify the router from per-question JSONL dumps.

Read-only. Predicted class is the router's easy/medium/hard label.
Official hardness is Spider's easy/medium/hard/extra. "extra" is collapsed
into "hard" only for the 3x3 view and for the correct-vs-wrong EX split.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROUTER = ("easy", "medium", "hard")
OFFICIAL = ("easy", "medium", "hard", "extra")
COLLAPSED = ("easy", "medium", "hard")
ROOT = Path("/var/www/aras-ai-data")
RESULTS = ROOT / "results"


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if isinstance(row, dict):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def discover(paths: list[Path]) -> list[Path]:
    if paths:
        found: list[Path] = []
        for path in paths:
            if path.is_dir():
                found.extend(sorted(path.glob("run_*.jsonl")))
            elif path.is_file():
                found.append(path)
            else:
                raise SystemExit(f"missing {path}")
        if not found:
            raise SystemExit("no JSONL files in the given paths")
        return found
    found = sorted(RESULTS.glob("run_*.jsonl"))
    if not found:
        raise SystemExit(f"no run_*.jsonl under {RESULTS}")
    return found


def _label(value: object, allowed: tuple[str, ...]) -> str | None:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return None


def collapse(label: str | None) -> str | None:
    if label == "extra":
        return "hard"
    if label in COLLAPSED:
        return label
    return None


def scanned(row: dict) -> tuple[float, float] | None:
    payload = row.get("num_candidates_scanned")
    if not isinstance(payload, dict):
        return None
    try:
        filtered = float(payload["with_class_filter"])
        global_n = float(payload["without_class_filter"])
    except (KeyError, TypeError, ValueError):
        return None
    if filtered < 0 or global_n < 0:
        return None
    return filtered, global_n


def ex_of(row: dict) -> int | None:
    value = row.get("ex")
    if value in (0, 1):
        return int(value)
    if value in ("0", "1"):
        return int(value)
    return None


def pairs(rows: list[dict]) -> tuple[list[tuple[str, str]], int]:
    """Mapped (predicted, official) pairs, plus rows skipped as unmapped."""
    kept: list[tuple[str, str]] = []
    skipped = 0
    for row in rows:
        predicted = _label(row.get("predicted_class"), ROUTER)
        official = _label(row.get("official_hardness"), OFFICIAL)
        if predicted is None or official is None:
            skipped += 1
            continue
        kept.append((predicted, official))
    return kept, skipped


def counts_matrix(
    paired: list[tuple[str, str]],
    row_labels: tuple[str, ...],
    col_labels: tuple[str, ...],
    *,
    collapse_extra: bool,
) -> list[list[int]]:
    index_r = {label: i for i, label in enumerate(row_labels)}
    index_c = {label: i for i, label in enumerate(col_labels)}
    grid = [[0 for _ in col_labels] for _ in row_labels]
    for predicted, official in paired:
        gold = collapse(official) if collapse_extra else official
        if predicted not in index_r or gold not in index_c:
            continue
        grid[index_r[predicted]][index_c[gold]] += 1
    return grid


def row_normalize(grid: list[list[int]]) -> list[list[float | None]]:
    out: list[list[float | None]] = []
    for row in grid:
        total = sum(row)
        if total == 0:
            out.append([None for _ in row])
        else:
            out.append([value / total for value in row])
    return out


def prf(
    grid: list[list[int]],
    row_labels: tuple[str, ...],
    col_labels: tuple[str, ...],
) -> list[dict]:
    """Precision uses predicted rows. Recall uses gold columns.

    Extra has no predicted row, so its precision is undefined.
    """
    col_index = {label: i for i, label in enumerate(col_labels)}
    row_index = {label: i for i, label in enumerate(row_labels)}
    reports: list[dict] = []
    labels = list(dict.fromkeys((*row_labels, *col_labels)))
    for label in labels:
        predicted_n = sum(grid[row_index[label]]) if label in row_index else 0
        gold_n = (
            sum(grid[r][col_index[label]] for r in range(len(grid)))
            if label in col_index
            else 0
        )
        hit = 0
        if label in row_index and label in col_index:
            hit = grid[row_index[label]][col_index[label]]
        precision = (hit / predicted_n) if predicted_n else None
        recall = (hit / gold_n) if gold_n else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        reports.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_n": predicted_n,
                "gold_n": gold_n,
            }
        )
    return reports


def accuracy(grid: list[list[int]], row_labels: tuple[str, ...], col_labels: tuple[str, ...]) -> tuple[int, int]:
    correct = 0
    total = 0
    for i, label in enumerate(row_labels):
        total += sum(grid[i])
        if label in col_labels:
            correct += grid[i][col_labels.index(label)]
    return correct, total


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * value:.2f}"


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [head, rule]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def latex_table(headers: list[str], body: list[list[str]], *, align: str) -> str:
    spec = align
    lines = [
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def matrix_tables(
    title: str,
    grid: list[list[int]],
    row_labels: tuple[str, ...],
    col_labels: tuple[str, ...],
) -> tuple[str, str]:
    normalized = row_normalize(grid)
    headers = ["predicted \\ gold", *col_labels, "n"]
    body: list[list[str]] = []
    for i, label in enumerate(row_labels):
        body.append(
            [label, *[_cell(value) for value in normalized[i]], str(sum(grid[i]))]
        )
    md = f"### {title}\n\n" + markdown_table(
        ["predicted \\ gold", *col_labels, "n"], body
    )
    tex_headers = ["predicted $\\backslash$ gold", *col_labels, "$n$"]
    tex = (
        f"% {title}\n"
        + latex_table(tex_headers, body, align="l" + "r" * (len(col_labels) + 1))
    )
    return md, tex


def metric_tables(title: str, grid: list[list[int]], rows: tuple[str, ...], cols: tuple[str, ...]) -> tuple[str, str]:
    correct, total = accuracy(grid, rows, cols)
    acc = (correct / total) if total else None
    wrong = (1.0 - acc) if acc is not None else None
    reports = prf(grid, rows, cols)
    headers = ["class", "precision", "recall", "F1", "predicted n", "gold n"]
    body = [
        [
            item["class"],
            fmt_rate(item["precision"]),
            fmt_rate(item["recall"]),
            fmt_rate(item["f1"]),
            str(item["predicted_n"]),
            str(item["gold_n"]),
        ]
        for item in reports
    ]
    summary = [
        ["accuracy", fmt_rate(acc), f"{correct}/{total}"],
        ["wrong class", fmt_rate(wrong), f"{total - correct}/{total}" if total else "—"],
    ]
    md = (
        f"### {title}\n\n"
        + markdown_table(["metric", "percent", "count"], summary)
        + "\n\n"
        + markdown_table(headers, body)
    )
    tex = (
        f"% {title}\n"
        + latex_table(["metric", "percent", "count"], summary, align="lrr")
        + "\n\n"
        + latex_table(
            ["class", "precision", "recall", "F1", "predicted $n$", "gold $n$"],
            body,
            align="lrrrrr",
        )
    )
    return md, tex


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def efficiency_tables(rows: list[dict]) -> tuple[str, str]:
    filtered: list[float] = []
    global_n: list[float] = []
    for row in rows:
        pair = scanned(row)
        if pair is None:
            continue
        filtered.append(pair[0])
        global_n.append(pair[1])
    mean_f, mean_g = _mean(filtered), _mean(global_n)
    med_f, med_g = _median(filtered), _median(global_n)
    factor = (mean_g / mean_f) if mean_f and mean_g is not None else None
    body = [
        ["with class filter", fmt_num(mean_f), fmt_num(med_f), str(len(filtered))],
        ["global (no class filter)", fmt_num(mean_g), fmt_num(med_g), str(len(global_n))],
        ["reduction factor", fmt_num(factor), "—", "mean global / mean filtered"],
    ]
    headers = ["comparison", "mean", "median", "n"]
    md = "### Candidate comparisons\n\n" + markdown_table(headers, body)
    tex = "% Candidate comparisons\n" + latex_table(headers, body, align="lrrl")
    return md, tex


def routing_tables(rows: list[dict]) -> tuple[str, str]:
    buckets: dict[str, list[int]] = {"correct": [], "wrong": []}
    for row in rows:
        predicted = collapse(_label(row.get("predicted_class"), ROUTER))
        official = collapse(_label(row.get("official_hardness"), OFFICIAL))
        score = ex_of(row)
        if predicted is None or official is None or score is None:
            continue
        key = "correct" if predicted == official else "wrong"
        buckets[key].append(score)
    body = []
    for key in ("correct", "wrong"):
        values = buckets[key]
        hits = sum(values)
        rate = (hits / len(values)) if values else None
        body.append([key, str(len(values)), str(hits), fmt_rate(rate)])
    headers = ["router", "questions", "correct EX", "EX"]
    md = (
        "### Execution accuracy by routing\n\n"
        + markdown_table(headers, body)
        + "\n\nRouting is correct when the predicted class matches official "
        "hardness after extra is collapsed into hard.\n"
    )
    tex = (
        "% Execution accuracy by routing\n"
        + latex_table(["router", "questions", "correct EX", "EX"], body, align="lrrr")
    )
    return md, tex


def analyze(rows: list[dict]) -> str:
    paired, skipped = pairs(rows)
    sections: list[str] = []
    sections.append(f"Questions in dump: {len(rows)}")
    sections.append(f"Mapped router rows: {len(paired)}")
    sections.append(
        "Unmapped (missing class, or predicted_class outside easy/medium/hard, "
        f"including medium+hard): {skipped}"
    )
    md_parts = ["## Markdown", ""]
    tex_parts = ["## LaTeX", ""]
    views = (
        ("3x3, extra collapsed into hard", COLLAPSED, COLLAPSED, True),
        ("3x4, official hardness", ROUTER, OFFICIAL, False),
    )
    for title, row_labels, col_labels, collapse_extra in views:
        grid = counts_matrix(
            paired, row_labels, col_labels, collapse_extra=collapse_extra
        )
        md_matrix, tex_matrix = matrix_tables(
            f"Row-normalized confusion, {title}", grid, row_labels, col_labels
        )
        md_metrics, tex_metrics = metric_tables(
            f"Router scores, {title}", grid, row_labels, col_labels
        )
        md_parts.extend([md_matrix, "", md_metrics, ""])
        tex_parts.extend([tex_matrix, "", tex_metrics, ""])
    md_eff, tex_eff = efficiency_tables(rows)
    md_route, tex_route = routing_tables(rows)
    md_parts.extend([md_eff, "", md_route])
    tex_parts.extend([tex_eff, "", tex_route])
    return "\n".join(sections) + "\n\n" + "\n".join(md_parts) + "\n\n" + "\n".join(tex_parts) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="*",
        type=Path,
        help="Per-question JSONL files or a directory of run_*.jsonl. "
        "Defaults to results/run_*.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover(list(args.jsonl))
    print("files:")
    for path in paths:
        print(f"  {path}")
    print()
    print(analyze(load_rows(paths)))


if __name__ == "__main__":
    main()
