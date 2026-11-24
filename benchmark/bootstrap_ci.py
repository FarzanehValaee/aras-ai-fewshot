#!/usr/bin/env python3
"""Paired bootstrap of execution accuracy for two JSONL dumps.

Resamples question indices with replacement. The same indices are applied
to both systems. Seeded and deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HARDNESS = ("easy", "medium", "hard", "extra")


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


def _key(row: dict) -> tuple[str, str]:
    return (str(row.get("db_id") or ""), str(row.get("question") or ""))


def _ex(row: dict) -> int | None:
    for field in ("ex", "score"):
        value = row.get(field)
        if value in (0, 1):
            return int(value)
    return None


def align(left_rows: list[dict], right_rows: list[dict]) -> list[tuple[dict, dict]]:
    right = {_key(row): row for row in right_rows}
    pairs: list[tuple[dict, dict]] = []
    missing: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in left_rows:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        other = right.get(key)
        if other is None or _ex(row) is None or _ex(other) is None:
            missing.append(key)
            continue
        pairs.append((row, other))
    extra = [key for key in right if key not in seen]
    if missing or extra or len(pairs) != len(seen):
        raise SystemExit(
            "question sets differ: "
            f"paired={len(pairs)} missing_or_unscored={len(missing)} "
            f"only_in_second={len(extra)}"
        )
    return pairs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def bootstrap(
    pairs: list[tuple[dict, dict]],
    *,
    resamples: int,
    seed: int,
) -> dict:
    left = [int(_ex(a) or 0) for a, _b in pairs]
    right = [int(_ex(b) or 0) for _a, b in pairs]
    rng = random.Random(seed)
    count = len(pairs)
    deltas: list[float] = []
    for _ in range(resamples):
        indexes = [rng.randrange(count) for _ in range(count)]
        mean_left = _mean([left[i] for i in indexes])
        mean_right = _mean([right[i] for i in indexes])
        deltas.append(mean_right - mean_left)
    ordered = sorted(deltas)
    observed_left = _mean(left)
    observed_right = _mean(right)
    return {
        "n": count,
        "mean_a": observed_left,
        "mean_b": observed_right,
        "mean_delta": observed_right - observed_left,
        "ci_low": _percentile(ordered, 0.025),
        "ci_high": _percentile(ordered, 0.975),
        "p_delta_le_0": sum(1 for value in deltas if value <= 0) / len(deltas),
        "resamples": resamples,
        "seed": seed,
    }


def _fmt(stats: dict) -> str:
    return (
        f"n={stats['n']}  "
        f"EX_a={100.0 * stats['mean_a']:.2f}  "
        f"EX_b={100.0 * stats['mean_b']:.2f}  "
        f"delta={100.0 * stats['mean_delta']:+.2f}  "
        f"95% CI [{100.0 * stats['ci_low']:+.2f}, {100.0 * stats['ci_high']:+.2f}]  "
        f"p(delta<=0)={stats['p_delta_le_0']:.4f}"
    )


def report(pairs: list[tuple[dict, dict]], *, resamples: int, seed: int) -> str:
    lines = [
        f"resamples={resamples} seed={seed}",
        "delta = EX(second) - EX(first)",
        "overall  " + _fmt(bootstrap(pairs, resamples=resamples, seed=seed)),
        "per hardness",
    ]
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        label = str(pair[0].get("official_hardness") or "").strip().lower()
        if label not in HARDNESS:
            label = "unknown"
        grouped[label].append(pair)
    for label in (*HARDNESS, "unknown"):
        bucket = grouped.get(label) or []
        if not bucket:
            continue
        lines.append(f"  {label:<8} " + _fmt(bootstrap(bucket, resamples=resamples, seed=seed)))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    from repro_config import load as load_repro_config

    cfg = load_repro_config()["bootstrap"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="JSONL for system A")
    parser.add_argument("second", type=Path, help="JSONL for system B, same questions")
    parser.add_argument("--resamples", type=int, default=int(cfg["resamples"]))
    parser.add_argument("--seed", type=int, default=int(cfg["seed"]))
    return parser.parse_args()


def main() -> None:
    from repro_config import seed_everything

    args = parse_args()
    seed_everything(args.seed)
    if args.resamples < 1:
        raise SystemExit("--resamples must be positive")
    pairs = align(load_rows(args.first), load_rows(args.second))
    print(f"first:  {args.first}")
    print(f"second: {args.second}")
    print(report(pairs, resamples=args.resamples, seed=args.seed))


if __name__ == "__main__":
    main()
