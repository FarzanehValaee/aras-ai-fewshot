"""Chart spec builder adapted from Numbers chart_builder (no HTML templates)."""

from __future__ import annotations

import re
from typing import Any

_MAX_CHARTS = 3
_MAX_POINTS = 20
_CHART_HINT = re.compile(r"chart|graph|plot", re.IGNORECASE)


def wants_chart(question: str, *, include_chart: bool = True) -> bool:
    if not include_chart:
        return False
    q = str(question or "")
    if _CHART_HINT.search(q):
        return True
    # Auto-chart when query looks aggregating/ranking
    return True


def _label_with_unit(label: str, unit: str) -> str:
    text = str(label or "").strip()
    suffix = str(unit or "").strip()
    if not suffix or suffix in text:
        return text or label
    return f"{text} ({suffix})"


def build_chart_spec_from_table(
    *,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
    sql: str = "",
    units: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build chart_spec from tabular result when suitable (≥2 rows, label+value)."""
    if len(rows) < 2 or len(columns) < 2:
        return None

    labels: list[str] = []
    values: list[float] = []
    for row in rows[:_MAX_POINTS]:
        if len(row) < 2:
            continue
        label = str(row[0])
        try:
            value = float(row[1])
        except (TypeError, ValueError):
            # try last numeric column
            value = None
            for cell in reversed(row[1:]):
                try:
                    value = float(cell)
                    break
                except (TypeError, ValueError):
                    continue
            if value is None:
                continue
        labels.append(label)
        values.append(value)

    if len(labels) < 2:
        return None

    sql_l = (sql or "").lower()
    if "group by" in sql_l and any(t in sql_l for t in ("date", "day", "month", "year")):
        chart_type, category = "line", "time_series"
        title = "Trend over time"
    elif len(labels) <= 6:
        chart_type, category = "bar", "grouped"
        title = (question or "Analysis result")[:80]
    else:
        chart_type, category = "horizontal_bar", "grouped"
        title = (question or "Analysis result")[:80]

    unit_of = list(units or [])
    value_label = _label_with_unit(
        columns[1] if len(columns) > 1 else "value",
        unit_of[1] if len(unit_of) > 1 else "",
    )
    name_label = _label_with_unit(
        columns[0] if columns else "label",
        unit_of[0] if unit_of else "",
    )
    chart = {
        "id": category,
        "chart_type": chart_type,
        "title": title,
        "category": category,
        "x_label": value_label,
        "y_label": name_label,
        "labels": labels,
        "values": values,
        "value_format": "number",
        "download_filename": f"data_{category}.csv",
        "columns": [
            {"key": "label", "label_fa": name_label},
            {"key": "value", "label_fa": value_label},
        ],
    }
    return {"version": 1, "charts": [chart][:_MAX_CHARTS]}
