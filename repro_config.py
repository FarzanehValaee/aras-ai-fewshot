"""Load configs/repro.yaml, apply env overrides, seed RNGs, hash the result.

API keys are never read from this file. Callers use ROUTER_API_KEY or
OPENAI_API_KEY from the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "configs" / "repro.yaml"

_CACHE: dict[str, Any] | None = None


def _need(node: dict, key: str) -> Any:
    if key not in node:
        raise SystemExit(f"configs/repro.yaml is missing {key}")
    return node[key]


def _env_int(name: str, current: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(current)
    return int(raw)


def _env_float(name: str, current: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(current)
    return float(raw)


def _env_str(name: str, current: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return str(current)
    return raw


def _env_bool(name: str, current: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(current)
    return raw.strip().lower() in {"1", "true", "yes"}


def load(path: Path | None = None, *, refresh: bool = False) -> dict[str, Any]:
    """Return the resolved config. The same object is reused until refresh."""
    global _CACHE
    if _CACHE is not None and not refresh and path is None:
        return _CACHE
    source = Path(path or os.environ.get("REPRO_CONFIG") or DEFAULT_PATH)
    if not source.is_file():
        raise SystemExit(f"repro config not found: {source}")
    import yaml

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"repro config must be a mapping: {source}")
    resolved = _resolve(raw)
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    resolved["config_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    resolved["config_path"] = str(source)
    _CACHE = resolved
    return resolved


def _resolve(raw: dict[str, Any]) -> dict[str, Any]:
    temperature = dict(_need(raw, "temperature"))
    pool = dict(_need(raw, "pool"))
    similarity = dict(_need(raw, "similarity"))
    icrl = dict(_need(raw, "icrl"))
    prices = dict(_need(raw, "prices"))
    paths = dict(_need(raw, "paths"))
    bootstrap = dict(_need(raw, "bootstrap"))
    pool_build = _resolve_pool_build(raw)
    weights = dict(_need(similarity, "feature_weights"))
    scores = dict(_need(icrl, "keyword_scores"))

    temperature["eval"] = _env_float("SPIDER_EVAL_TEMPERATURE", temperature["eval"])
    temperature["sql"] = float(temperature["sql"])
    temperature["grounding"] = float(temperature["grounding"])
    temperature["planner"] = float(temperature["planner"])
    temperature["icrl"] = float(temperature["icrl"])
    temperature["translation"] = float(temperature["translation"])

    paths["items"] = _env_str("SPIDER_EVAL_ITEMS", paths["items"])
    paths["out"] = _env_str("SPIDER_EVAL_OUT", paths["out"])
    paths["results_dir"] = _env_str("SPIDER_EVAL_RESULTS", paths["results_dir"])
    paths["schema_cache"] = _env_str("SPIDER_SCHEMA_CACHE", paths["schema_cache"])
    paths["nltk_data"] = _env_str("NLTK_DATA", paths["nltk_data"])
    paths["spider_db_dir"] = _env_str("SPIDER_DB_DIR", paths.get("spider_db_dir") or "")
    paths["train"] = _env_str("SPIDER_TRAIN", paths.get("train") or "")
    paths["dev"] = _env_str("SPIDER_DEV", paths.get("dev") or paths["items"])
    paths["tables"] = _env_str("SPIDER_TABLES", paths.get("tables") or "")
    paths["pool"] = _env_str("SPIDER_POOL", paths.get("pool") or "")

    prices["usd_per_mtok_in"] = _env_float(
        "SPIDER_EVAL_USD_PER_MTOK_IN", prices["usd_per_mtok_in"]
    )
    prices["usd_per_mtok_out"] = _env_float(
        "SPIDER_EVAL_USD_PER_MTOK_OUT", prices["usd_per_mtok_out"]
    )

    return {
        "model": _env_str("SPIDER_EVAL_MODEL", _need(raw, "model")),
        "chat_model": _env_str("SPIDER_CHAT_MODEL", _need(raw, "chat_model")),
        "temperature": temperature,
        "seed": _env_int("SPIDER_EVAL_SEED", _need(raw, "seed")),
        "k": _env_int("FEWSHOT_RAG_TOP_K", _need(raw, "k")),
        "k_cap": int(_need(raw, "k_cap")),
        "mixed_band_cap": int(_need(raw, "mixed_band_cap")),
        "workers": _env_int("SPIDER_EVAL_WORKERS", _need(raw, "workers")),
        "limit": _env_int("SPIDER_EVAL_LIMIT", _need(raw, "limit")),
        "use_evidence": _env_bool("SPIDER_EVAL_USE_EVIDENCE", bool(raw.get("use_evidence", True))),
        "disable_fewshots": _env_bool(
            "SPIDER_EVAL_DISABLE_FEWSHOTS", bool(raw.get("disable_fewshots", False))
        ),
        "selection_strategy": _env_str(
            "SPIDER_EVAL_SELECTION_STRATEGY",
            str(raw.get("selection_strategy") or "full"),
        ),
        "fewshot_mode": _env_str(
            "SPIDER_EVAL_FEWSHOT_MODE",
            _env_str("FEWSHOT_RETRIEVAL_MODE", str(raw.get("fewshot_mode") or "skeleton")),
        ),
        "pool": {key: int(pool[key]) for key in (
            "base",
            "per_extra_table",
            "per_join_pair",
            "min_total",
            "max_total",
            "reserved_per_band",
            "no_join_easy",
            "no_join_medium",
            "no_join_hard",
        )},
        "similarity": {
            "function": str(similarity["function"]),
            "thresholds": dict(similarity.get("thresholds") or {}),
            "feature_weights": {key: int(weights[key]) for key in (
                "join_count",
                "group_by",
                "having",
                "where",
                "cte",
                "subquery",
                "agg_overlap",
                "table_overlap",
            )},
        },
        "icrl": {
            "iterations": _env_int("FEWSHOT_ICRL_ITERATIONS", icrl["iterations"]),
            "max_traversals": _env_int("FEWSHOT_ICRL_MAX_TRAVERSALS", icrl["max_traversals"]),
            "max_attempts": _env_int("FEWSHOT_ICRL_MAX_ATTEMPTS", icrl["max_attempts"]),
            "min_examples": _env_int("FEWSHOT_ICRL_MIN_EXAMPLES", icrl["min_examples"]),
            "on_upload": _env_bool("FEWSHOT_ICRL_ON_UPLOAD", bool(icrl["on_upload"])),
            "row_limit": _env_int("SQL_ROW_LIMIT", icrl["row_limit"]),
            "sql_timeout_seconds": _env_int("SQL_TIMEOUT_SECONDS", icrl["sql_timeout_seconds"]),
            "reward_split": str(icrl["reward_split"]),
            "keyword_scores": {str(key): float(value) for key, value in scores.items()},
            "temperature": float(temperature["icrl"]),
        },
        "bootstrap": {
            "resamples": int(bootstrap["resamples"]),
            "seed": int(bootstrap["seed"]),
        },
        "prices": {
            "usd_per_mtok_in": float(prices["usd_per_mtok_in"]),
            "usd_per_mtok_out": float(prices["usd_per_mtok_out"]),
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "pool_build": pool_build,
    }


def _resolve_pool_build(raw: dict[str, Any]) -> dict[str, Any]:
    """Train-pool builder. Env overrides are part of the hashed config."""
    node = dict(_need(raw, "pool_build"))
    paths = dict(_need(node, "paths"))
    quotas = dict(_need(node, "quotas"))
    excluded = [str(name) for name in (node.get("excluded_databases") or [])]
    override = os.environ.get("POOL_BUILD_EXCLUDED")
    if override:
        excluded = [part.strip() for part in override.split(",") if part.strip()]
    return {
        "paths": {
            "train_json": _env_str("POOL_BUILD_TRAIN", paths["train_json"]),
            "tables_json": _env_str("POOL_BUILD_TABLES", paths["tables_json"]),
            "sqlite_dir": _env_str("POOL_BUILD_SQLITE", paths["sqlite_dir"]),
            "dev_json": _env_str("POOL_BUILD_DEV", paths.get("dev_json") or ""),
            "output": _env_str("POOL_BUILD_OUTPUT", paths["output"]),
        },
        "excluded_databases": excluded,
        "quotas": {band: int(quotas[band]) for band in ("easy", "medium", "hard")},
        "shape_cap": _env_int("POOL_BUILD_SHAPE_CAP", node["shape_cap"]),
        "shape_cap_fallback": _env_int(
            "POOL_BUILD_SHAPE_FALLBACK", node["shape_cap_fallback"]
        ),
        "per_database_cap": _env_int("POOL_BUILD_DB_CAP", node["per_database_cap"]),
        "seed": _env_int("POOL_BUILD_SEED", node["seed"]),
        "execution_timeout_seconds": _env_int(
            "POOL_BUILD_TIMEOUT", node["execution_timeout_seconds"]
        ),
        "generate_sql": _env_bool("POOL_BUILD_GENERATE_SQL", bool(node["generate_sql"])),
    }


def overlay(**updates: Any) -> dict[str, Any]:
    """Replace resolved fields and recompute the hash. Skips None values."""
    cfg = load()
    path = cfg.get("config_path")
    for key, value in updates.items():
        if value is not None:
            cfg[key] = value
    body = {key: value for key, value in cfg.items() if key not in {"config_hash", "config_path"}}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    cfg["config_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    cfg["config_path"] = path
    return cfg


def public_config() -> dict[str, Any]:
    """Config written into result rows. No secrets."""
    cfg = load()
    body = {key: value for key, value in cfg.items() if key != "config_hash"}
    return {"config": body, "config_hash": cfg["config_hash"]}


def seed_everything(seed: int | None = None) -> int:
    """Seed random, and numpy/torch when those packages are installed."""
    value = int(load()["seed"] if seed is None else seed)
    random.seed(value)
    try:
        import numpy as np

        np.random.seed(value)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except ImportError:
        pass
    return value
