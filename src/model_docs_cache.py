from __future__ import annotations

from datetime import datetime, UTC
import argparse
import json
from pathlib import Path
from typing import Any

import sklearn

from src.models import MODELS
from src.parsers import parse_model_desc, parse_param_desc

CACHE_SCHEMA_VERSION = 1
CACHE_PATH = Path(__file__).with_name("model_docs_cache.json")


def build_model_docs_cache() -> dict[str, Any]:
    models_payload: dict[str, dict[str, Any]] = {}

    for model_key in sorted(MODELS.keys()):
        model = MODELS[model_key].factory()
        models_payload[model_key] = {
            "model_desc": parse_model_desc(model),
            "param_desc": parse_param_desc(model),
        }

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "models": models_payload,
    }


def write_model_docs_cache(path: Path = CACHE_PATH) -> Path:
    payload = build_model_docs_cache()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_model_docs_cache(path: Path = CACHE_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    models = payload.get("models")
    if not isinstance(models, dict):
        return None
    return payload


def get_cached_model_docs(
    model_key: str,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]] | None:
    cache = payload if payload is not None else load_model_docs_cache()
    if cache is None:
        return None
    models = cache.get("models")
    if not isinstance(models, dict):
        return None
    model_payload = models.get(model_key)
    if not isinstance(model_payload, dict):
        return None

    model_desc = model_payload.get("model_desc")
    param_desc = model_payload.get("param_desc")
    if not isinstance(model_desc, str) or not isinstance(param_desc, dict):
        return None

    normalized_param_desc: dict[str, str] = {}
    for key, value in param_desc.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
        normalized_param_desc[key] = value

    return model_desc, normalized_param_desc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate parser markdown cache for all registered models."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CACHE_PATH,
        help="Output JSON path (default: src/model_docs_cache.json).",
    )
    args = parser.parse_args()

    out_path = write_model_docs_cache(args.output)
    print(f"Wrote cache: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
