from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_docs_cache import (  # noqa
    CACHE_SCHEMA_VERSION,
    build_model_docs_cache,
    get_cached_model_docs,
)
from src.models import MODELS  # noqa


def test_build_model_docs_cache_has_expected_top_level_shape():
    payload = build_model_docs_cache()

    assert payload["schema_version"] == CACHE_SCHEMA_VERSION
    assert isinstance(payload["generated_at_utc"], str)
    assert isinstance(payload["sklearn_version"], str)
    assert isinstance(payload["models"], dict)


def test_build_model_docs_cache_covers_all_models():
    payload = build_model_docs_cache()
    models_payload = payload["models"]

    assert sorted(models_payload.keys()) == sorted(MODELS.keys())


def test_get_cached_model_docs_returns_renderable_payload_for_all_models():
    payload = build_model_docs_cache()

    for model_key in sorted(MODELS.keys()):
        cached = get_cached_model_docs(model_key, payload=payload)
        assert cached is not None
        model_desc, param_desc = cached
        assert isinstance(model_desc, str)
        assert model_desc.strip()
        assert isinstance(param_desc, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in param_desc.items())
