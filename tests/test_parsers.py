from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MODELS  # noqa
from src.parsers import (  # noqa
    BASE_URL,
    SKLEARN_REF_MAP,
    parse_model_desc,
    parse_param_desc,
    replace_sklearn_refs,
)

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
UNSAFE_SCHEMES = ("javascript:", "data:", "file:", "vbscript:")


def extract_md_urls(text: str) -> list[str]:
    return MD_LINK_RE.findall(text or "")


def assert_urls_safe(urls: list[str]) -> None:
    for url in urls:
        normalized = url.strip()
        assert not normalized.lower().startswith(UNSAFE_SCHEMES), (
            f"Unsafe URL scheme: {normalized}"
        )
        assert normalized.startswith("https://scikit-learn.org/"), (
            f"Unexpected domain: {normalized}"
        )
        assert normalized.startswith("https://scikit-learn.org/stable/"), (
            f"Not stable docs: {normalized}"
        )
        assert normalized.startswith(BASE_URL), (
            f"Not in allowed modules base: {normalized}"
        )


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parse_model_desc_returns_non_empty_markdown(model_key):
    model = MODELS[model_key].factory()

    desc = parse_model_desc(model)

    assert isinstance(desc, str)
    assert desc.strip()
    assert "```python" in desc
    assert repr(model) in desc


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parse_param_desc_returns_dict_for_model(model_key):
    model = MODELS[model_key].factory()

    params_desc = parse_param_desc(model)

    assert isinstance(params_desc, dict)
    assert set(params_desc).issubset(model.get_params().keys())


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parse_param_desc_covers_documented_params_for_model(model_key):
    model = MODELS[model_key].factory()

    params_desc = parse_param_desc(model)

    for key, value in params_desc.items():
        assert key in model.get_params()
        assert isinstance(value, str)
        assert value.strip()


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parser_outputs_do_not_leave_rst_ref_markup_when_model_uses_known_refs(model_key):
    model = MODELS[model_key].factory()

    model_desc = parse_model_desc(model)
    params_desc = parse_param_desc(model)
    combined_output = "\n".join([model_desc, *params_desc.values()])
    raw_doc = model.__doc__ or ""

    for slug, relative_url in SKLEARN_REF_MAP.items():
        if f"<{slug}>`" not in raw_doc:
            continue
        assert f"<{slug}>`" not in combined_output
        if relative_url in combined_output:
            assert f"{BASE_URL}{relative_url}" in combined_output


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parse_model_desc_smoke_and_links_are_safe(model_key):
    model = MODELS[model_key].factory()

    out = parse_model_desc(model)

    assert ":ref:`" not in out
    assert_urls_safe(extract_md_urls(out))


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parse_param_desc_smoke_and_links_are_safe(model_key):
    model = MODELS[model_key].factory()

    out = parse_param_desc(model)

    assert isinstance(out, dict)
    combined = "\n".join(out.values())
    assert ":ref:`" not in combined
    assert_urls_safe(extract_md_urls(combined))


def test_replace_sklearn_refs_replaces_known_slug():
    text = "See :ref:`User Guide <adaboost>` for more details."

    replaced = replace_sklearn_refs(text)

    assert (
        replaced
        == "See [User Guide](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) for more details."  # noqa
    )


def test_replace_sklearn_refs_falls_back_to_label_for_unknown_slug():
    text = "See :ref:`User Guide <unknown-slug>` for more details."

    replaced = replace_sklearn_refs(text)

    assert replaced == "See User Guide for more details."


def test_replace_sklearn_refs_replaces_simple_references_with_plain_text():
    text = "See :ref:`svm_kernels` and :ref:`tree_mathematical_formulation`."

    replaced = replace_sklearn_refs(text)

    assert replaced == "See svm_kernels and tree_mathematical_formulation."


def test_replace_sklearn_refs_handles_multiple_references():
    text = (
        "Read :ref:`User Guide <adaboost>` and "
        ":ref:`Linear Models <logistic-regression>`."
    )

    replaced = replace_sklearn_refs(text)

    assert (
        replaced
        == "Read [User Guide](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) "  # noqa
        "and [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)."
    )
