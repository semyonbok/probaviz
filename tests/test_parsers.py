from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MODELS  # noqa
from src.parsers import (  # noqa
    API_BASE_URL,
    BASE_URL,
    GENERATED_BASE_URL,
    GLOSSARY_URL,
    FUNC_PATTERN,
    MOD_PATTERN,
    METH_PATTERN,
    OBJ_PATTERN,
    CLASS_PATTERN,
    TERM_PATTERN,
    SKLEARN_REF_MAP,
    parse_model_desc,
    parse_param_desc,
    replace_external_links,
    replace_sklearn_refs,
    rst_roles_to_markdown,
)

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
UNSAFE_SCHEMES = ("javascript:", "data:", "file:", "vbscript:")
ROLES = ["ref", "term", "class", "meth", "obj", "mod", "func"]
ROLE_PATTERNS = {
    "term": TERM_PATTERN,
    "class": CLASS_PATTERN,
    "meth": METH_PATTERN,
    "obj": OBJ_PATTERN,
    "mod": MOD_PATTERN,
    "func": FUNC_PATTERN,
}


def extract_md_urls(text: str) -> list[str]:
    return MD_LINK_RE.findall(text or "")


def assert_urls_safe(urls: list[str]) -> None:
    for url in urls:
        normalized = url.strip()
        assert not normalized.lower().startswith(UNSAFE_SCHEMES), (
            f"Unsafe URL scheme: {normalized}"
        )
        assert (
            normalized.startswith("https://") or normalized.startswith("http://")
        ), f"Unexpected URL scheme: {normalized}"
        if normalized.startswith("https://scikit-learn.org/"):
            assert normalized.startswith("https://scikit-learn.org/stable/"), (
                f"Not stable docs: {normalized}"
            )
            assert (
                normalized.startswith(BASE_URL)
                or normalized.startswith(GENERATED_BASE_URL)
                or normalized.startswith(GLOSSARY_URL)
                or normalized.startswith(API_BASE_URL)
            ), (
                f"Not in allowed sklearn docs bases: {normalized}"
            )


def assert_no_rst_role_markup(text: str) -> None:
    for role in ROLES:
        assert f":{role}:`" not in text


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

    assert_no_rst_role_markup(out)
    assert_urls_safe(extract_md_urls(out))


@pytest.mark.parametrize("model_key", sorted(MODELS.keys()))
def test_parse_param_desc_smoke_and_links_are_safe(model_key):
    model = MODELS[model_key].factory()

    out = parse_param_desc(model)

    assert isinstance(out, dict)
    combined = "\n".join(out.values())
    assert_no_rst_role_markup(combined)
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


def test_replace_external_links_converts_rst_style_external_link():
    text = (
        "`scipy.spatial.distance "
        "<https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_"
    )
    replaced = replace_external_links(text)
    assert (
        replaced
        == "[scipy.spatial.distance](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)"
    )


@pytest.mark.parametrize("role", ROLES)
def test_rst_roles_to_markdown_removes_role_markup(role):
    examples = {
        "ref": "See :ref:`User Guide <adaboost>`.",
        "term": "See :term:`Glossary <n_jobs>`.",
        "class": "Use :class:`~sklearn.svm.SVC`.",
        "meth": "Call :meth:`fit` first.",
        "obj": "Configure :obj:`joblib.parallel_backend`.",
        "mod": "Import :mod:`sklearn.preprocessing`.",
        "func": "Compute :func:`~sklearn.metrics.accuracy_score`.",
    }
    out = rst_roles_to_markdown(examples[role])
    assert f":{role}:`" not in out
    if role in {"ref", "term", "class", "mod", "func"}:
        assert "https://scikit-learn.org/stable/" in out
    if role in {"meth", "obj"}:
        assert out.endswith(".")


def test_rst_roles_to_markdown_applies_all_roles_in_priority_order():
    text = (
        "A :ref:`Guide <adaboost>` and :term:`Glossary <n_jobs>` use "
        ":class:`~sklearn.svm.SVC`, :meth:`fit`, :obj:`joblib.parallel_backend`, "
        ":mod:`sklearn.preprocessing`, :func:`~sklearn.metrics.accuracy_score`."
    )
    out = rst_roles_to_markdown(text)
    assert_no_rst_role_markup(out)
    assert "[Guide](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)" in out
    assert "[Glossary](https://scikit-learn.org/stable/glossary.html#term-n_jobs)" in out
    assert "[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)" in out
    assert "fit" in out
    assert "joblib.parallel_backend" in out
    assert "[sklearn.preprocessing](https://scikit-learn.org/stable/api/sklearn.preprocessing.html#module-sklearn.preprocessing)" in out
    assert "[accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)" in out


def test_replace_sklearn_mods_links_to_api_module_reference():
    text = ":mod:`sklearn.covariance` and :mod:`sklearn.preprocessing`"
    out = rst_roles_to_markdown(text)
    assert (
        "[sklearn.covariance](https://scikit-learn.org/stable/api/sklearn.covariance.html#module-sklearn.covariance)"
        in out
    )
    assert (
        "[sklearn.preprocessing](https://scikit-learn.org/stable/api/sklearn.preprocessing.html#module-sklearn.preprocessing)"
        in out
    )


def test_replace_sklearn_classes_links_to_generated_api_reference():
    text = ":class:`~sklearn.tree.DecisionTreeClassifier`"
    out = rst_roles_to_markdown(text)
    assert (
        out
        == "[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)"
    )


def test_replace_sklearn_terms_links_to_glossary_anchor():
    text = ":term:`Glossary <random_state>`"
    out = rst_roles_to_markdown(text)
    assert (
        out
        == "[Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state)"
    )


def test_replace_sklearn_terms_hyphenates_space_separated_targets():
    text = ":term:`sparse graph`"
    out = rst_roles_to_markdown(text)
    assert (
        out
        == "[sparse graph](https://scikit-learn.org/stable/glossary.html#term-sparse-graph)"
    )


def test_replace_sklearn_terms_handles_multiline_label_target_payload():
    text = ":term:`Glossary\n    <n_jobs>`"
    out = rst_roles_to_markdown(text)
    assert out == "[Glossary](https://scikit-learn.org/stable/glossary.html#term-n_jobs)"


class FakeRoleModel:
    __doc__ = """
    Role-rich description using :term:`Glossary <n_jobs>` and :class:`~sklearn.svm.SVC`.

    Parameters
    ----------
    alpha : float, default=1.0
        Uses :meth:`fit`, :obj:`joblib.parallel_backend`, :mod:`sklearn.preprocessing`,
        :func:`~sklearn.metrics.accuracy_score`, and :ref:`User Guide <adaboost>`.

    Attributes
    ----------
    coef_ : ndarray
    """

    def get_params(self, deep=True):
        return {"alpha": 1.0}

    def __repr__(self):
        return "FakeRoleModel(alpha=1.0)"


def test_parse_model_desc_opt_out_preserves_rst_role_markup():
    model = FakeRoleModel()

    converted = parse_model_desc(model)
    raw = parse_model_desc(model, convert_rst_roles=False)

    assert_no_rst_role_markup(converted)
    assert ":term:`" in raw
    assert ":class:`" in raw


def test_parse_param_desc_opt_out_preserves_rst_role_markup():
    model = FakeRoleModel()

    converted = parse_param_desc(model)
    raw = parse_param_desc(model, convert_rst_roles=False)

    assert_no_rst_role_markup("\n".join(converted.values()))
    combined_raw = "\n".join(raw.values())
    assert ":meth:`" in combined_raw
    assert ":obj:`" in combined_raw
    assert ":mod:`" in combined_raw
    assert ":func:`" in combined_raw
    assert ":ref:`" in combined_raw
