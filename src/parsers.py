import re
from typing import Callable

BASE_URL = "https://scikit-learn.org/stable/modules/"
GENERATED_BASE_URL = "https://scikit-learn.org/stable/modules/generated/"
GLOSSARY_URL = "https://scikit-learn.org/stable/glossary.html"
API_BASE_URL = "https://scikit-learn.org/stable/api/"
SKLEARN_REF_MAP = {
    "accuracy_score": "model_evaluation.html#accuracy-score",
    "adaboost": "ensemble.html#adaboost",
    "array_api": "array_api.html",
    "bagging": "ensemble.html#bagging",
    "bernoulli_naive_bayes": "naive_bayes.html#bernoulli-naive-bayes",
    "categorical_naive_bayes": "naive_bayes.html#categorical-naive-bayes",
    "categorical_support_gbdt": "ensemble.html#categorical-support-gbdt",
    "classification": "neighbors.html#nearest-neighbors-classification",
    "complement_naive_bayes": "naive_bayes.html#complement-naive-bayes",
    "decision-trees": "tree.html#classification",
    "gradient_boosting": "ensemble.html#gradient-boosting",
    "gradient-boosting": "ensemble.html#gradient-boosting",
    "forest": "ensemble.html#forest",
    "gaussian_naive_bayes": "naive_bayes.html#gaussian-naive-bayes",
    "gaussian_process": "gaussian_process.html#gaussian-process-classification-gpc",
    "histogram_based_gradient_boosting": "ensemble.html#histogram-based-gradient-boosting",
    "histogram-based-gradient-boosting": "ensemble.html#histogram-based-gradient-boosting",
    "ice-vs-pdp": "partial_dependence.html#individual-conditional-expectation-ice-plot",
    "label_propagation": "semi_supervised.html#label-propagation",
    "lda-qda": "lda_qda.html",
    "lda_qda": "lda_qda.html",
    "logistic_regression": "linear_model.html#logistic-regression",
    "logistic-regression": "linear_model.html#logistic-regression",
    "logistic_regression_solvers": "linear_model.html#solvers",
    "liblinear_differences": "linear_model.html#differences-between-solvers",
    "monotonic_cst_gbdt": "ensemble.html#monotonic-constraints",
    "multinomial_naive_bayes": "naive_bayes.html#multinomial-naive-bayes",
    "sgd": "sgd.html#sgd",
    "nearest-centroid-classification": "neighbors.html#nearest-centroid-classifier",
    "nearest-neighbors-classification": "neighbors.html#nearest-neighbors-classification",
    "nearest_centroid_classifier": "neighbors.html#nearest-centroid-classifier",
    "neighbors": "neighbors.html",
    "nu_svc": "svm.html#nusvc",
    "regularized-logistic-loss": "linear_model.html#binary-case",
    "scores_probabilities": "svm.html#scores-and-probabilities",
    "shrinking_svm": "svm.html#tips-on-practical-use",
    "sgd_mathematical_formulation": "sgd.html#mathematical-formulation",
    "svm_classification": "svm.html#classification",
    "svm_multi_class": "svm.html#multi-class-classification",
    "tree": "tree.html#classification",
    "decision_trees": "tree.html#classification",
}
REF_WITH_TARGET_PATTERN = re.compile(r":ref:`([^`<>]+?)\s*<\s*([^>]+?)\s*>`")
REF_SIMPLE_PATTERN = re.compile(r":ref:`([^`]+?)`")
TERM_PATTERN = re.compile(r":term:`([^`]+?)`")
CLASS_PATTERN = re.compile(r":class:`([^`]+?)`")
METH_PATTERN = re.compile(r":meth:`([^`]+?)`")
OBJ_PATTERN = re.compile(r":obj:`([^`]+?)`")
MOD_PATTERN = re.compile(r":mod:`([^`]+?)`")
FUNC_PATTERN = re.compile(r":func:`([^`]+?)`")
EXTERNAL_LINK_PATTERN = re.compile(r"`([^`<>]+?)\s*<https?://([^`<>]+?)>`_?")
RST_DIRECTIVES = (
    "versionadded",
    "versionchanged",
    "following",
    "deprecated",
    "warning",
    "note",
    "seealso",
    "signature",
    "are",
)
DIRECTIVE_START_PATTERN = re.compile(
    rf"^(\s*)\.\.\s*({'|'.join(RST_DIRECTIVES)})::\s*(.*)$"
)


def _split_rst_role_content(content: str) -> tuple[str, str]:
    raw = " ".join(content.strip().split())
    is_short_form = raw.startswith("~")
    label_target_match = re.match(r"^(.*?)<\s*(.+?)\s*>$", raw)
    if label_target_match is not None:
        label = label_target_match.group(1).strip()
        target = label_target_match.group(2).strip()
        return label, target
    clean_target = raw.lstrip("~")
    if is_short_form and "." in clean_target:
        label = clean_target.rsplit(".", 1)[-1]
    else:
        label = clean_target
    return label, clean_target


def _markdown_link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def _generated_api_url(target: str) -> str:
    return f"{GENERATED_BASE_URL}{target}.html"


def _sklearn_target_link(label: str, target: str) -> str:
    if target.startswith("sklearn."):
        return _markdown_link(label, _generated_api_url(target))
    return label


def _replace_role(
    text: str,
    pattern: re.Pattern[str],
    resolver: Callable[[str, str], str],
) -> str:
    def replace_match(match: re.Match[str]) -> str:
        label, target = _split_rst_role_content(match.group(1))
        return resolver(label, target)

    return pattern.sub(replace_match, text)


def replace_rst_directives(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        match = DIRECTIVE_START_PATTERN.match(line)
        if match is None:
            out.append(line)
            i += 1
            continue

        base_indent, directive_name, inline_body = match.groups()
        body_lines: list[str] = []
        j = i + 1
        while j < len(lines):
            candidate = lines[j]
            if candidate.startswith(base_indent + " ") or candidate.startswith(base_indent + "\t"):
                body_lines.append(candidate.strip())
                j += 1
                continue
            if candidate.strip() == "":
                body_lines.append("")
                j += 1
                continue
            break

        directive_title = directive_name.replace("_", " ").title()
        inline_body = inline_body.strip()

        if inline_body:
            out.append(f"> **{directive_title}**: {inline_body}")
        else:
            out.append(f"> **{directive_title}**")

        for body in body_lines:
            if body:
                out.append(f"> {body}")
            else:
                out.append(">")

        # Keep adjacent directives visually separated in markdown rendering.
        out.append("")
        i = j

    return "\n".join(out)


def replace_external_links(text: str) -> str:
    def replace_external(match: re.Match[str]) -> str:
        label = " ".join(match.group(1).split())
        target = match.group(2).strip()
        scheme = "https" if "<https://" in match.group(0) else "http"
        return _markdown_link(label, f"{scheme}://{target}")

    return EXTERNAL_LINK_PATTERN.sub(replace_external, text)


def replace_sklearn_refs(text: str) -> str:
    def replace_targeted_ref(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        slug = match.group(2).strip()

        if slug in SKLEARN_REF_MAP:
            return f"[{label}]({BASE_URL}{SKLEARN_REF_MAP[slug]})"
        return label

    text = REF_WITH_TARGET_PATTERN.sub(replace_targeted_ref, text)
    return REF_SIMPLE_PATTERN.sub(lambda match: match.group(1).strip(), text)


def replace_sklearn_terms(text: str) -> str:
    def resolve_term(label: str, target: str) -> str:
        slug = target.strip().replace(" ", "-")
        return _markdown_link(label, f"{GLOSSARY_URL}#term-{slug}")

    return _replace_role(text, TERM_PATTERN, resolve_term)


def replace_sklearn_classes(text: str) -> str:
    return _replace_role(text, CLASS_PATTERN, _sklearn_target_link)


def replace_sklearn_meths(text: str) -> str:
    def resolve_meth(label: str, target: str) -> str:
        if target.startswith("sklearn.") and "." in target:
            parent, _ = target.rsplit(".", 1)
            return _markdown_link(
                label, f"{_generated_api_url(parent)}#{target}"
            )
        return label

    return _replace_role(text, METH_PATTERN, resolve_meth)


def replace_sklearn_objs(text: str) -> str:
    return _replace_role(text, OBJ_PATTERN, _sklearn_target_link)


def replace_sklearn_mods(text: str) -> str:
    def resolve_mod(label: str, target: str) -> str:
        if target.startswith("sklearn."):
            return _markdown_link(
                label, f"{API_BASE_URL}{target}.html#module-{target}"
            )
        return label

    return _replace_role(text, MOD_PATTERN, resolve_mod)


def replace_sklearn_funcs(text: str) -> str:
    return _replace_role(text, FUNC_PATTERN, _sklearn_target_link)


def rst_roles_to_markdown(text: str) -> str:
    text = replace_rst_directives(text)
    text = replace_external_links(text)
    text = replace_sklearn_refs(text)
    text = replace_sklearn_terms(text)
    text = replace_sklearn_classes(text)
    text = replace_sklearn_meths(text)
    text = replace_sklearn_objs(text)
    text = replace_sklearn_mods(text)
    text = replace_sklearn_funcs(text)
    return text


def parse_param_desc(model, convert_rst_roles: bool = True) -> dict[str, str]:
    params = model.get_params().keys()
    n_params = len(params)
    pattern = "|".join([p + " : " for p in params])

    param_chunks = re.split(pattern, model.__doc__)[1: n_params + 1]
    param_chunks[-1] = param_chunks[-1].split("Attributes\n")[0]

    parser = rst_roles_to_markdown if convert_rst_roles else (lambda text: text)

    params_desc = {
        str(k[:-3]): parser("  \n".join(v.split("\n\n")))
        for k, v in zip(re.findall(pattern, model.__doc__), param_chunks)
    }

    return params_desc


def parse_model_desc(model, convert_rst_roles: bool = True) -> str:
    """
    Return a compact markdown description of an sklearn estimator:
    - constructor-style repr (shows non-default params)
    - short docstring description (before 'Parameters')
    """
    doc = model.__doc__ or ""
    desc = doc.split("Parameters", 1)[0].strip()

    # Collapse excessive whitespace but keep paragraphs
    desc = "\n\n".join(p.strip() for p in desc.split("\n\n") if p.strip())
    if convert_rst_roles:
        desc = rst_roles_to_markdown(desc)

    return f"```python\n{repr(model)}\n``` \n\n{desc}"
