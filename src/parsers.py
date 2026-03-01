import re

BASE_URL = "https://scikit-learn.org/stable/modules/"
SKLEARN_REF_MAP = {
    "accuracy_score": "model_evaluation.html#accuracy-score",
    "adaboost": "ensemble.html#adaboost",
    "array_api": "array_api.html",
    "bagging": "ensemble.html#bagging",
    "bernoulli_naive_bayes": "naive_bayes.html#bernoulli-naive-bayes",
    "categorical_naive_bayes": "naive_bayes.html#categorical-naive-bayes",
    "categorical_support_gbdt": "ensemble.html#support-for-categorical-features",
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
    "lda-qda": "discriminant_analysis.html",
    "logistic_regression": "linear_model.html#logistic-regression",
    "logistic-regression": "linear_model.html#logistic-regression",
    "logistic_regression_solvers": "linear_model.html#solvers",
    "liblinear_differences": "linear_model.html#differences-between-solvers",
    "monotonic_cst_gbdt": "ensemble.html#monotonic-constraints",
    "multinomial_naive_bayes": "naive_bayes.html#multinomial-naive-bayes",
    "sgd": "linear_model.html#sgd",
    "nearest-centroid-classification": "neighbors.html#nearest-centroid-classifier",
    "nearest-neighbors-classification": "neighbors.html#nearest-neighbors-classification",
    "nearest_centroid_classifier": "neighbors.html#nearest-centroid-classifier",
    "lda_qda": "discriminant_analysis.html",
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


def replace_sklearn_refs(text: str) -> str:
    def replace_targeted_ref(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        slug = match.group(2).strip()

        if slug in SKLEARN_REF_MAP:
            return f"[{label}]({BASE_URL}{SKLEARN_REF_MAP[slug]})"
        return label

    text = REF_WITH_TARGET_PATTERN.sub(replace_targeted_ref, text)
    return REF_SIMPLE_PATTERN.sub(lambda match: match.group(1).strip(), text)


def parse_param_desc(model):
    params = model.get_params().keys()
    params = "|".join([p + " : " for p in params])

    params_desc = re.split(params, model.__doc__)[1:]
    params_desc[-1] = params_desc[-1].split("Attributes\n")[0]
    params_desc = {
        k[:-3]: replace_sklearn_refs("  \n".join(v.split("\n\n")))
        for k, v in zip(re.findall(params, model.__doc__), params_desc)
    }
    return params_desc


def parse_model_desc(model) -> str:
    """
    Return a compact markdown description of an sklearn estimator:
    - constructor-style repr (shows non-default params)
    - short docstring description (before 'Parameters')
    """
    doc = model.__doc__ or ""
    desc = doc.split("Parameters", 1)[0].strip()

    # Collapse excessive whitespace but keep paragraphs
    desc = "\n\n".join(p.strip() for p in desc.split("\n\n") if p.strip())
    desc = replace_sklearn_refs(desc)

    return f"```python\n{repr(model)}\n``` \n\n{desc}"
