from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar
import streamlit as st
T = TypeVar("T")


def none_or_widget(
    name: str,
    *wargs: Any,
    widget: Callable[..., T] = st.slider,
    checkbox_kwargs: Optional[dict[str, Any]] = None,
    **wkwargs: Any,
) -> Optional[T]:
    """
    If user ticks a checkbox, show a widget and return its value;
    otherwise return `None`.
    """
    default_checkbox_kwargs: dict[str, Any] = dict(
        help=(
            "Default parameter value is `None`. "
            "Select the checkbox to set another value."
        )
    )

    if checkbox_kwargs is None:
        checkbox_kwargs = default_checkbox_kwargs
    elif isinstance(checkbox_kwargs, dict):
        checkbox_kwargs = {**default_checkbox_kwargs, **checkbox_kwargs}

    name = " ".join(name.split("_")).capitalize()
    if st.checkbox(f"Set {name}", **checkbox_kwargs):
        return widget(name, *wargs, **wkwargs)
    return None


def lr_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}

    penalty_options = ["l1", "l2", "elasticnet", None]
    penalty_default = "l2"
    hp["penalty"] = st.selectbox(
        "Penalty",
        penalty_options,
        index=penalty_options.index(penalty_default),
        help=hp_desc["penalty"],
        key="lr_penalty",
    )
    # Conservative bounds: estimator does not define explicit limits for C.
    hp["C"] = st.number_input(
        "Inverse of Regularization Strength (C)",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.01,
        help=hp_desc["C"],
        key="lr_C",
    )
    hp["l1_ratio"] = st.number_input(
        "L1 Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help=hp_desc["l1_ratio"],
        key="lr_l1_ratio",
    )
    hp["dual"] = st.checkbox(
        "Dual Formulation",
        value=False,
        help=hp_desc["dual"],
        key="lr_dual",
    )
    # Conservative bounds: estimator does not define explicit limits for tol.
    hp["tol"] = st.number_input(
        "Tolerance",
        min_value=0.0,
        max_value=1.0,
        value=0.0001,
        step=0.0001,
        format="%.2e",
        help=hp_desc["tol"],
        key="lr_tol",
    )
    hp["fit_intercept"] = st.checkbox(
        "Fit Intercept",
        value=True,
        help=hp_desc["fit_intercept"],
        key="lr_fit_intercept",
    )
    # Conservative bounds: estimator does not define explicit limits for intercept_scaling.
    hp["intercept_scaling"] = st.number_input(
        "Intercept Scaling",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help=hp_desc["intercept_scaling"],
        key="lr_intercept_scaling",
    )
    class_weight_options = [None, "balanced"]
    class_weight_default = None
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(class_weight_default),
        help=hp_desc["class_weight"],
        key="lr_class_weight",
    )
    solver_options = [
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga",
    ]
    solver_default = "lbfgs"
    hp["solver"] = st.selectbox(
        "Solver",
        solver_options,
        index=solver_options.index(solver_default),
        help=hp_desc["solver"],
        key="lr_solver",
    )
    # Conservative bounds: estimator does not define explicit limits for max_iter.
    hp["max_iter"] = st.slider(
        "Max Iterations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help=hp_desc["max_iter"],
        key="lr_max_iter",
    )
    return hp


def knc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["n_neighbors"] = st.slider(
        "N Neighbors",
        min_value=1,
        max_value=100,
        value=5,
        help=hp_desc["n_neighbors"],
        key="knc_n_neighbors",
    )
    hp["weights"] = st.selectbox(
        "Weights",
        ["uniform", "distance"],
        index=0,
        help=hp_desc["weights"],
        key="knc_weights",
    )
    hp["algorithm"] = st.selectbox(
        "Algorithm",
        ["auto", "ball_tree", "kd_tree", "brute"],
        index=0,
        help=hp_desc["algorithm"],
        key="knc_algorithm",
    )
    hp["leaf_size"] = st.slider(
        "Leaf Size",
        min_value=1,
        max_value=100,
        value=30,
        help=hp_desc["leaf_size"],
        key="knc_leaf_size",
    )
    hp["p"] = st.slider(
        "Power",
        min_value=1,
        max_value=100,
        value=2,
        help=hp_desc["p"],
        key="knc_p",
    )
    hp["metric"] = st.selectbox(
        "Metric",
        [
            "minkowski", "cityblock", "cosine", "euclidean",
            "haversine", "l1", "l2", "manhattan", "nan_euclidean",
        ],
        index=0,
        help=hp_desc["metric"],
        key="knc_metric",
    )
    return hp


def nc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["metric"] = st.selectbox(
        "Metric",
        ["euclidean", "manhattan"],
        index=0,
        help=hp_desc["metric"],
        key="nc_metric",
    )
    # Conservative bounds: estimator does not define explicit limits for shrink_threshold.
    hp["shrink_threshold"] = none_or_widget(
        "shrink_threshold",
        widget=st.number_input,
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help=hp_desc["shrink_threshold"],
        key="nc_shrink_threshold",
        checkbox_kwargs={"key": "nc_shrink_threshold__is_set"},
    )
    hp["priors"] = st.selectbox(
        "Priors",
        ["uniform", "empirical"],
        index=0,
        help=hp_desc["priors"],
        key="nc_priors",
    )
    return hp


def rnc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    # Conservative bounds: estimator does not define explicit limits for radius.
    hp["radius"] = st.number_input(
        "Radius",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help=hp_desc["radius"],
        key="rnc_radius",
    )
    hp["weights"] = st.selectbox(
        "Weights",
        ["uniform", "distance"],
        index=0,
        help=hp_desc["weights"],
        key="rnc_weights",
    )
    hp["algorithm"] = st.selectbox(
        "Algorithm",
        ["auto", "ball_tree", "kd_tree", "brute"],
        index=0,
        help=hp_desc["algorithm"],
        key="rnc_algorithm",
    )
    hp["leaf_size"] = st.slider(
        "Leaf Size",
        min_value=1,
        max_value=100,
        value=30,
        help=hp_desc["leaf_size"],
        key="rnc_leaf_size",
    )
    # Conservative bounds: estimator expects positive p values.
    hp["p"] = st.slider(
        "Power",
        min_value=1,
        max_value=10,
        value=2,
        help=hp_desc["p"],
        key="rnc_p",
    )
    hp["metric"] = st.selectbox(
        "Metric",
        [
            "minkowski", "cityblock", "cosine", "euclidean",
            "haversine", "l1", "l2", "manhattan", "nan_euclidean",
        ],
        index=0,
        help=hp_desc["metric"],
        key="rnc_metric",
    )
    return hp


def dtc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["criterion"] = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=0,
        help=hp_desc["criterion"],
        key="dtc_criterion",
    )
    hp["splitter"] = st.selectbox(
        "Splitter",
        ["best", "random"],
        index=0,
        help=hp_desc["splitter"],
        key="dtc_splitter",
    )
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
        key="dtc_max_depth",
        checkbox_kwargs={"key": "dtc_max_depth__is_set"},
    )
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
        key="dtc_min_samples_split",
    )
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help=hp_desc["min_samples_leaf"],
        key="dtc_min_samples_leaf",
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help=hp_desc["min_weight_fraction_leaf"],
        key="dtc_min_weight_fraction_leaf",
    )
    hp["max_features"] = st.selectbox(
        "Max Features",
        [None, "sqrt", "log2"],
        index=0,
        help=hp_desc["max_features"],
        key="dtc_max_features",
    )
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=100,
        value=10,
        help=hp_desc["max_leaf_nodes"],
        key="dtc_max_leaf_nodes",
        checkbox_kwargs={"key": "dtc_max_leaf_nodes__is_set"},
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
        key="dtc_min_impurity_decrease",
    )
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        [None, "balanced"],
        index=0,
        help=hp_desc["class_weight"],
        key="dtc_class_weight",
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
        key="dtc_ccp_alpha",
    )
    return hp


def etc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["criterion"] = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=0,
        help=hp_desc["criterion"],
        key="etc_criterion",
    )
    hp["splitter"] = st.selectbox(
        "Splitter",
        ["random", "best"],
        index=0,
        help=hp_desc["splitter"],
        key="etc_splitter",
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
        key="etc_max_depth",
        checkbox_kwargs={"key": "etc_max_depth__is_set"},
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
        key="etc_min_samples_split",
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help=hp_desc["min_samples_leaf"],
        key="etc_min_samples_leaf",
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help=hp_desc["min_weight_fraction_leaf"],
        key="etc_min_weight_fraction_leaf",
    )
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index("sqrt"),
        help=hp_desc["max_features"],
        key="etc_max_features",
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=100,
        value=10,
        help=hp_desc["max_leaf_nodes"],
        key="etc_max_leaf_nodes",
        checkbox_kwargs={"key": "etc_max_leaf_nodes__is_set"},
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
        key="etc_min_impurity_decrease",
    )
    class_weight_options = [None, "balanced"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
        help=hp_desc["class_weight"],
        key="etc_class_weight",
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
        key="etc_ccp_alpha",
    )
    return hp


def rfc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=100,
        help=hp_desc["n_estimators"],
        key="rfc_n_estimators",
    )
    criterion_options = ["gini", "entropy", "log_loss"]
    hp["criterion"] = st.selectbox(
        "Criterion",
        criterion_options,
        index=criterion_options.index("gini"),
        help=hp_desc["criterion"],
        key="rfc_criterion",
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
        key="rfc_max_depth",
        checkbox_kwargs={"key": "rfc_max_depth__is_set"},
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
        key="rfc_min_samples_split",
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help=hp_desc["min_samples_leaf"],
        key="rfc_min_samples_leaf",
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help=hp_desc["min_weight_fraction_leaf"],
        key="rfc_min_weight_fraction_leaf",
    )
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index("sqrt"),
        help=hp_desc["max_features"],
        key="rfc_max_features",
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=100,
        value=10,
        help=hp_desc["max_leaf_nodes"],
        key="rfc_max_leaf_nodes",
        checkbox_kwargs={"key": "rfc_max_leaf_nodes__is_set"},
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
        key="rfc_min_impurity_decrease",
    )
    hp["bootstrap"] = st.checkbox(
        "Bootstrap",
        value=True,
        help=hp_desc["bootstrap"],
        key="rfc_bootstrap",
    )
    hp["oob_score"] = st.checkbox(
        "OOB Score",
        value=False,
        help=hp_desc["oob_score"],
        key="rfc_oob_score",
    )
    class_weight_options = [None, "balanced", "balanced_subsample"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
        help=hp_desc["class_weight"],
        key="rfc_class_weight",
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
        key="rfc_ccp_alpha",
    )
    # Conservative bounds: max_samples allows int sample counts or (0, 1] fractions.
    hp["max_samples"] = none_or_widget(
        "max_samples",
        widget=st.number_input,
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help=hp_desc["max_samples"],
        key="rfc_max_samples",
        checkbox_kwargs={"key": "rfc_max_samples__is_set"},
    )
    return hp


def etsc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=100,
        help=hp_desc["n_estimators"],
        key="etsc_n_estimators",
    )
    hp["criterion"] = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=0,
        help=hp_desc["criterion"],
        key="etsc_criterion",
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
        key="etsc_max_depth",
        checkbox_kwargs={"key": "etsc_max_depth__is_set"},
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
        key="etsc_min_samples_split",
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help=hp_desc["min_samples_leaf"],
        key="etsc_min_samples_leaf",
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help=hp_desc["min_weight_fraction_leaf"],
        key="etsc_min_weight_fraction_leaf",
    )
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index("sqrt"),
        help=hp_desc["max_features"],
        key="etsc_max_features",
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=100,
        value=10,
        help=hp_desc["max_leaf_nodes"],
        key="etsc_max_leaf_nodes",
        checkbox_kwargs={"key": "etsc_max_leaf_nodes__is_set"},
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
        key="etsc_min_impurity_decrease",
    )
    hp["bootstrap"] = st.checkbox(
        "Bootstrap",
        value=False,
        help=hp_desc["bootstrap"],
        key="etsc_bootstrap",
    )
    hp["oob_score"] = st.checkbox(
        "OOB Score",
        value=False,
        help=hp_desc["oob_score"],
        key="etsc_oob_score",
    )
    class_weight_options = [None, "balanced", "balanced_subsample"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
        help=hp_desc["class_weight"],
        key="etsc_class_weight",
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
        key="etsc_ccp_alpha",
    )
    # Conservative bounds: max_samples allows int sample counts or (0, 1] fractions.
    hp["max_samples"] = none_or_widget(
        "max_samples",
        widget=st.number_input,
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help=hp_desc["max_samples"],
        key="etsc_max_samples",
        checkbox_kwargs={"key": "etsc_max_samples__is_set"},
    )
    return hp


def gbc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    loss_options = ["log_loss", "exponential"]
    hp["loss"] = st.selectbox(
        "Loss",
        loss_options,
        index=loss_options.index("log_loss"),
        help=hp_desc["loss"],
        key="gbc_loss",
    )
    hp["learning_rate"] = st.number_input(
        "Learning Rate",
        min_value=0.0,
        value=0.1,
        step=0.01,
        help=hp_desc["learning_rate"],
        key="gbc_learning_rate",
    )
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=100,
        help=hp_desc["n_estimators"],
        key="gbc_n_estimators",
    )
    hp["subsample"] = st.number_input(
        "Subsample",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help=hp_desc["subsample"],
        key="gbc_subsample",
    )
    criterion_options = ["friedman_mse", "squared_error"]
    hp["criterion"] = st.selectbox(
        "Criterion",
        criterion_options,
        index=criterion_options.index("friedman_mse"),
        help=hp_desc["criterion"],
        key="gbc_criterion",
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=200,
        value=2,
        help=hp_desc["min_samples_split"],
        key="gbc_min_samples_split",
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=200,
        value=1,
        help=hp_desc["min_samples_leaf"],
        key="gbc_min_samples_leaf",
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help=hp_desc["min_weight_fraction_leaf"],
        key="gbc_min_weight_fraction_leaf",
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = st.slider(
        "Max Depth",
        min_value=1,
        max_value=20,
        value=3,
        help=hp_desc["max_depth"],
        key="gbc_max_depth",
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
        key="gbc_min_impurity_decrease",
    )
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index(None),
        help=hp_desc["max_features"],
        key="gbc_max_features",
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=200,
        value=10,
        help=hp_desc["max_leaf_nodes"],
        key="gbc_max_leaf_nodes",
        checkbox_kwargs={"key": "gbc_max_leaf_nodes__is_set"},
    )
    hp["validation_fraction"] = st.number_input(
        "Validation Fraction",
        min_value=0.01,
        max_value=0.99,
        value=0.1,
        step=0.01,
        help=hp_desc["validation_fraction"],
        key="gbc_validation_fraction",
    )
    # Conservative bounds: estimator does not define explicit limits for n_iter_no_change.
    hp["n_iter_no_change"] = none_or_widget(
        "n_iter_no_change",
        min_value=1,
        max_value=200,
        value=10,
        help=hp_desc["n_iter_no_change"],
        key="gbc_n_iter_no_change",
        checkbox_kwargs={"key": "gbc_n_iter_no_change__is_set"},
    )
    hp["tol"] = st.number_input(
        "Tolerance",
        min_value=0.0,
        value=0.0001,
        step=0.0001,
        format="%.2e",
        help=hp_desc["tol"],
        key="gbc_tol",
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
        key="gbc_ccp_alpha",
    )
    return hp


def abc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=50,
        help=hp_desc["n_estimators"],
        key="abc_n_estimators",
    )
    # Conservative bounds: estimator does not define explicit limits for learning_rate.
    hp["learning_rate"] = st.number_input(
        "Learning Rate",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help=hp_desc["learning_rate"],
        key="abc_learning_rate",
    )
    return hp


def bc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=10,
        help=hp_desc["n_estimators"],
        key="bc_n_estimators",
    )
    # Conservative bounds: max_samples allows (0, 1] fractions.
    hp["max_samples"] = none_or_widget(
        "max_samples",
        widget=st.number_input,
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help=hp_desc["max_samples"],
        key="bc_max_samples",
        checkbox_kwargs={"key": "bc_max_samples__is_set"},
    )
    # Conservative bounds: max_features allows (0, 1] fractions.
    hp["max_features"] = st.number_input(
        "Max Features",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help=hp_desc["max_features"],
        key="bc_max_features",
    )
    hp["bootstrap"] = st.checkbox(
        "Bootstrap",
        value=True,
        help=hp_desc["bootstrap"],
        key="bc_bootstrap",
    )
    hp["bootstrap_features"] = st.checkbox(
        "Bootstrap Features",
        value=False,
        help=hp_desc["bootstrap_features"],
        key="bc_bootstrap_features",
    )
    hp["oob_score"] = st.checkbox(
        "OOB Score",
        value=False,
        help=hp_desc["oob_score"],
        key="bc_oob_score",
    )
    return hp


def hgbc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    loss_options = ["log_loss"]
    hp["loss"] = st.selectbox(
        "Loss",
        loss_options,
        index=loss_options.index("log_loss"),
        help=hp_desc["loss"],
        key="hgbc_loss",
    )
    hp["learning_rate"] = st.number_input(
        "Learning Rate",
        min_value=0.0,
        value=0.1,
        step=0.01,
        help=hp_desc["learning_rate"],
        key="hgbc_learning_rate",
    )
    # Conservative bounds: estimator does not define explicit limits for max_iter.
    hp["max_iter"] = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        help=hp_desc["max_iter"],
        key="hgbc_max_iter",
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=255,
        value=31,
        help=hp_desc["max_leaf_nodes"],
        key="hgbc_max_leaf_nodes",
        checkbox_kwargs={
            "key": "hgbc_max_leaf_nodes__is_set",
            "value": True,
        },
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=50,
        value=3,
        help=hp_desc["max_depth"],
        key="hgbc_max_depth",
        checkbox_kwargs={"key": "hgbc_max_depth__is_set"},
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=200,
        value=20,
        help=hp_desc["min_samples_leaf"],
        key="hgbc_min_samples_leaf",
    )
    # Conservative bounds: estimator does not define explicit limits for l2_regularization.
    hp["l2_regularization"] = st.number_input(
        "L2 Regularization",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.01,
        help=hp_desc["l2_regularization"],
        key="hgbc_l2_regularization",
    )
    # Conservative bounds: estimator does not define explicit limits for max_features.
    hp["max_features"] = st.number_input(
        "Max Features",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help=hp_desc["max_features"],
        key="hgbc_max_features",
    )
    hp["max_bins"] = st.slider(
        "Max Bins",
        min_value=2,
        max_value=255,
        value=255,
        help=hp_desc["max_bins"],
        key="hgbc_max_bins",
    )
    categorical_features_options = ["from_dtype", None]
    hp["categorical_features"] = st.selectbox(
        "Categorical Features",
        categorical_features_options,
        index=categorical_features_options.index("from_dtype"),
        help=hp_desc["categorical_features"],
        key="hgbc_categorical_features",
    )
    interaction_cst_options = [None, "pairwise", "no_interactions"]
    hp["interaction_cst"] = st.selectbox(
        "Interaction Constraints",
        interaction_cst_options,
        index=interaction_cst_options.index(None),
        help=hp_desc["interaction_cst"],
        key="hgbc_interaction_cst",
    )
    early_stopping_options = ["auto", True, False]
    hp["early_stopping"] = st.selectbox(
        "Early Stopping",
        early_stopping_options,
        index=early_stopping_options.index("auto"),
        help=hp_desc["early_stopping"],
        key="hgbc_early_stopping",
    )
    hp["validation_fraction"] = none_or_widget(
        "validation_fraction",
        widget=st.number_input,
        min_value=0.01,
        max_value=0.99,
        value=0.1,
        step=0.01,
        help=hp_desc["validation_fraction"],
        key="hgbc_validation_fraction",
        checkbox_kwargs={
            "key": "hgbc_validation_fraction__is_set",
            "value": True,
        },
    )
    # Conservative bounds: estimator does not define explicit limits for n_iter_no_change.
    hp["n_iter_no_change"] = st.slider(
        "N Iterations No Change",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
        help=hp_desc["n_iter_no_change"],
        key="hgbc_n_iter_no_change",
    )
    # Conservative bounds: estimator does not define explicit limits for tol.
    hp["tol"] = st.number_input(
        "Tolerance",
        min_value=0.0,
        max_value=1.0,
        value=0.0000001,
        step=0.0000001,
        format="%.2e",
        help=hp_desc["tol"],
        key="hgbc_tol",
    )
    class_weight_options = [None, "balanced"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
        help=hp_desc["class_weight"],
        key="hgbc_class_weight",
    )
    return hp


def sgdc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    loss_options = [
        "hinge",
        "log_loss",
        "modified_huber",
        "squared_hinge",
        "perceptron",
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ]
    hp["loss"] = st.selectbox(
        "Loss",
        loss_options,
        index=loss_options.index("hinge"),
        help=hp_desc["loss"],
        key="sgdc_loss",
    )
    penalty_options = ["l2", "l1", "elasticnet", None]
    hp["penalty"] = st.selectbox(
        "Penalty",
        penalty_options,
        index=penalty_options.index("l2"),
        help=hp_desc["penalty"],
        key="sgdc_penalty",
    )
    # Conservative bounds: estimator does not define explicit limits for alpha.
    hp["alpha"] = st.number_input(
        "Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.0001,
        step=0.0001,
        format="%.4g",
        help=hp_desc["alpha"],
        key="sgdc_alpha",
    )
    hp["l1_ratio"] = st.number_input(
        "l1_ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help=hp_desc["l1_ratio"],
        key="sgdc_l1_ratio",
    )
    hp["fit_intercept"] = st.checkbox(
        "Fit Intercept",
        value=True,
        help=hp_desc["fit_intercept"],
        key="sgdc_fit_intercept",
    )
    # Conservative bounds: estimator does not define explicit limits for max_iter.
    hp["max_iter"] = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=5000,
        value=1000,
        step=50,
        help=hp_desc["max_iter"],
        key="sgdc_max_iter",
    )
    # Conservative bounds: estimator does not define explicit limits for tol.
    hp["tol"] = st.number_input(
        "tol",
        min_value=0.0,
        max_value=1.0,
        value=0.001,
        step=0.0001,
        format="%.2e",
        help=hp_desc["tol"],
        key="sgdc_tol",
    )
    hp["shuffle"] = st.checkbox(
        "Shuffle",
        value=True,
        help=hp_desc["shuffle"],
        key="sgdc_shuffle",
    )
    # Conservative bounds: estimator does not define explicit limits for epsilon.
    hp["epsilon"] = st.number_input(
        "Epsilon",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help=hp_desc["epsilon"],
        key="sgdc_epsilon",
    )
    learning_rate_options = [
        "optimal",
        "constant",
        "invscaling",
        "adaptive",
        "pa1",
        "pa2",
    ]
    hp["learning_rate"] = st.selectbox(
        "Learning Rate Schedule",
        learning_rate_options,
        index=learning_rate_options.index("optimal"),
        help=hp_desc["learning_rate"],
        key="sgdc_learning_rate",
    )
    # Conservative bounds: estimator does not define explicit limits for eta0.
    hp["eta0"] = st.number_input(
        "Eta0",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.01,
        help=hp_desc["eta0"],
        key="sgdc_eta0",
    )
    # Conservative bounds: estimator does not define explicit limits for power_t.
    hp["power_t"] = st.number_input(
        "Power T",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=hp_desc["power_t"],
        key="sgdc_power_t",
    )
    hp["early_stopping"] = st.checkbox(
        "Early Stopping",
        value=False,
        help=hp_desc["early_stopping"],
        key="sgdc_early_stopping",
    )
    hp["validation_fraction"] = st.number_input(
        "Validation Fraction",
        min_value=0.01,
        max_value=0.99,
        value=0.1,
        step=0.01,
        help=hp_desc["validation_fraction"],
        key="sgdc_validation_fraction",
    )
    # Conservative bounds: estimator does not define explicit limits for n_iter_no_change.
    hp["n_iter_no_change"] = st.slider(
        "N Iterations No Change",
        min_value=1,
        max_value=200,
        value=5,
        step=1,
        help=hp_desc["n_iter_no_change"],
        key="sgdc_n_iter_no_change",
    )
    hp["average"] = st.checkbox(
        "Average",
        value=False,
        help=hp_desc["average"],
        key="sgdc_average",
    )
    return hp


def lda_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    solver_options = ["svd", "lsqr", "eigen"]
    hp["solver"] = st.selectbox(
        "Solver",
        solver_options,
        index=solver_options.index("svd"),
        help=hp_desc["solver"],
        key="lda_solver",
    )
    shrinkage_mode_options = [None, "auto", "float"]
    shrinkage_mode = st.selectbox(
        "Shrinkage Mode",
        shrinkage_mode_options,
        index=shrinkage_mode_options.index(None),
        help=hp_desc["shrinkage"],
        key="lda_shrinkage_mode",
    )
    if shrinkage_mode == "float":
        # Conservative bounds: estimator does not define explicit limits for shrinkage.
        hp["shrinkage"] = st.number_input(
            "Shrinkage",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help=hp_desc["shrinkage"],
            key="lda_shrinkage",
        )
    else:
        hp["shrinkage"] = shrinkage_mode
    # Conservative bounds: estimator does not define explicit limits for n_components.
    hp["n_components"] = none_or_widget(
        "n_components",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
        help=hp_desc["n_components"],
        key="lda_n_components",
        checkbox_kwargs={"key": "lda_n_components__is_set"},
    )
    hp["store_covariance"] = st.checkbox(
        "Store Covariance",
        value=False,
        help=hp_desc["store_covariance"],
        key="lda_store_covariance",
    )
    # Conservative bounds: estimator does not define explicit limits for tol.
    hp["tol"] = st.number_input(
        "Tolerance",
        min_value=0.0,
        max_value=1.0,
        value=0.0001,
        step=0.0001,
        format="%.2e",
        help=hp_desc["tol"],
        key="lda_tol",
    )
    return hp


def qda_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    solver_options = ["svd", "eigen"]
    hp["solver"] = st.selectbox(
        "Solver",
        solver_options,
        index=solver_options.index("svd"),
        help=hp_desc["solver"],
        key="qda_solver",
    )
    shrinkage_mode_options = [None, "auto", "float"]
    shrinkage_mode = st.selectbox(
        "Shrinkage Mode",
        shrinkage_mode_options,
        index=shrinkage_mode_options.index(None),
        help=hp_desc["shrinkage"],
        key="qda_shrinkage_mode",
    )
    if shrinkage_mode == "float":
        # Conservative bounds: estimator does not define explicit limits for shrinkage.
        hp["shrinkage"] = st.number_input(
            "Shrinkage",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help=hp_desc["shrinkage"],
            key="qda_shrinkage",
        )
    else:
        hp["shrinkage"] = shrinkage_mode
    hp["reg_param"] = st.number_input(
        "Reg Param",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=hp_desc["reg_param"],
        key="qda_reg_param",
    )
    hp["store_covariance"] = st.checkbox(
        "Store Covariance",
        value=False,
        help=hp_desc["store_covariance"],
        key="qda_store_covariance",
    )
    # Conservative bounds: estimator does not define explicit limits for tol.
    hp["tol"] = st.number_input(
        "Tolerance",
        min_value=0.0,
        max_value=1.0,
        value=0.0001,
        step=0.0001,
        format="%.2e",
        help=hp_desc["tol"],
        key="qda_tol",
    )
    return hp
