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
    )
    # Conservative bounds: estimator does not define explicit limits for C.
    hp["C"] = st.number_input(
        "Inverse of Regularization Strength (C)",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.01,
        help=hp_desc["C"],
    )
    hp["l1_ratio"] = st.number_input(
        "L1 Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help=hp_desc["l1_ratio"],
    )
    hp["dual"] = st.checkbox(
        "Dual Formulation",
        value=False,
        help=hp_desc["dual"],
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
    )
    hp["fit_intercept"] = st.checkbox(
        "Fit Intercept",
        value=True,
        help=hp_desc["fit_intercept"],
    )
    # Conservative bounds: estimator does not define explicit limits for intercept_scaling.
    hp["intercept_scaling"] = st.number_input(
        "Intercept Scaling",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help=hp_desc["intercept_scaling"],
    )
    class_weight_options = [None, "balanced"]
    class_weight_default = None
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(class_weight_default),
        help=hp_desc["class_weight"],
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
    )
    # Conservative bounds: estimator does not define explicit limits for max_iter.
    hp["max_iter"] = st.slider(
        "Max Iterations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help=hp_desc["max_iter"],
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
    )
    hp["weights"] = st.selectbox(
        "Weights",
        ["uniform", "distance"],
        index=0,
        help=hp_desc["weights"],
    )
    hp["algorithm"] = st.selectbox(
        "Algorithm",
        ["auto", "ball_tree", "kd_tree", "brute"],
        index=0,
        help=hp_desc["algorithm"],
    )
    hp["leaf_size"] = st.slider(
        "Leaf Size",
        min_value=1,
        max_value=100,
        value=30,
        help=hp_desc["leaf_size"],
    )
    hp["p"] = st.slider(
        "Power",
        min_value=1,
        max_value=100,
        value=2,
        help=hp_desc["p"],
    )
    hp["metric"] = st.selectbox(
        "Metric",
        [
            "minkowski", "cityblock", "cosine", "euclidean",
            "haversine", "l1", "l2", "manhattan", "nan_euclidean",
        ],
        index=0,
        help=hp_desc["metric"],
    )
    return hp


def dtc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["criterion"] = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=0,
        help=hp_desc["criterion"],
    )
    hp["splitter"] = st.selectbox(
        "Splitter",
        ["best", "random"],
        index=0,
        help=hp_desc["splitter"],
    )
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
    )
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
    )
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help=hp_desc["min_samples_leaf"],
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help=hp_desc["min_weight_fraction_leaf"],
    )
    hp["max_features"] = st.selectbox(
        "Max Features",
        [None, "sqrt", "log2"],
        index=0,
        help=hp_desc["max_features"],
    )
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=100,
        value=10,
        help=hp_desc["max_leaf_nodes"],
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
    )
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        [None, "balanced"],
        index=0,
        help=hp_desc["class_weight"],
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
    )
    return hp


def rfc_widgets(data, hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["n_estimators"] = st.slider(
        "Number of Estimators", 1, 500, 100,
        help=hp_desc["n_estimators"]
    )
    hp["criterion"] = st.selectbox(
        "Criterion", ["gini", "entropy", "log_loss"],
        help=hp_desc["criterion"]
    )
    hp["max_depth"] = none_or_widget(
        "max_depth", 1, 20, 5,
        help=hp_desc["max_depth"]
    )
    hp["min_samples_split"] = st.slider(
        "Min Samples Split", 2, 20, 2,
        help=hp_desc["min_samples_split"]
    )
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf", 1, 20, 1,
        help=hp_desc["min_samples_leaf"]
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf", 0.0, 0.5, 0.0, 0.01,
        help=hp_desc["min_weight_fraction_leaf"],
    )
    hp["max_features"] = st.selectbox(
        "Max Features", ["sqrt", "log2", None],
        help=hp_desc["max_features"]
    )
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes", 2, 100,
        help=hp_desc["max_leaf_nodes"]
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease", 0.0, 1.0, 0.0, 0.01,
        help=hp_desc["min_impurity_decrease"],
    )
    hp["bootstrap"] = st.checkbox(
        "Bootstrap", True,
        help=hp_desc["bootstrap"]
    )
    disable_oob_score = not hp["bootstrap"]
    if disable_oob_score:
        st.session_state.oob_score_checkbox = False
        st.session_state.max_samples_checkbox = False
    hp["oob_score"] = st.checkbox(
        "OOB score",
        value=False,
        help=hp_desc["oob_score"],
        disabled=disable_oob_score,
        key="oob_score_checkbox",
    )
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        [None, "balanced", "balanced_subsample"],
        help=hp_desc["class_weight"],
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0, value=0.0, step=0.01,
        help=hp_desc["ccp_alpha"],
    )
    hp["max_samples"] = none_or_widget(
        "max_samples", 1, data.shape[0], 5,
        checkbox_kwargs=dict(
            disabled=disable_oob_score,
            value=False,
            key="max_samples_checkbox"
        ),
        help=hp_desc["max_samples"]
    )
    return hp


def gbc_widgets(target, hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    if target.nunique() == 2:
        hp["loss"] = st.selectbox(
            "loss", ["log_loss", "exponential"],
            help=hp_desc["loss"]
        )
    else:
        hp["loss"] = st.selectbox(
            "loss", ["log_loss"],
            help=hp_desc["loss"]
        )
    hp["learning_rate"] = st.number_input(
        "Learning Rate", 0.0, 1.0, 0.1, 0.01,
        help=hp_desc["learning_rate"]
    )
    hp["n_estimators"] = st.slider(
        "Number of Estimators", 1, 500, 100,
        help=hp_desc["n_estimators"]
    )
    hp["subsample"] = st.number_input(
        "Subsample", 0.01, 1.0, 1.0, 0.01,
        help=hp_desc["subsample"]
    )
    hp["criterion"] = st.selectbox(
        "Criterion", ["friedman_mse", "squared_error"],
        help=hp_desc["criterion"],
    )
    hp["min_samples_split"] = st.slider(
        "Min Samples Split", 2, 500, 2,
        help=hp_desc["min_samples_split"]
    )
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf", 1, 500, 1,
        help=hp_desc["min_samples_leaf"]
    )
    hp["min_weight_fraction_leaf"] = st.number_input(
        "Min Weight Fraction Leaf", 0.0, 0.5, 0.0, 0.01,
        help=hp_desc["min_weight_fraction_leaf"],
    )
    hp["max_depth"] = st.slider(
        "Max Depth", 1, 500, 3,
        help=hp_desc["max_depth"]
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease", 0.0, 1.0, 0.0, 0.01,
        help=hp_desc["min_impurity_decrease"],
    )
    hp["init"] = none_or_widget(
        "Init", ["zero"],
        widget=st.selectbox,
        help=hp_desc["init"]
    )
    hp["max_features"] = none_or_widget(
        "max_features", ["sqrt", "log2"],
        widget=st.selectbox,
        help=hp_desc["max_features"],
    )
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes", 2, 500, 10, 1,
        help=hp_desc["max_leaf_nodes"]
    )
    hp["validation_fraction"] = st.number_input(
        "Validation Fraction", 0.01, 0.99, 0.1, 0.01,
        help=hp_desc["validation_fraction"],
    )
    hp["n_iter_no_change"] = none_or_widget(
        "n_iter_no_change", 1, 500, 10, 1,
        help=hp_desc["n_iter_no_change"]
    )
    hp["tol"] = st.number_input(
        "Tol", 0.0, 1.0, 1e-4, 1e-4,
        help=hp_desc["tol"]
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha", 0.0, 1.0, 0.0, 0.01,
        help=hp_desc["ccp_alpha"]
    )
    return hp
