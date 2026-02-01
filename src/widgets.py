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


def nc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["metric"] = st.selectbox(
        "Metric",
        ["euclidean", "manhattan"],
        index=0,
        help=hp_desc["metric"],
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
    )
    hp["priors"] = st.selectbox(
        "Priors",
        ["uniform", "empirical"],
        index=0,
        help=hp_desc["priors"],
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
    # Conservative bounds: estimator expects positive p values.
    hp["p"] = st.slider(
        "Power",
        min_value=1,
        max_value=10,
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


def etc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    hp["criterion"] = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=0,
        help=hp_desc["criterion"],
    )
    hp["splitter"] = st.selectbox(
        "Splitter",
        ["random", "best"],
        index=0,
        help=hp_desc["splitter"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
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
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index("sqrt"),
        help=hp_desc["max_features"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
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
    class_weight_options = [None, "balanced"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
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


def rfc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=100,
        help=hp_desc["n_estimators"],
    )
    criterion_options = ["gini", "entropy", "log_loss"]
    hp["criterion"] = st.selectbox(
        "Criterion",
        criterion_options,
        index=criterion_options.index("gini"),
        help=hp_desc["criterion"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
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
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index("sqrt"),
        help=hp_desc["max_features"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
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
    hp["bootstrap"] = st.checkbox(
        "Bootstrap",
        value=True,
        help=hp_desc["bootstrap"],
    )
    hp["oob_score"] = st.checkbox(
        "OOB Score",
        value=False,
        help=hp_desc["oob_score"],
    )
    class_weight_options = [None, "balanced", "balanced_subsample"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
        help=hp_desc["class_weight"],
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
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
    )
    hp["criterion"] = st.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=0,
        help=hp_desc["criterion"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = none_or_widget(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        help=hp_desc["max_depth"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help=hp_desc["min_samples_split"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
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
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index("sqrt"),
        help=hp_desc["max_features"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
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
    hp["bootstrap"] = st.checkbox(
        "Bootstrap",
        value=False,
        help=hp_desc["bootstrap"],
    )
    hp["oob_score"] = st.checkbox(
        "OOB Score",
        value=False,
        help=hp_desc["oob_score"],
    )
    class_weight_options = [None, "balanced", "balanced_subsample"]
    hp["class_weight"] = st.selectbox(
        "Class Weight",
        class_weight_options,
        index=class_weight_options.index(None),
        help=hp_desc["class_weight"],
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
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
    )
    hp["learning_rate"] = st.number_input(
        "Learning Rate",
        min_value=0.0,
        value=0.1,
        step=0.01,
        help=hp_desc["learning_rate"],
    )
    # Conservative bounds: estimator does not define explicit limits for n_estimators.
    hp["n_estimators"] = st.slider(
        "Number of Estimators",
        min_value=1,
        max_value=500,
        value=100,
        help=hp_desc["n_estimators"],
    )
    hp["subsample"] = st.number_input(
        "Subsample",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help=hp_desc["subsample"],
    )
    criterion_options = ["friedman_mse", "squared_error"]
    hp["criterion"] = st.selectbox(
        "Criterion",
        criterion_options,
        index=criterion_options.index("friedman_mse"),
        help=hp_desc["criterion"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_split.
    hp["min_samples_split"] = st.slider(
        "Min Samples Split",
        min_value=2,
        max_value=200,
        value=2,
        help=hp_desc["min_samples_split"],
    )
    # Conservative bounds: estimator does not define explicit limits for min_samples_leaf.
    hp["min_samples_leaf"] = st.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=200,
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
    # Conservative bounds: estimator does not define explicit limits for max_depth.
    hp["max_depth"] = st.slider(
        "Max Depth",
        min_value=1,
        max_value=20,
        value=3,
        help=hp_desc["max_depth"],
    )
    hp["min_impurity_decrease"] = st.number_input(
        "Min Impurity Decrease",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["min_impurity_decrease"],
    )
    max_features_options = [None, "sqrt", "log2"]
    hp["max_features"] = st.selectbox(
        "Max Features",
        max_features_options,
        index=max_features_options.index(None),
        help=hp_desc["max_features"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_leaf_nodes.
    hp["max_leaf_nodes"] = none_or_widget(
        "max_leaf_nodes",
        min_value=2,
        max_value=200,
        value=10,
        help=hp_desc["max_leaf_nodes"],
    )
    hp["validation_fraction"] = st.number_input(
        "Validation Fraction",
        min_value=0.01,
        max_value=0.99,
        value=0.1,
        step=0.01,
        help=hp_desc["validation_fraction"],
    )
    # Conservative bounds: estimator does not define explicit limits for n_iter_no_change.
    hp["n_iter_no_change"] = none_or_widget(
        "n_iter_no_change",
        min_value=1,
        max_value=200,
        value=10,
        help=hp_desc["n_iter_no_change"],
    )
    hp["tol"] = st.number_input(
        "Tolerance",
        min_value=0.0,
        value=0.0001,
        step=0.0001,
        format="%.2e",
        help=hp_desc["tol"],
    )
    hp["ccp_alpha"] = st.number_input(
        "CCP Alpha",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help=hp_desc["ccp_alpha"],
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
    )
    penalty_options = ["l2", "l1", "elasticnet", None]
    hp["penalty"] = st.selectbox(
        "Penalty",
        penalty_options,
        index=penalty_options.index("l2"),
        help=hp_desc["penalty"],
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
    )
    hp["l1_ratio"] = st.number_input(
        "l1_ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help=hp_desc["l1_ratio"],
    )
    hp["fit_intercept"] = st.checkbox(
        "Fit Intercept",
        value=True,
        help=hp_desc["fit_intercept"],
    )
    # Conservative bounds: estimator does not define explicit limits for max_iter.
    hp["max_iter"] = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=5000,
        value=1000,
        step=50,
        help=hp_desc["max_iter"],
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
    )
    hp["shuffle"] = st.checkbox(
        "Shuffle",
        value=True,
        help=hp_desc["shuffle"],
    )
    # Conservative bounds: estimator does not define explicit limits for epsilon.
    hp["epsilon"] = st.number_input(
        "Epsilon",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help=hp_desc["epsilon"],
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
    )
    # Conservative bounds: estimator does not define explicit limits for eta0.
    hp["eta0"] = st.number_input(
        "Eta0",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        step=0.01,
        help=hp_desc["eta0"],
    )
    # Conservative bounds: estimator does not define explicit limits for power_t.
    hp["power_t"] = st.number_input(
        "Power T",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=hp_desc["power_t"],
    )
    hp["early_stopping"] = st.checkbox(
        "Early Stopping",
        value=False,
        help=hp_desc["early_stopping"],
    )
    hp["validation_fraction"] = st.number_input(
        "Validation Fraction",
        min_value=0.01,
        max_value=0.99,
        value=0.1,
        step=0.01,
        help=hp_desc["validation_fraction"],
    )
    # Conservative bounds: estimator does not define explicit limits for n_iter_no_change.
    hp["n_iter_no_change"] = st.slider(
        "N Iterations No Change",
        min_value=1,
        max_value=200,
        value=5,
        step=1,
        help=hp_desc["n_iter_no_change"],
    )
    hp["average"] = st.checkbox(
        "Average",
        value=False,
        help=hp_desc["average"],
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
    )
    shrinkage_mode_options = [None, "auto", "float"]
    shrinkage_mode = st.selectbox(
        "Shrinkage Mode",
        shrinkage_mode_options,
        index=shrinkage_mode_options.index(None),
        help=hp_desc["shrinkage"],
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
    )
    hp["store_covariance"] = st.checkbox(
        "Store Covariance",
        value=False,
        help=hp_desc["store_covariance"],
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
    return hp


def qda_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}
    solver_options = ["svd", "eigen"]
    hp["solver"] = st.selectbox(
        "Solver",
        solver_options,
        index=solver_options.index("svd"),
        help=hp_desc["solver"],
    )
    shrinkage_mode_options = [None, "auto", "float"]
    shrinkage_mode = st.selectbox(
        "Shrinkage Mode",
        shrinkage_mode_options,
        index=shrinkage_mode_options.index(None),
        help=hp_desc["shrinkage"],
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
    )
    hp["store_covariance"] = st.checkbox(
        "Store Covariance",
        value=False,
        help=hp_desc["store_covariance"],
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
    return hp
