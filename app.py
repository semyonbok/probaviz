import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.pipeline import Pipeline

from src.synthetic import (
    build_synthetic_dataset,
    deserialize_synthetic_params,
    get_synthetic_labels,
    get_synthetic_spec,
)
from src.viz import ProbaViz
from src.widgets import SYNTH_WIDGETS, none_or_widget
from src.models import MODELS
from src.parsers import parse_model_desc, parse_param_desc, format_sig_md
from src.model_docs_cache import get_cached_model_docs, load_model_docs_cache


# streamlit data processing functions
def _on_dataset_change() -> None:
    """When dataset changes, reset everything derived from it"""
    for k in ["pv", "data_config", "f1", "f2"]:
        st.session_state.pop(k, None)


@st.cache_data
def process_toy(set_name):
    if set_name == "Wine (multi-class)":
        data_set = load_wine(as_frame=True)
    elif set_name == "Iris (multi-class)":
        data_set = load_iris(as_frame=True)
    elif set_name == "Cancer (binary)":
        data_set = load_breast_cancer(as_frame=True)

    target_names_map = {k: v for k, v in enumerate(data_set["target_names"])}

    return data_set["data"], data_set["target"].map(target_names_map)


@st.cache_data(max_entries=100)
def process_synth(synth_name, synth_config_json):
    return build_synthetic_dataset(synth_name, deserialize_synthetic_params(synth_config_json))


@st.cache_data
def load_cached_model_docs():
    return load_model_docs_cache()


@st.cache_data
def load_tab_explainers():
    payload = json.loads((Path(__file__).resolve().parent / "src" / "tab_explainers.json").read_text())
    return payload["tabs"]


def build_model_pipeline(model, scaling):
    if model is None or scaling == "None":
        return model

    scaler = {
        "Standardization": StandardScaler(),
        "Min-Max": MinMaxScaler(),
        "Robust": RobustScaler(),
        "Max-Abs": MaxAbsScaler(),
        "Power Transform": PowerTransformer(),
        "Quantile Transform": QuantileTransformer(),
        "Normalizer": Normalizer(),
    }[scaling]
    return Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])


def prefix_model_params(params, scaling):
    if scaling == "None":
        return params
    return {f"model__{name}": value for name, value in params.items()}


def plot_matrices(tab_conf, tab_err):
    # confusion matrices
    train_col, test_col = tab_conf.columns(2, gap="medium")
    train_col.subheader("Train Subset")
    train_col.pyplot(
        st.session_state["pv"].plot_matrices(
            return_fig=True, fig_size=(9, 16)
        )
    )
    test_col.subheader("Test Subset")
    test_col.pyplot(
        st.session_state["pv"].plot_matrices(
            return_fig=True, fig_size=(9, 16), data_split="test"
        )
    )

    # error matrices
    train_col, test_col = tab_err.columns(2, gap="medium")
    train_col.subheader("Train Subset")
    train_col.pyplot(
        st.session_state["pv"].plot_matrices(
            return_fig=True, mode="error", fig_size=(9, 16)
        )
    )
    test_col.subheader("Test Subset")
    test_col.pyplot(
        st.session_state["pv"].plot_matrices(
            return_fig=True, mode="error", fig_size=(9, 16), data_split="test"
        )
    )


def plot_rocs(tab_roc):
    roc_explainer = load_tab_explainers()["roc_curves"]
    train_col, test_col = tab_roc.columns(2, gap="medium")

    train_col.subheader("Train Subset")
    train_col.pyplot(
        st.session_state["pv"].plot_roc(
            return_fig=True, fig_size=(9, 9), mode="class"
        )
    )
    train_col.pyplot(
        st.session_state["pv"].plot_roc(
            return_fig=True, fig_size=(9, 9), mode="micro_macro"
        )
    )

    test_col.subheader("Test Subset")
    test_col.pyplot(
        st.session_state["pv"].plot_roc(
            return_fig=True, fig_size=(9, 9), data_split="test", mode="class"
        )
    )
    test_col.pyplot(
        st.session_state["pv"].plot_roc(
            return_fig=True, fig_size=(9, 9), data_split="test", mode="micro_macro"
        )
    )

    with tab_roc.expander(roc_explainer["title"], icon=roc_explainer["icon"]):
        st.info(
            roc_explainer["body_markdown"]
        )


def plot_prs(tab_pr):
    pr_explainer = load_tab_explainers()["precision_recall_curves"]
    train_col, test_col = tab_pr.columns(2, gap="medium")

    train_col.subheader("Train Subset")
    train_col.pyplot(
        st.session_state["pv"].plot_pr(
            return_fig=True, fig_size=(9, 9), mode="class"
        )
    )
    train_col.pyplot(
        st.session_state["pv"].plot_pr(
            return_fig=True, fig_size=(9, 9), mode="micro_macro"
        )
    )

    test_col.subheader("Test Subset")
    test_col.pyplot(
        st.session_state["pv"].plot_pr(
            return_fig=True, fig_size=(9, 9), data_split="test", mode="class"
        )
    )
    test_col.pyplot(
        st.session_state["pv"].plot_pr(
            return_fig=True, fig_size=(9, 9), data_split="test", mode="micro_macro"
        )
    )

    with tab_pr.expander(pr_explainer["title"], icon=pr_explainer["icon"]):
        st.info(
            pr_explainer["body_markdown"]
        )


# main display space
st.set_page_config(layout='wide')
st.header("Welcome to ProbaViz")

data = None
target = None
set_name = None
display_set_name = None
f1 = 0
f2 = 1
train_size = None
split_random_state = None
data_config = None
synth_summary: str | None = None

# side bar controls: data, model, and hyperparameters
with st.sidebar:
    st.subheader(
        "Data",
        help=(
            "Pick a dataset and two of its numerical features (columns) "
            "that will be used for model training. Currently, three "
            "['toy' datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) "
            "are available: wine, iris, and breast cancer."
        )
    )

    dataset = st.radio("Select a Dataset", ["Toy", "Synthetic"])
    if dataset == "Toy":
        set_name = st.selectbox(
            "Select a Toy Dataset",
            [None, "Wine (multi-class)", "Iris (multi-class)", "Cancer (binary)"],
            on_change=_on_dataset_change, key="set_name"
        )

        # once set is chosen, process data and allow to pick features
        if set_name is not None:
            data, target = process_toy(set_name)
            display_set_name = set_name
            f1 = st.selectbox(
                "Pick Feature 1 (X-axis)",
                data.columns, key=f"{set_name}_f1"
            )
            f2 = st.selectbox(
                "Pick Feature 2 (Y-axis)",
                data.columns[data.columns != f1], key=f"{set_name}_f2"
            )

    elif dataset == "Synthetic":
        synth_name = st.selectbox(
            "Select a Data Synthesis Method",
            [None, *get_synthetic_labels()],
            on_change=_on_dataset_change, key="synth_name"
        )
        if synth_name is not None:
            synth_spec = get_synthetic_spec(synth_name)
            st.caption(synth_spec.help_text)
            with st.container(border=True):
                synth_params = SYNTH_WIDGETS[synth_name](synth_spec)
            synth_config_json = json.dumps(synth_params, sort_keys=True)
            data, target = process_synth(synth_name, synth_config_json)
            target_bytes = pd.Series(target).to_numpy().tobytes()
            set_name = sha256(
                data.values.tobytes() + target_bytes + synth_config_json.encode()
            ).hexdigest()
            display_set_name = f"Synthetic: {synth_name}"
            f1, f2 = 0, 1

    if set_name is not None:
        with st.container(border=True):
            split_random_state = st.number_input(
                "Data Split Random State", 0, 999999, 42, 1,
                key=f"{set_name}_split_random_state",
                help=(
                    "Controls the shuffling applied to the data before applying the split.  \n"
                    "Please choose an integer for reproducible output."
                )
            )
            train_size = st.number_input(
                "Pick Train Size", 0.5, 0.9, 0.75, 0.05,
                help="Controls the fraction of data allocated for training."
            )
            train_col, test_col = st.columns(2)
        data_config = (set_name, f1, f2, train_size, split_random_state)

    st.divider()
    st.subheader("Pipeline")
    scaling = st.selectbox(
        "Select Feature Scaling",
        [
            "None",
            "Standardization",
            "Min-Max",
            "Robust",
            "Max-Abs",
            "Power Transform",
            "Quantile Transform",
            "Normalizer",
        ],
        help=(
            "Optionally apply a preprocessing transform before fitting the model. "
            "When enabled, the model is wrapped in a pipeline. "
            "For more information, please refer to "
            "[API Reference](https://scikit-learn.org/stable/api/sklearn.preprocessing.html)"
        ),
    )
    model_pick = st.selectbox("Select a Model", [None, *sorted(MODELS.keys())])
    model = MODELS[model_pick].factory() if model_pick else None

    st.divider()
    st.subheader("Hyper-parameters")
    hp = {}

    # set `random_state` if the model has this parameter
    if (model is not None) and (set_name is not None):
        cached_docs = (
            get_cached_model_docs(model_pick, payload=load_cached_model_docs())
            if model_pick is not None else None
        )
        if cached_docs is not None:
            model_desc, hp_desc = cached_docs
        else:
            hp_desc = parse_param_desc(model)
        if "random_state" in model.get_params().keys():
            hp["random_state"] = none_or_widget(
                "Random State", 0, 999999, 1, 1,
                widget=st.number_input,
                help=hp_desc["random_state"]
            )
        # model_pick cannot be None under this condition
        hp = {**hp, **MODELS[model_pick].widgets(hp_desc=hp_desc)}  # type: ignore
    else:
        st.caption("Select dataset and model to configure hyper-parameters.")

# Session State and Plotting Logic
# If data is None, don't plot anything
# If data is not None but model is None, plot blank scatter
# if data and model are not None, plot contour
if set_name is None:
    st.session_state.pop("pv", None)
    st.session_state.pop("data_config", None)
    st.stop()

if "data_config" not in st.session_state:
    st.session_state["data_config"] = data_config

# call `set_dataset` only when there is change in... data!
if "pv" not in st.session_state:
    st.session_state["pv"] = ProbaViz(
        model=model,
        data=data,
        target=target,
        train_size=train_size,
        split_random_state=split_random_state,
        features=[f1, f2],
    )
elif st.session_state["data_config"] != data_config:
    st.session_state["data_config"] = data_config
    st.session_state["pv"].set_dataset(
        data,
        target,
        [f1, f2],
        train_size=train_size,
        split_random_state=split_random_state,
    )

train_col.caption(
    (
        "Train subset:  \n"
        f" {len(st.session_state['pv']._train_target)} samples"
    )
)
test_col.caption(
    (
        "Test subset:  \n"
        f" {len(st.session_state['pv']._test_target)} samples"
    )
)

if model is None:
    st.pyplot(
        st.session_state["pv"].plot(
            contour_on=False, return_fig=True, fig_size=(16, 9)
        )
    )
else:
    active_model = build_model_pipeline(model, scaling)
    try:
        st.session_state["pv"].model = active_model
        st.session_state["pv"].update_params(**prefix_model_params(hp, scaling))

        tab_contour, tab_conf, tab_err, tab_roc, tab_pr = st.tabs(
            ["Decision Boundary", "Confusion Matrices", "Error Matrices", "ROC Curves", "PR Curves"]
        )
        tab_contour.pyplot(
            st.session_state["pv"].plot(
                contour_on=True, return_fig=True, fig_size=(16, 9)
            )
        )
        plot_matrices(tab_conf, tab_err)
        plot_rocs(tab_roc)
        plot_prs(tab_pr)
    except AttributeError:
        tab_contour.error(
            "❌ **This model configuration cannot predict probability scores.** "
            "Try changing hyper-parameters (e.g., Loss, Metric or Probability "
            "Estimates for support vector machines) or refer to documentation."
        )
        plot_matrices(tab_conf, tab_err)
        tab_roc.error(
            "❌ **ROC curves are unavailable for this model configuration.** "
            "This view requires `predict_proba` support."
        )
        tab_pr.error(
            "❌ **Precision-recall curves are unavailable for this model configuration.** "
            "This view requires `predict_proba` support."
        )
    except (ValueError, NotImplementedError) as e:
        st.error(f"❌ **Model failed to fit.** {e}")
    finally:
        with st.expander("Model Info", icon="ℹ️"):
            if model_pick is not None:
                model_sig = format_sig_md(active_model)
                if cached_docs is not None:
                    st.info(model_sig + model_desc)
                else:
                    st.info(model_sig + parse_model_desc(model))
