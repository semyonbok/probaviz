import re
import streamlit as st

from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.viz import ProbaViz


# streamlit helpers
def _on_dataset_change() -> None:
    # When dataset changes (incl. to None), reset everything derived from it
    for k in ["pv", "set_and_features"]:
        st.session_state.pop(k, None)

    for k in ["f1", "f2"]:
        st.session_state.pop(k, None)


# data processing functions
@st.cache_data
def process_toy(set_name):
    if set_name == "Wine":
        data_set = load_wine(as_frame=True)
    elif set_name == "Iris":
        data_set = load_iris(as_frame=True)
    target_names_map = {
        k: v
        for k, v in zip(
            range(data_set["target"].nunique()), data_set["target_names"]
        )
    }
    data_set["target"] = data_set["target"].map(target_names_map)
    return data_set["data"], data_set["target"]


def fetch_model(model_pick):
    """Needed to avoid terminal error caused by model selection checkbox"""
    if model_pick == "Logistic Regression":
        return LogisticRegression()
    if model_pick == "K Nearest Neighbors":
        return KNeighborsClassifier()
    if model_pick == "Decision Tree":
        return DecisionTreeClassifier()
    if model_pick == "Random Forest":
        return RandomForestClassifier()
    if model_pick == "Gradient Boosting":
        return GradientBoostingClassifier()


def parse_param_desc(model):
    params = model.get_params().keys()
    params = "|".join([p + " : " for p in params])

    params_desc = re.split(params, model.__doc__)[1:]
    params_desc[-1] = params_desc[-1].split("Attributes\n")[0]
    params_desc = {
        k[:-3]: "\n".join(v.split("\n\n"))
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

    return f"```python\n{repr(model)}\n``` \n\n{desc}"


# routine to pick a default sklearn model
all_models = [
    None, "Logistic Regression", "K Nearest Neighbors", "Decision Tree",
    "Random Forest", "Gradient Boosting"
]

# main display space
st.set_page_config(layout='wide')
st.header("Welcome to ProbaViz")

# side bar controls: data, model, plot aesthetics
with st.sidebar:
    # data (only toy datasets for now)
    st.subheader(
        "Dataset",
        help=(
            "Pick a dataset and two of its numerical features (columns) "
            "that will be used for model training. Currently, only two "
            "'toy' datasets are available: wine and iris "
            "(https://scikit-learn.org/stable/datasets/toy_dataset.html)."
        )
    )
    if st.checkbox("Synthetic Dataset", False, disabled=True):
        pass

    if st.checkbox("Toy Dataset", True, disabled=True):
        set_name = st.selectbox(
            "Select a Toy Dataset", [None, "Wine", "Iris"],
            on_change=_on_dataset_change, key="set_name"
        )

        # once set is chosen, process data and ozffer to pick feature 1 & 2
        if set_name is not None:
            data, target = process_toy(set_name)
            st.write("Pick Features:")
            f1 = st.selectbox(
                "X-axis", data.columns, key="f1"
            )
            f2 = st.selectbox(
                "Y-axis", data.columns[data.columns != f1], key="f2"
            )

    st.subheader("Classifier")
    model_pick = st.selectbox("Select a Model", all_models)
    model = fetch_model(model_pick)

    # set `random_state` if the model has this parameter
    if (model is not None) and (set_name is not None):
        hp = {}
        hp_desc = parse_param_desc(model)
        if "random_state" in model.get_params().keys():
            hp["random_state"] = st.number_input(
                "Input Random State", 0, 500, 1, 1,
                help=hp_desc["random_state"]
            )

        if isinstance(model, LogisticRegression):
            from src.widgets import lr_widgets
            hp = {**hp, **lr_widgets(hp_desc)}

        if isinstance(model, KNeighborsClassifier):
            from src.widgets import knc_widgets
            hp = {**hp, **knc_widgets(hp_desc)}

        if isinstance(model, DecisionTreeClassifier):
            from src.widgets import dtc_widgets
            hp = {**hp, **dtc_widgets(hp_desc)}

        if isinstance(model, RandomForestClassifier):
            from src.widgets import rfc_widgets
            hp = {**hp, **rfc_widgets(data, hp_desc)}

        if isinstance(model, GradientBoostingClassifier):
            from src.widgets import gbc_widgets
            hp = {**hp, **gbc_widgets(target, hp_desc)}


# Session State and Plotting Logic
# If data is None, don't plot anything
# If data is not None but model is None, plot blank scatter
# if data and model are not None, plot contour
if set_name is None:
    st.session_state.pop("pv", None)
    st.session_state.pop("set_and_features", None)
    st.stop()

if "set_and_features" not in st.session_state:
    st.session_state["set_and_features"] = (set_name, f1, f2)

# call `set_data` only when there is change in... data!
if "pv" not in st.session_state:
    st.session_state["pv"] = ProbaViz(model, data, target, [f1, f2])
elif st.session_state["set_and_features"] != (set_name, f1, f2):
    st.session_state["set_and_features"] = (set_name, f1, f2)
    st.session_state["pv"].set_data(data, target, [f1, f2])

if model is None:
    st.pyplot(
        st.session_state["pv"].plot(
            contour_on=False, return_fig=True, fig_size=(16, 9)
        )
    )
else:
    try:
        st.session_state["pv"].set_model(model.set_params(**hp))

        tab_contour, tab_conf, tab_err = st.tabs(
            ["Decision Boundary", "Confusion Matrices", "Error Matrices"]
        )
        tab_contour.pyplot(
            st.session_state["pv"].plot(
                contour_on=True, return_fig=True, fig_size=(16, 9)
            )
        )
        tab_conf.pyplot(
            st.session_state["pv"].plot_confusion_matrices(
                return_fig=True, fig_size=(16, 9)
            )
        )
        tab_err.pyplot(
            st.session_state["pv"].plot_error_matrices(
                return_fig=True, fig_size=(16, 9)
            )
        )
        with st.expander("Model Info", icon="ℹ️"):
            st.info(parse_model_desc(model))
    except ValueError as e:
        st.error(f"❌ **Model failed to fit.** {e}")
        st.stop()
