import re
import streamlit as st

from sklearn.datasets import load_iris, load_wine

from src.viz import ProbaViz
from src.widgets import none_or_widget
from src.models import MODELS


# streamlit data processing functions
def _on_dataset_change() -> None:
    """When dataset changes, reset everything derived from it"""
    for k in ["pv", "set_and_features", "f1", "f2"]:
        st.session_state.pop(k, None)


@st.cache_data
def process_toy(set_name):
    if set_name == "Wine":
        data_set = load_wine(as_frame=True)
    elif set_name == "Iris":
        data_set = load_iris(as_frame=True)

    target_names_map = {k: v for k, v in enumerate(data_set["target_names"])}

    return data_set["data"], data_set["target"].map(target_names_map)


# parsers
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

        # once set is chosen, process data and allow to pick features
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
    model_pick = st.selectbox("Select a Model", [None, *MODELS.keys()])
    model = MODELS[model_pick].factory() if model_pick else None

    # set `random_state` if the model has this parameter
    if (model is not None) and (set_name is not None):
        hp = {}
        hp_desc = parse_param_desc(model)
        if "random_state" in model.get_params().keys():
            hp["random_state"] = none_or_widget(
                "Random State", 0, 999999, 1, 1,
                widget=st.number_input,
                help=hp_desc["random_state"]
            )
        # model_pick cannot be None under this condition
        hp = {**hp, **MODELS[model_pick].widgets(hp_desc=hp_desc)}  # type: ignore

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
    except (ValueError, NotImplementedError) as e:
        st.error(f"❌ **Model failed to fit.** {e}")
        st.stop()
    except AttributeError:
        st.error(
            "❌ **This model configuration cannot predict probability scores.** "
            "Try changing hyper-parameters (e.g., Loss, Metric or Probability "
            "Estimates for support vector machines) or refer to documentation."
        )
        st.stop()
