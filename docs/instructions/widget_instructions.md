## Widget generation instructions

### Context and assumptions

* Use **only** the model information provided in the prompt.
* Do **not** rely on external web access.
* Parameter order must follow the estimator’s `__init__` signature (this corresponds to the documented parameter order).
* Do not include parameters marked as deprecated or scheduled for deprecation, based on the provided model information.

### Function contract

* Implement a function with the signature:

  `make_model_widgets(model, hp_desc) -> dict[str, Any]`

* The function must return a dictionary `hp` mapping hyper-parameter names to widget values.

### Parameter inclusion rules

* Exclude system / infrastructure parameters such as:
  `n_jobs`, `verbose`, `warm_start`, `random_state`.
* Exclude parameters requiring complex objects or structures, including:
  callables, estimators, arrays, dicts, tuples, lists, or other non-scalar objects.
* Generate widgets **only** for parameters that can reasonably be expressed as:
  `None`, `bool`, `int`, `float`, or `str` (including string enums).

### Widget selection rules

* `bool` parameters → `st.checkbox`
* Parameters with discrete string choices → `st.selectbox`

  * Set the default using `index=options.index(default)` when possible.
* `int` parameters → `st.slider`
* `float` parameters → `st.number_input`
* If a parameter accepts both `int` and `float`, choose the widget based on the default value’s type:

  * int default → `st.slider`
  * float default → `st.number_input`
* If a parameter accepts `None` and numeric values, wrap the numeric widget using `none_or_widget`.

### Widget configuration rules

* Every widget **must** define an explicit `key=` argument.
* Key format:
  `f"{prefix}_{param_name}"`

  * `prefix` is the function prefix (e.g. `lr` for `lr_widgets`)
  * `param_name` is the exact sklearn hyper-parameter name.
* When using `none_or_widget`, provide:

  * `key=f"{prefix}_{param_name}"` for the underlying widget
  * `checkbox_kwargs={"key": f"{prefix}_{param_name}__is_set"}` for the checkbox.
* Always pass the default parameter value via:

  * `value=` for sliders and number inputs
  * `index=` for selectboxes.
* Use keyword arguments for all widgets (`min_value=`, `max_value=`, `step=`, etc.).
* If explicit numeric bounds are unavailable, use conservative generic bounds and document them in code comments.
* Pass `hp_desc[param_name]` to the widget’s `help=` argument.
* When using `none_or_widget`, apply the description to the widget help; keep the checkbox help as default.

### Other rules

* Collect all widget return values into a dictionary named `hp`.
* Do not attempt to validate or guard against invalid hyper-parameter combinations; this is handled elsewhere in the application.
* Do not modify function names, signatures, return types, or unrelated code.


### Examples

Example for `KNeighborsClassifier`:

```python
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
```

Example for `DecisionTreeClassifier`:
```python
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
```
