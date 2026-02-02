## Widget key conventions

### Required keys for all widgets

* **Every Streamlit widget call must include an explicit `key=` argument.**
* Keys must be **unique strings across the entire app**.
* The key format is:

  `key = f"{prefix}_{param_name}"`

  where:

  * `prefix` is the **function prefix** (e.g. `"lr"` for `lr_widgets`, `"dtc"` for `dtc_widgets`)
  * `param_name` is the **exact sklearn hyper-parameter name** used as the dictionary key in `hp[...]`

    * Use the parameter name (e.g. `"C"`, `"max_iter"`, `"shrink_threshold"`), **not** the widget label.

### Required keys for `none_or_widget`

When a hyper-parameter is implemented using `none_or_widget(...)`, you must provide **two keys**:

1. **Underlying widget key** (passed as `key=` to `none_or_widget` so it is forwarded to the underlying Streamlit widget):

* `key = f"{prefix}_{param_name}"`

2. **Checkbox key** (passed via `checkbox_kwargs` so it is forwarded to `st.checkbox`):

* `checkbox_kwargs={"key": f"{prefix}_{param_name}__is_set"}`

### Examples

```python
def lr_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}

    hp["C"] = st.number_input(
        "Inverse of Regularization Strength (C)",
        min_value=0.0001,
        value=1.0,
        step=0.1,
        help=hp_desc["C"],
        key="lr_C",
    )
    return hp
```

```python
def nc_widgets(hp_desc: dict[str, str]) -> dict:
    hp: dict[str, Any] = {}

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
    return hp
```

### Notes

* This instruction assumes `none_or_widget` accepts `checkbox_kwargs` and forwards it to `st.checkbox`, while forwarding `key=` (and other widget kwargs) to the underlying widget.
* Do not invent alternative key formats; always use the standard patterns above.

---
