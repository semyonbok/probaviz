Implement ROC curve visualization in the app, following the existing confusion-matrix integration pattern.

Scope:

* Add a dedicated plotting method to `ProbaViz` in `viz.py`.
* Integrate ROC plots into a dedicated tab in `app.py`.
* Support both binary and multiclass classification.
* For multiclass, implement one-vs-rest ROC curves, including micro-average and macro-average curves.
* Use the scikit-learn example below as the reference for the ROC computation approach:
  `https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html`

Implementation requirements

1. `ProbaViz` API

* Add a new plotting method in `viz.py`, analogous to `plot_matrices`.
* Follow the same conventions as `plot_matrices` where applicable:

  * similar argument structure
  * similar validation style
  * similar return style
  * similar handling of train/test splits
* Reuse existing project conventions and helpers where possible instead of introducing a parallel style.

2. Streamlit integration

* Add a dedicated ROC tab in `app.py`.
* In that tab:

  * left column: train-set ROC plots
  * right column: test-set ROC plots
* Each column must contain exactly two plots:

  * top: micro/macro ROC curves
  * bottom: class-specific ROC curves

3. Supported scenarios

* Binary classification:

  * class-specific ROC plot should show the relevant one-vs-rest ROC curve(s) in the same style as multiclass where applicable
  * micro/macro plot should still render consistently if supported by the underlying computation
* Multiclass classification:

  * compute one-vs-rest ROC per class
  * compute micro-average ROC
  * compute macro-average ROC

4. Plot styling
   Apply the following styling consistently to all ROC plots:

* main plotting axes must be square
* x-axis limits: `[-0.1, 1.1]`
* y-axis limits: `[-0.1, 1.1]`
* legend position: bottom right
* legend entries must include AUC values

Micro/macro plot styling:

* macro curve: black, solid line
* micro curve: black, dashed line

Class-specific plot styling:

* one-vs-rest semantics for each class
* line colors must match the corresponding class color from `CMAP_COLORS` in `viz.py`
* markers must match the corresponding class marker from `MARKER_STYLES` in `viz.py`
* also show scatter markers along each ROC curve, colored by probability threshold

Use the threshold-colored marker approach shown below for class-specific curves:

```python
idx_ = np.argmax(model.classes_ == class_)
y_test_ = y_test == class_
y_pred_ = y_pred_proba[:, idx_]
fprs_, tprs_, threshs_ = roc_curve(y_test_, y_pred_)
display = RocCurveDisplay(fpr=fprs_, tpr=tprs_)
display.plot(curve_kwargs={"zorder": 1})
display.ax_.set_xlim(-0.09, 1.09)
display.ax_.set_ylim(-0.09, 1.09)
display.ax_.scatter(
    fprs_, tprs_,
    c=threshs_,
    zorder=2,
    cmap=cmap_,
    edgecolor="k",
    vmin=0,
    vmax=1,
)
```

Project constraints

* Preserve the current app structure and coding style.
* Prefer a small, maintainable change over a broad refactor.
* Reuse existing constants and plotting conventions from `viz.py`.
* Do not duplicate logic already handled elsewhere in `ProbaViz`.

Validation expectations

* Validate inputs similarly to `plot_matrices`.
* Handle unsupported or invalid cases gracefully.
* Ensure the method works for both train and test data already available in `ProbaViz`.

Acceptance criteria

* `ProbaViz` exposes a dedicated ROC plotting method.
* `app.py` contains a ROC tab with two columns: train on the left, test on the right.
* Each column contains exactly two plots: micro/macro on top, class-specific on bottom.
* Binary and multiclass classifiers both work.
* Multiclass plots include one-vs-rest, micro-average, and macro-average ROC curves.
* Plot styling matches the requirements above.
* Class-specific ROC plots use `CMAP_COLORS` and `MARKER_STYLES`.
* Legends include AUC values.
* Axes are square and bounded to `[-0.1, 1.1]` for both x and y.

Please first inspect the current implementations of `plot_matrices` in `viz.py` and the related tab/layout code in `app.py`, then implement the feature in the same style.
