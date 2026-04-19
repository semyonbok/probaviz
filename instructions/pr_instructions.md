Implement precision-recall curve visualization in the app, following the existing ROC integration pattern.

Scope:

* Add a dedicated plotting method to `ProbaViz` in `viz.py`.
* Integrate precision-recall plots into a dedicated tab in `app.py`.
* Support both binary and multiclass classification.
* For multiclass, implement one-vs-rest precision-recall curves, including micro-average and macro-average curves.
* Use the scikit-learn example below as the primary reference for the precision-recall computation approach:
  `https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html`
* Reuse the current ROC feature as the structural reference for API shape, layout, and validation style.

Implementation requirements

1. `ProbaViz` API

* Add a new plotting method in `viz.py`, analogous to `plot_roc` and consistent with `plot_matrices`.
* Follow the same conventions as the existing plotting methods where applicable:

  * similar argument structure
  * similar validation style
  * similar return style
  * similar handling of train/test splits
  * similar reuse of cached predicted probabilities
* Reuse existing project conventions and helpers where possible instead of introducing a parallel style.

2. Streamlit integration

* Add a dedicated Precision-Recall tab in `app.py`.
* In that tab:

  * left column: train-set precision-recall plots
  * right column: test-set precision-recall plots
* Each column must contain exactly two plots:

  * top: class-specific precision-recall curves
  * bottom: micro/macro precision-recall curves
* Follow the same layout and interaction style already used for the ROC tab.

3. Supported scenarios

* Binary classification:

  * class-specific plot should show the relevant one-vs-rest precision-recall curve(s) in the same style as multiclass where applicable
  * micro/macro plot should still render consistently
* Multiclass classification:

  * compute one-vs-rest precision-recall per class
  * compute micro-average precision-recall
  * compute macro-average precision-recall

4. Metric computation

* Use `precision_recall_curve` for per-class one-vs-rest curves.
* Use `average_precision_score` for average-precision values shown in legends.
* For multiclass handling, binarize `y_true` in one-vs-rest form similarly to the current ROC implementation.
* Compute micro-average precision-recall by flattening the binarized labels and predicted probabilities, following the scikit-learn example.
* Compute macro-average precision-recall with an explicit project rule so the implementation is deterministic:

  * create a common recall grid over `[0, 1]`
  * interpolate each class precision curve onto that grid
  * average interpolated precision values across classes
  * compute macro area with `auc(common_recall, mean_precision)`
* When interpolating, account for the ordering returned by `precision_recall_curve` so interpolation is performed on recall values in ascending order.

5. Plot styling
   Apply the following styling consistently to all precision-recall plots:

* main plotting axes must be square
* x-axis limits: `[-0.1, 1.1]`
* y-axis limits: `[-0.1, 1.1]`
* x-axis label: `Recall`
* y-axis label: `Precision`
* legend position: bottom left
* legend entries must include AP values

Micro/macro plot styling:

* macro curve: black, solid line
* micro curve: black, dashed line
* include the positive-class prevalence baseline using the same spirit as `plot_chance_level=True` in the scikit-learn example, adapted to the app's plotting style

Class-specific plot styling:

* one-vs-rest semantics for each class
* line colors must match the corresponding class color from `CMAP_COLORS` in `viz.py`
* markers must match the corresponding class marker from `MARKER_STYLES` in `viz.py`
* show scatter markers along each precision-recall curve, colored by probability threshold, analogous to the class-specific ROC plot styling

Use the threshold-colored marker approach below as the visual reference for class-specific curves, adapted from ROC to precision-recall:

```python
idx_ = np.argmax(model.classes_ == class_)
y_test_ = y_test == class_
y_pred_ = y_pred_proba[:, idx_]
precisions_, recalls_, threshs_ = precision_recall_curve(y_test_, y_pred_)

# precision_recall_curve returns one more precision/recall point than thresholds.
# Align threshold-colored markers to the threshold-defined points only.
display.ax_.scatter(
    recalls_[1:],
    precisions_[1:],
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
* Reuse the ROC implementation pattern where it fits, rather than inventing a separate architecture for precision-recall plots.
* Do not duplicate logic already handled elsewhere in `ProbaViz`.

Validation expectations

* Validate inputs similarly to `plot_roc` and `plot_matrices`.
* Handle unsupported or invalid cases gracefully.
* Ensure the method works for both train and test data already available in `ProbaViz`.
* Ensure the plot method requires a fitted model with probability-capable outputs, consistent with the ROC feature.

Acceptance criteria

* `ProbaViz` exposes a dedicated precision-recall plotting method.
* `app.py` contains a Precision-Recall tab with two columns: train on the left, test on the right.
* Each column contains exactly two plots: class-specific on top, micro/macro on bottom.
* Binary and multiclass classifiers both work.
* Multiclass plots include one-vs-rest, micro-average, and macro-average precision-recall curves.
* Plot styling matches the requirements above.
* Class-specific precision-recall plots use `CMAP_COLORS` and `MARKER_STYLES`.
* Legends include AP values.
* Axes are square and bounded to `[-0.1, 1.1]` for both x and y.
* The implementation reuses the current ROC feature conventions for layout, validation, and cached probability access.

Testing expectations

* Add tests in `tests/test_viz.py` analogous to the existing ROC tests.
* Cover at least the following scenarios:

  * cached probabilities are reused when the model is clean
  * invalid plotting mode raises a clear error
  * plotting without a probability-capable fitted model raises
  * binary class-mode plot renders with expected styling and limits
  * multiclass micro/macro mode renders both aggregate curves with AP labels
  * multiclass class-mode renders one curve per class with AP labels

Please first inspect the current implementations of `plot_roc` in `viz.py` and the related tab/layout code in `app.py`, then implement the feature in the same style.
