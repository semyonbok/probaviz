# ProbaViz

## ProbaViz API (refactor note)

`ProbaViz` now supports a property-driven API with lazy model fitting:

- Set state via `model`, `set_dataset(...)`, `update_params(...)`.
- Updates mark the instance as dirty; fitting occurs automatically on plotting calls.
- Use `is_dirty` / `is_fitted` to inspect lifecycle state.
- `train_size` is configurable via constructor, `set_dataset(...)`, and the `train_size` property; `None` delegates to sklearn defaults.
- Splitting is strictly stratified; impossible class/sample configurations raise a targeted error rather than silently falling back.

## Decisions 

### Exclude `LogisticRegressionCV`

`LogisticRegressionCV` is intentionally excluded from the core model registry.

**Rationale:**
The app is designed for interactive, user-driven exploration of hyperparameters and their effect on decision boundaries and class probability scores. Cross-validated estimators (e.g., `LogisticRegressionCV`) internally perform hyperparameter search and select values automatically, which:

* obscures which hyperparameters are actively shaping the displayed decision surface,
* introduces additional UI complexity (grids, folds, scoring),
* overlaps with the app’s core purpose of manual, visual trial-and-error tuning.

If automated tuning is added in the future, it should be introduced as a separate optional mode rather than as a standalone model in the main registry.

### Matrix Arrangement Choice
Allocating a column for train/test subsets and arranging the matrices vertically makes it a bit text-heavy but:
* removing y label and y tick labels from the test matrices will resize them
* looking at the tab through a mobile app should place train/test matrices one by one, actually helping with visualization

### Allow for invalid Hyper-parameter Combinations
* used to be a significant roadblock
* would have prevented an important educational component - errors
* instead, added graceful failures, hinting at what can be toggled to fix the errors
