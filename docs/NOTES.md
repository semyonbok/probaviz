# ProbaViz

## ProbaViz API (refactor note)

`ProbaViz` now supports a property-driven API with lazy model fitting:

- Set state via `model`, `set_dataset(...)`, `update_params(...)`.
- Updates mark the instance as dirty; fitting occurs automatically on plotting calls.
- Use `is_dirty` / `is_fitted` to inspect lifecycle state.
- `train_size` is configurable via constructor, `set_dataset(...)`, and the `train_size` property; `None` delegates to sklearn defaults.
- Splitting is strictly stratified; impossible class/sample configurations raise a targeted error rather than silently falling back.

## ProbaCoach Feature (Pre-planning)
**Main idea:** supply llm with a payload on current app state and get a response with a few actionable tips, things to try, encouragement and maybe even a joke.
### Overall Design
- cycle through models if allowance on one is exceeded
- lean on input token caching: https://console.groq.com/docs/prompt-caching
- limit max output token to 512
- always just 2 messages in chat completion: instructions from system and payload from user
- payload should include:
    - minimal info on dataset and pre-processing selected
    - info on model: exploit `load_cached_model_docs` from `model_docs_cache.py` to get model and hyper-param descriptions + `get_params()` to get current hyperparams values
    - metrics: exploit `ProbaViz` method: `get_classification_metrics` for both train and test subsets
- UI
    - Press button -> get chat response from assistant in `st.chat_message`
    - indicate what model was used and number of toggles left
    - optionally, indicate (some of) remaining allowances
    - place coach at the bottom of the main space
### Devs
- analyze token use for all 26 models
- try passing invalid API key
### Safeguards
- limit number of coach toggles per session
- warning about genai output
- ensure easy revert
### Production
- start with a  free tier of groq
- crete a separate API key and manage as a streamlit secret
- update env: add groq, add API key as a secret

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
