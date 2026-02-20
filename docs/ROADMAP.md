# ProbaViz

---

## 🏗️ Ongoing Work

### 🧱 Project Setup & Structure
- [X] Update repo README file
- [ ] Finalize and clean `requirements.txt`
- [ ] Publish app on Streamlit Community Cloud
- [ ] Add “Open in Streamlit” badge

### 🎓 Learning & Tools
- [ ] Explore latest capabilities of  `sklearn.inspection.DecisionBoundaryDisplay`
- [ ] Re-read Streamlit documentation (best practices, deployment, genAI apps)
- [ ] Re-read VS Code documentation on AI tools and try GitHub Copilot
- [X] Familiarise with and try Codex
  - [ ] Set up and run a Cloud env to implement a feature
  - [ ] Try CLI

### 📊 Data Handling
- [X] Binary/multiclass toy datasets
- [ ] Synthetic data generation
- [ ] Train/test data split

### 📈 Metrics & Evaluation
- [X] Modify confusion/error matrix visualization
- [ ] ROC curves
- [ ] Precision–Recall curves
- [ ] Dedicated tab with common classification metrics (perhaps leveraging `skore`)

### 🎛️ Model & Interaction Features
- [ ] Reset model button
- [X] Ensure switching model resets widgets
- [X] Graceful failure in case of invalid hyper-parameter combo
- [X] Add all "standalone" scikit-learn classifiers:
  - [X] `sklearn.linear_model.LogisticRegression`
  - [X] `sklearn.linear_model.SGDClassifier`
  - [X] `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
  - [X] `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
  - [X] `sklearn.neighbors.KNeighborsClassifier`
  - [X] `sklearn.neighbors.RadiusNeighborsClassifier`
  - [X] `sklearn.neighbors.NearestCentroid`
  - [X] `sklearn.tree.DecisionTreeClassifier`
  - [X] `sklearn.tree.ExtraTreeClassifier`
  - [X] `sklearn.ensemble.RandomForestClassifier`
  - [X] `sklearn.ensemble.ExtraTreesClassifier`
  - [X] `sklearn.ensemble.GradientBoostingClassifier`
  - [X] `sklearn.ensemble.HistGradientBoostingClassifier`
  - [X] `sklearn.ensemble.AdaBoostClassifier`
  - [X] `sklearn.ensemble.BaggingClassifier`
  - [X] `sklearn.svm.SVC`
  - [X] `sklearn.svm.NuSVC`
  - [X] `sklearn.neural_network.MLPClassifier`
  - [X] `sklearn.gaussian_process.GaussianProcessClassifier`
  - [X] `sklearn.naive_bayes.GaussianNB`
  - [X] `sklearn.naive_bayes.BernoulliNB`
  - [X] `sklearn.naive_bayes.MultinomialNB`
  - [X] `sklearn.naive_bayes.ComplementNB`
  - [X] `sklearn.naive_bayes.CategoricalNB`
  - [X] `sklearn.semi_supervised.LabelPropagation`
  - [X] `sklearn.semi_supervised.LabelSpreading`
- [ ] Allow kernel customization for `GaussianProcessClassifier` 
- [ ] Allow picking an estimator for `AdaBoostClassifier` and `BaggingClassifier`
- [ ] Add meta ensemble models (voting, stacking, etc.)
- [ ] Add feature pre-processing
- [ ] Output (some) model attributes (post-learning parameters with "_" suffix)
- [ ] Add classifiers from other frameworks

### 🧩 UI / UX Improvements
- [ ] Migrate to a more interactive plotting framework like `plotly` 
- [ ] Colour picking for probability surfaces
- [ ] Add emojis/icons to model selection widget
- [ ] Dedicated help/info toggles for all data visualization tabs
- [X] Move model params out of contour plot title
- [X] Standardize widget generation patterns (model registry?)
- [X] Allow for optional selection of `random_state`
- [ ] Add deprecated badge against affected hyper-parameters
- [X] Skip probability surface plotting for `SGDClassifier` and SVMs if model configs have no `predict_proba` method, allowing to explore other tabs.
- [ ] Improve dataset / config parser (try replacing text like ":ref:User Guide <adaboost>." with actual links)
- [ ] Add dedicated model description definitions
- [ ] Balloons when 100% accuracy reached? Maybe a nice touch, but reinforces over-fitting on training data

### 🧠 GenAI Component
- [ ] LLM component reacting to user's most recent change (explain what changed / why)

### 🧹 Code Quality & Refactoring
- [ ] Add usage docstring in `viz.py` showing how to work with it in Jupyter
- [ ] Create dedicated `utils.py` module
- [X] Resolve reuse issues when switching datasets (e.g. `None → toy → None`)
- [X] Refactor toward more Pythonic code (properties, clearer APIs)
- [ ] Improve efficiency through cashing predictions/reducing conversions
- [ ] Add/update docstrings in viz.py
- [ ] Add DataViz tests
- [ ] Add app tests

---

## Notes

### ProbaViz API (refactor note)

`ProbaViz` now supports a property-driven API with lazy model fitting:

- Set state via `model`, `set_dataset(...)`, `update_params(...)`.
- Updates mark the instance as dirty; fitting occurs automatically on plotting calls.
- Use `is_dirty` / `is_fitted` to inspect lifecycle state.

### Decision: Exclude `LogisticRegressionCV`

`LogisticRegressionCV` is intentionally excluded from the core model registry.

**Rationale:**
The app is designed for interactive, user-driven exploration of hyperparameters and their effect on decision boundaries and class probability scores. Cross-validated estimators (e.g., `LogisticRegressionCV`) internally perform hyperparameter search and select values automatically, which:

* obscures which hyperparameters are actively shaping the displayed decision surface,
* introduces additional UI complexity (grids, folds, scoring),
* overlaps with the app’s core purpose of manual, visual trial-and-error tuning.

If automated tuning is added in the future, it should be introduced as a separate optional mode rather than as a standalone model in the main registry.
