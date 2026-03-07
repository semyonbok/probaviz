# ProbaViz

## 🏗️ Ongoing Work

### 🧱 Project Setup & Structure
- [X] Update repo README file
- [ ] Finalize and clean `requirements.txt`
- [ ] ⭐ Publish app on Streamlit Community Cloud
- [ ] ⭐ Add “Open in Streamlit” badge

### 🎓 Learning & Tools
- [ ] Explore latest capabilities of  `sklearn.inspection.DecisionBoundaryDisplay`
- [ ] Deep-dive into sklearn classification metrics (skolar, user docs)
- [ ] Re-read Streamlit documentation (best practices, deployment, genAI apps)
- [ ] Re-read Groq Documentation
- [X] Familiarise with and try Codex
  - [ ] Set up and run a Cloud env to implement a feature
  - [ ] Try CLI

### 📊 Data Handling
- [X] Binary/multiclass toy datasets
- [ ] ⭐ Synthetic data generation
- [X] Train/test data split
- [ ] Decide on whether cashing datasets is needed at all

### 📈 Metrics & Evaluation
- [X] Modify confusion/error matrix visualization
- [ ] ⭐ ROC curves
- [ ] ⭐ Precision–Recall curves
- [ ] ⭐ Dedicated tab with common classification metrics (perhaps leveraging `skore.EstimatorReport`)
- [ ] Calibration display for binary classification
- [ ] Learning curve

### 🎛️ Model & Interaction Features
- [ ] ⭐ Reset model button
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
- [ ] ⭐ Dedicated help/info toggles for all data visualization tabs
- [X] Move model params out of contour plot title
- [X] Standardize widget generation patterns (model registry?)
- [X] Allow for optional selection of `random_state`
- [ ] Add deprecated badge against affected hyper-parameters
- [X] Skip probability surface plotting for `SGDClassifier` and SVMs if model configs have no `predict_proba` method, allowing to explore other tabs.
- [ ] Improve model / parameter parser
  - [ ] replace restructured text with markdown links (there are also arxiv and doi roles):
    - [X] ref
    - [X] term
    - [X] class
    - [X] meth
    - [X] obj
    - [X] mod
    - [X] func
  - [X] replace directives with markdown:
    - [X] ('versionadded', 71)
    - [X] ('versionchanged', 35)
    - [X] ('following', 5)
    - [X] ('deprecated', 4)
    - [X] ('warning', 3)
    - [X] ('note', 3)
    - [X] ('seealso', 1)
    - [X] ('signature', 1)
    - [X] ('are', 1)
  - [X] replace external links like: scipy.spatial.distance     <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>
  - [ ] replace examples like: `sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py` 
  - [ ] include more of model description (e.g., Attributes, Notes)
  - [ ] turn section names into subheader (again Attributes, Notes)
  - [ ] turn model names into links to documentation
  - [X] link validation
- [ ] Add dedicated model description definitions
- [ ] Balloons when 100% accuracy reached? Maybe a nice touch, but reinforces over-fitting on training data
- [ ] ⭐ Catch and display warnings during model fit (e.g., "ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.")

### 🧠 GenAI Component
- [ ] ⭐ LLM component reacting to user's most recent change (explain what changed / why)

### 🧹 Code Quality & Refactoring
- [X] Create dedicated `parsers.py` module
- [X] Resolve reuse issues when switching datasets (e.g. `None → toy → None`)
- [X] Refactor toward more Pythonic code (properties, clearer APIs)
- [X] Improve efficiency through cashing predictions/reducing conversions
- [X] Add `ProbaViz` tests
- [ ] Add/update docstrings in viz.py
- [ ] Add usage docstring in `viz.py` showing how to work with it in Jupyter
- [ ] Add app tests
