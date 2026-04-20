# ProbaViz

## 🏗️ Ongoing Work

### 🧱 Project Setup & Structure
- [ ] ⭐ Publish app on Streamlit Community Cloud
- [ ] ⭐ Add “Open in Streamlit” badge
- [ ] Finalize and clean `requirements.txt`
- [X] Update repo README file

### 🎓 Learning & Tools
- [ ] Deep-dive into sklearn documentation:
  - [X] Model Evaluation, Metrics & Scorers at [Skolar](https://skolar.probabl.ai/en/a/6004069601880409583;p=1,1006594772100055529;pa=0)
  - [X] Metrics and scoring: quantifying the quality of predictions at [User Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
  - [X] Multiclass Receiver Operating Characteristic (ROC) at [Model Selection Examples](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
  - [X] Precision-Recal at [Model Selection Examples](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
  - [ ] `sklearn.inspection.DecisionBoundaryDisplay`
- [ ] Re-read Streamlit documentation (best practices, deployment, genAI apps)
- [ ] Re-read Groq Documentation
- [X] Familiarise with and try Codex
  - [ ] Set up and run a Cloud env to implement a feature
  - [X] Try CLI
- [ ] Agentic AI course
- [ ] Check out Andrew's Ng Context Hub

### 📊 Data Handling
- [ ] ⭐ Synthetic data generation
- [ ] Decide on whether cashing datasets is needed at all
- [X] Binary/multiclass toy datasets
- [X] Train/test data split

### 📈 Metrics & Evaluation
- [ ] ⭐ Dedicated tab with common classification metrics (perhaps leveraging `skore.EstimatorReport`)
- [ ] Calibration display for binary classification
- [ ] Learning curve
- [X] ROC curves
- [X] Precision–Recall curves
- [X] Modify confusion/error matrix visualization

### 🎛️ Model & Interaction Features
- [ ] Reset model button
- [ ] Allow kernel customization for `GaussianProcessClassifier` 
- [ ] Allow picking an estimator for `AdaBoostClassifier` and `BaggingClassifier`
- [ ] Add meta ensemble models (voting, stacking, etc.)
- [ ] Output (some) model attributes (post-learning parameters with "_" suffix)
- [ ] Add classifiers from other frameworks
- [ ] Add feature pre-processing
  - [X] Identify relevant pre-processors
  - [X] Integrate into `app.py` and `parsers.py`
  - [ ] Allow for parameter customization
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

### 🧩 UI / UX Improvements
- [ ] ⭐ Dedicated help/info toggles for all data visualization tabs
- [ ] ⭐ Catch and display warnings during model fit (e.g., "ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.")
- [ ] Migrate to a more interactive plotting framework like `plotly` 
- [ ] Colour picking for probability surfaces
- [ ] Add emojis/icons to model selection widget
- [ ] Add deprecated badge against affected hyper-parameters
- [ ] Add dedicated model description definitions
- [ ] Balloons when a 100% metric reached on a test subset
- [ ] Improve model / parameter parsing
  - [ ] model info expander improvements
    - [ ] turn model name into link to `class` documentation
    - [ ] include more model details (e.g., Attributes, Notes)
    - [ ] turn section names into subheaders
    - [X] in the code snippet, also give model import statement
  - [ ] instead of caching with JSON, consider caching at runtime with `streamlit`:
    ```python
    @st.cache_data
    def get_model_desc(model_key: str) -> str:
        model = MODELS[model_key].factory()
        return parse_model_desc(model)

    @st.cache_data
    def get_param_desc(model_key: str) -> dict[str, str]:
        model = MODELS[model_key].factory()
        return parse_param_desc(model)
    ```
  - [X] replace restructured text with markdown links (there are also arxiv and doi roles):
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
  - [X] replace examples like: `sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py` 
  - [X] link validation
  - [X] simply cache processed model/params markdown
- [X] Move model params out of contour plot title
- [X] Standardize widget generation patterns (model registry?)
- [X] Allow for optional selection of `random_state`
- [X] Skip probability surface plotting for `SGDClassifier` and SVMs if model configs have no `predict_proba` method, allowing to explore other tabs.

### 🧠 GenAI Component
- [ ] ⭐ LLM component reacting to user's most recent change (explain what changed / why)

### 🧹 Code Quality & Refactoring
- [ ] Add/update docstrings in viz.py
- [ ] Add usage docstring in `viz.py` showing how to work with it in Jupyter
- [ ] Add app tests
- [X] Create dedicated `parsers.py` module
- [X] Resolve reuse issues when switching datasets (e.g. `None → toy → None`)
- [X] Refactor toward more Pythonic code (properties, clearer APIs)
- [X] Improve efficiency through cashing predictions/reducing conversions
- [X] Add `ProbaViz` tests
