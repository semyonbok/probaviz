# ProbaViz

**Probaviz** is a Streamlit app for interactive visualization of class probabilities and decision boundaries of 2D classifiers (primarily from scikit-learn).

The project is inspired by the many visual examples in the [scikit-learn gallery](https://scikit-learn.org/stable/auto_examples/index.html) and by the decision boundary helper utilities presented in a [scikit-learn MOOC](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/). While most existing examples focus on binary classification, Probaviz extends these ideas to **multiclass settings**, enabling visualization of predicted class probabilities for datasets with more than two classes.

The figures below illustrate a synthetic dataset with four classes. The probability contours are produced using classifiers such as `sklearn.neighbors.KNeighborsClassifier` and `sklearn.ensemble.RandomForestClassifier`, trained on two numerical features. Beyond static plots, the underlying visualization logic is designed to support interactivity, allowing users to explore how model hyperparameters affect decision boundaries in real time.

<details>
  <summary>Example visualizations (multiclass probability surfaces)</summary>

## Training Data Set
![image](https://user-images.githubusercontent.com/94805866/166163074-6eb26a9d-d6c6-4c7d-860a-1bf9d9e1c5b7.png)

## K Nearest Neigbors
![image](https://user-images.githubusercontent.com/94805866/166163537-976b8c0d-911d-4fa9-8571-5b625a734a8d.png)

## Random Forest
![image](https://user-images.githubusercontent.com/94805866/166163493-3c123e4a-2a98-4922-8a97-4122d0d02d0d.png)
</details>

---

## 🎯 Project Goals
- Interactive visualization of decision boundaries, probability surfaces and classification metrics
- Support for multiple classifiers on 2D toy and synthetic datasets
- Clear, educational UI suitable for demos and teaching
- Deployable on Streamlit Community Cloud

---

## 🏗️ Ongoing Work
### Project Setup & Structure
- [X] Update repo README file
- [ ] Finalize and clean `requirements.txt`
- [ ] Publish app on Streamlit Community Cloud
- [ ] Add “Open in Streamlit” badge

### 🎓 Learning & Tools
- [ ] Explore latest capabilities of  `sklearn.inspection.DecisionBoundaryDisplay`
- [ ] Re-read Streamlit documentation (best practices, deployment, genAI apps)
- [ ] Re-read VS Code documentation on AI tools and try GitHub Copilot
- [X] Familiarise with and try Codex

### 📊 Data Handling
- [ ] Binary/multiclass toy datasets
- [ ] Synthetic data generation
- [ ] Train/test data split

### 📈 Metrics & Evaluation
- [ ] F1 score support
- [ ] Modify confusion matrix visualization
- [ ] ROC curves
- [ ] Precision–Recall curves

### 🎛️ Model & Interaction Features
- [ ] Reset model button
- [X] Ensure switching model resets widgets
- [X] Graceful failure in case of invalid hyper-parameter combo
- [ ] Add all "standalone" scikit-learn classifiers:
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
  - [ ] `sklearn.ensemble.HistGradientBoostingClassifier`
  - [ ] `sklearn.ensemble.AdaBoostClassifier`
  - [ ] `sklearn.ensemble.BaggingClassifier`
  - [ ] `sklearn.svm.SVC`
  - [ ] `sklearn.svm.NuSVC`
  - [ ] `sklearn.neural_network.MLPClassifier`
  - [ ] `sklearn.gaussian_process.GaussianProcessClassifier`
  - [ ] `sklearn.naive_bayes.GaussianNB`
  - [ ] `sklearn.naive_bayes.BernoulliNB`
  - [ ] `sklearn.naive_bayes.MultinomialNB`
  - [ ] `sklearn.naive_bayes.ComplementNB`
  - [ ] `sklearn.naive_bayes.CategoricalNB`
  - [ ] `sklearn.semi_supervised.LabelPropagation`
  - [ ] `sklearn.semi_supervised.LabelSpreading`

- [ ] Add meta ensemble models (voting, stacking, etc.)
- [ ] Add feature pre-processing

### 🧩 UI / UX Improvements
- [ ] Migrate to a more interactive plotting framework like `plotly` 
- [ ] Colour picking for probability surfaces
- [ ] Add emojis/icons to model selection widget
- [ ] Dedicated help/info toggles for all data visualization tabs
- [X] Move model params out of contour plot title
- [X] Standardize widget generation patterns (model registry?)
- [X] Allow for optional selection of `random_state`
- [ ] Add deprecated badge against affected hyper-parameters
- [ ] Skip probability surface plotting for `SGDClassifier` to enable all loss functions

### 🧠 GenAI Component
- [ ] LLM component reacting to user's most recent change (explain what changed / why)

### 🧹 Code Quality & Refactoring
- [ ] Add usage docstring in `viz.py` showing how to work with it in Jupyter
- [ ] Create dedicated `utils.py` module
- [ ] Improve dataset / config parser
- [ ] Add dedicated model description definitions
- [X] Resolve reuse issues when switching datasets (e.g. `None → toy → None`)
- [ ] Add class attribute `n_unique` (e.g. from `self.train_target`)
- [ ] Refactor toward more Pythonic code (properties, clearer APIs)
