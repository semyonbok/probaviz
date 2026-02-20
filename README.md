# ProbaViz

**Probaviz** is a Streamlit app for interactive visualization of class probability scores and decision boundaries of 2D classifiers (primarily from scikit-learn).

The project is inspired by the many visual examples in the [scikit-learn gallery](https://scikit-learn.org/stable/auto_examples/index.html) and by the decision boundary helper utilities presented in a [scikit-learn MOOC](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/). While most existing examples focus on binary classification, Probaviz extends these ideas to **multiclass settings**, enabling visualization of predicted class probability scores for datasets with more than two classes.

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
