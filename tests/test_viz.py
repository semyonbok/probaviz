from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC

matplotlib.use("Agg")


# Ensure project root is importable when pytest is executed via `conda run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import ProbaViz  # type: ignore


class CountingSVC(SVC):
    def __init__(self, C: float = 1.0):
        super().__init__(C=C, probability=True)
        self.fit_count = 0
        self.predict_count = 0

    def fit(self, X, y):
        self.fit_count += 1
        return super().fit(X, y)

    def predict(self, X):
        self.predict_count += 1
        return super().predict(X)


@pytest.fixture
def binary_dataset():
    iris = load_iris(as_frame=True)
    data = iris["data"].iloc[:100].copy()
    target = iris["target"].iloc[:100].to_numpy()
    return data, target


def test_validation_length_mismatch(binary_dataset):
    data, target = binary_dataset
    with pytest.raises(ValueError, match="same length"):
        ProbaViz(data=data, target=target[:-1], features=[0, 1])


def test_validation_feature_count(binary_dataset):
    data, target = binary_dataset
    with pytest.raises(ValueError, match="two features"):
        ProbaViz(data=data, target=target, features=[0])


def test_validation_non_numeric_feature():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [True, False], "z": [3.0, 4.0]})
    target = np.array([0, 1])
    with pytest.raises(ValueError, match="not numeric"):
        ProbaViz(data=data, target=target, features=["x", "y"])


def test_validation_grid_res_type(binary_dataset):
    data, target = binary_dataset
    with pytest.raises(TypeError, match="grid_res"):
        ProbaViz(data=data, target=target, features=[0, 1], grid_res=(100, "a"))


def test_fit_populates_train_prediction_cache(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    viz.fit()

    assert viz.is_fitted is True
    assert viz.is_dirty is False
    assert model.fit_count == 1
    assert model.predict_count == 2  # train/test splits
    assert viz._train_predictions is not None
    assert viz._prediction_cache_valid is True


def test_plot_triggers_fit_once_when_dirty_then_reuses(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    fig = viz.plot(contour_on=True, return_fig=True)
    assert fig is not None
    plt.close(fig)

    assert model.fit_count == 1
    prev_predict_count = model.predict_count

    fig = viz.plot(contour_on=True, return_fig=True)
    assert fig is not None
    plt.close(fig)

    assert model.fit_count == 1
    assert model.predict_count >= prev_predict_count


def test_plot_matrices_reuses_cached_predictions_when_clean(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    viz.fit()
    predict_count_after_fit = model.predict_count

    fig = viz.plot_matrices(return_fig=True, mode="confusion")
    assert fig is not None
    plt.close(fig)

    fig = viz.plot_matrices(return_fig=True, mode="error")
    assert fig is not None
    plt.close(fig)

    assert model.predict_count == predict_count_after_fit


def test_plot_matrices_does_not_refit_if_already_clean(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    viz.fit()
    fit_count_after_fit = model.fit_count

    fig = viz.plot_matrices(return_fig=True)
    assert fig is not None
    plt.close(fig)

    assert model.fit_count == fit_count_after_fit


def test_update_params_invalidates_prediction_cache(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    viz.fit()
    assert viz._prediction_cache_valid is True

    viz.update_params(C=0.25)

    assert viz.is_dirty is True
    assert viz._prediction_cache_valid is False


def test_set_dataset_invalidates_prediction_cache(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    viz.fit()
    assert viz._prediction_cache_valid is True

    viz.set_dataset(data, target, [1, 2])

    assert viz.is_dirty is True
    assert viz.is_fitted is False
    assert viz._prediction_cache_valid is False


def test_model_reassignment_invalidates_prediction_cache(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    viz.fit()
    assert viz._prediction_cache_valid is True

    viz.model = CountingSVC(C=0.8)

    assert viz.is_dirty is True
    assert viz._prediction_cache_valid is False


def test_set_dataset_marks_dirty_and_resets_is_fitted(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    viz.fit()
    viz.set_dataset(data, target, [2, 3])

    assert viz.is_dirty is True
    assert viz.is_fitted is False


def test_grid_res_change_updates_mesh_shape(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1], grid_res=(20, 10))

    assert viz._coord_dict["x"].shape == (10, 20)

    viz.grid_res = (30, 15)

    assert viz._coord_dict["x"].shape == (15, 30)
    assert viz._mesh_entries.shape == (450, 2)


def test_plot_without_model_contour_off(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    fig = viz.plot(contour_on=False, return_fig=True)
    assert fig is not None
    plt.close(fig)


def test_plot_without_model_contour_on_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="model must be set"):
        viz.plot(contour_on=True)


def test_plot_with_model_without_predict_proba_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=LinearSVC(), data=data, target=target, features=[0, 1])

    with pytest.raises(AttributeError, match="predict_proba"):
        viz.plot(contour_on=True)


def test_fit_without_model_raises_model_required(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="model must be set"):
        viz.fit()


def test_plot_matrices_without_model_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="model must be set"):
        viz.plot_matrices()


def test_plot_matrices_invalid_mode_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="mode must be either"):
        viz.plot_matrices(mode="oops")  # type: ignore[arg-type]


def test_plot_return_fig_false_returns_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    assert viz.plot(contour_on=False, return_fig=False) is None


def test_plot_matrices_return_fig_false_returns_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    assert viz.plot_matrices(return_fig=False, mode="confusion") is None


def test_streamlit_like_smoke_flow(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    viz.update_params(C=0.1)

    fig = viz.plot(contour_on=True, return_fig=True)
    assert fig is not None
    plt.close(fig)

    fig = viz.plot_matrices(return_fig=True, mode="confusion")
    assert fig is not None
    plt.close(fig)

    fig = viz.plot_matrices(return_fig=True, mode="error")
    assert fig is not None
    plt.close(fig)
