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

from src.viz import ProbaViz  # noqa


class CountingSVC(SVC):
    def __init__(self, C: float = 1.0):
        super().__init__(C=C, probability=True)
        self.fit_count = 0
        self.predict_count = 0
        self.predict_proba_count = 0

    def fit(self, X, y):
        self.fit_count += 1
        return super().fit(X, y)

    def predict(self, X):
        self.predict_count += 1
        return super().predict(X)

    def predict_proba(self, X):
        self.predict_proba_count += 1
        return super().predict_proba(X)


@pytest.fixture
def binary_dataset():
    iris = load_iris(as_frame=True)
    data = iris["data"].iloc[:100].copy()
    target = iris["target"].iloc[:100].to_numpy()
    return data, target


@pytest.fixture
def multiclass_dataset():
    iris = load_iris(as_frame=True)
    return iris["data"].copy(), iris["target"].to_numpy()


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


def test_train_size_property_defaults_to_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])
    assert viz.train_size is None


def test_train_size_property_setter_roundtrip(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    viz.train_size = 0.6
    assert viz.train_size == pytest.approx(0.6)

    viz.train_size = None
    assert viz.train_size is None


def test_split_random_state_property_defaults_to_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])
    assert viz.split_random_state is None


def test_split_random_state_property_setter_roundtrip(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    viz.split_random_state = 7
    assert viz.split_random_state == 7

    viz.split_random_state = None
    assert viz.split_random_state is None


@pytest.mark.parametrize("bad_train_size", ["0.8", object(), True])
def test_train_size_validation_rejects_non_numeric(binary_dataset, bad_train_size):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])
    with pytest.raises(TypeError, match="train_size"):
        viz.train_size = bad_train_size  # type: ignore[assignment]


@pytest.mark.parametrize("bad_train_size", [0.0, 1.0, -0.1, 1.1])
def test_train_size_validation_rejects_out_of_bounds(binary_dataset, bad_train_size):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])
    with pytest.raises(ValueError, match="train_size"):
        viz.train_size = bad_train_size


@pytest.mark.parametrize("bad_split_random_state", ["7", object(), True])
def test_split_random_state_validation_rejects_non_integer(binary_dataset, bad_split_random_state):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])
    with pytest.raises(TypeError, match="split_random_state"):
        viz.split_random_state = bad_split_random_state  # type: ignore[assignment]


def test_split_random_state_validation_rejects_negative(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])
    with pytest.raises(ValueError, match="split_random_state"):
        viz.split_random_state = -1


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


def test_plot_roc_reuses_cached_probabilities_when_clean(binary_dataset):
    data, target = binary_dataset
    model = CountingSVC()
    viz = ProbaViz(model=model, data=data, target=target, features=[0, 1])

    viz.fit()
    predict_proba_count_after_fit = model.predict_proba_count

    fig = viz.plot_roc(return_fig=True, mode="micro_macro")
    assert fig is not None
    plt.close(fig)

    fig = viz.plot_roc(return_fig=True, mode="class")
    assert fig is not None
    plt.close(fig)

    assert model.predict_proba_count == predict_proba_count_after_fit


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


def test_train_size_setter_invalidates_and_recomputes_split(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    viz.fit()
    viz.train_size = 0.6

    assert viz.is_dirty is True
    assert viz.is_fitted is False
    assert viz._prediction_cache_valid is False
    assert viz.train_size == pytest.approx(0.6)
    assert viz._train_xy.shape[0] == 60
    assert viz._test_xy.shape[0] == 40


def test_train_size_persists_across_property_updates(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    viz.train_size = 0.6
    viz.target = target.copy()

    assert viz.train_size == pytest.approx(0.6)
    assert viz._train_xy.shape[0] == 60
    assert viz._test_xy.shape[0] == 40


def test_set_dataset_train_size_override_updates_state(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    viz.train_size = 0.6
    assert viz.train_size == pytest.approx(0.6)

    viz.set_dataset(data, target, [1, 2], train_size=0.8)
    assert viz.train_size == pytest.approx(0.8)

    viz.set_dataset(data, target, [0, 1], train_size=None)
    assert viz.train_size is None


def test_set_dataset_split_random_state_override_updates_state(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1], train_size=0.6)

    viz.split_random_state = 7
    assert viz.split_random_state == 7

    viz.set_dataset(data, target, [1, 2], train_size=0.6, split_random_state=9)
    assert viz.split_random_state == 9

    viz.set_dataset(data, target, [0, 1], train_size=0.6, split_random_state=None)
    assert viz.split_random_state is None


def test_split_random_state_makes_split_reproducible(binary_dataset):
    data, target = binary_dataset
    viz_1 = ProbaViz(
        data=data, target=target, features=[0, 1], train_size=0.6, split_random_state=11
    )
    viz_2 = ProbaViz(
        data=data, target=target, features=[0, 1], train_size=0.6, split_random_state=11
    )

    assert np.array_equal(viz_1._train_xy, viz_2._train_xy)
    assert np.array_equal(viz_1._test_xy, viz_2._test_xy)


def test_strict_stratified_split_failure_message():
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    target = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="strict stratified train/test split"):
        ProbaViz(data=data, target=target, features=[0, 1], train_size=0.8)


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


def test_plot_roc_without_model_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="model must be set"):
        viz.plot_roc()


def test_plot_matrices_invalid_mode_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="mode must be either"):
        viz.plot_matrices(mode="oops")  # type: ignore[arg-type]


def test_plot_roc_invalid_mode_raises(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    with pytest.raises(ValueError, match="mode must be either"):
        viz.plot_roc(mode="oops")  # type: ignore[arg-type]


def test_plot_return_fig_false_returns_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    assert viz.plot(contour_on=False, return_fig=False) is None


def test_plot_matrices_return_fig_false_returns_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    assert viz.plot_matrices(return_fig=False, mode="confusion") is None


def test_plot_roc_return_fig_false_returns_none(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    assert viz.plot_roc(return_fig=False, mode="micro_macro") is None


def test_plot_roc_binary_styles_and_limits(binary_dataset):
    data, target = binary_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    fig = viz.plot_roc(return_fig=True, mode="class")
    assert fig is not None

    axes = fig.axes[0]
    assert axes.get_xlim() == pytest.approx((-0.1, 1.1))
    assert axes.get_ylim() == pytest.approx((-0.1, 1.1))
    assert axes.get_legend() is not None
    assert "AUC =" in axes.get_legend().get_texts()[0].get_text()

    plt.close(fig)


def test_plot_roc_multiclass_has_micro_macro_curves(multiclass_dataset):
    data, target = multiclass_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    fig = viz.plot_roc(return_fig=True, mode="micro_macro", data_split="test")
    assert fig is not None

    axes = fig.axes[0]
    labels = [line.get_label() for line in axes.lines]
    assert any("Macro-average" in label for label in labels)
    assert any("Micro-average" in label for label in labels)

    plt.close(fig)


def test_plot_roc_multiclass_class_mode_has_one_curve_per_class(multiclass_dataset):
    data, target = multiclass_dataset
    viz = ProbaViz(model=CountingSVC(), data=data, target=target, features=[0, 1])

    fig = viz.plot_roc(return_fig=True, mode="class", data_split="test")
    assert fig is not None

    axes = fig.axes[0]
    legend = axes.get_legend()
    assert legend is not None
    curve_labels = [text.get_text() for text in legend.get_texts()]
    assert len(curve_labels) == len(viz.classes)
    assert all("AUC =" in label for label in curve_labels)

    plt.close(fig)


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

    fig = viz.plot_roc(return_fig=True, mode="micro_macro")
    assert fig is not None
    plt.close(fig)

    fig = viz.plot_roc(return_fig=True, mode="class", data_split="test")
    assert fig is not None
    plt.close(fig)
