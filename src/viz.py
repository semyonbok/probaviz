"""
Utilities to visualize classifier probabilities and training performance.
"""

from __future__ import annotations

import warnings
from itertools import cycle
from typing import Any, Optional, Sequence, Tuple, Literal
from numbers import Real
from numbers import Integral
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from sklearn.base import is_classifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

plt.style.use("seaborn-v0_8-ticks")

# Validation and lifecycle error messages.
MSG_MISSING_DATA = "data, target, and features must all be set"
MSG_MODEL_REQUIRED = "a model must be set to compute this plot"
MSG_INVALID_TRAIN_SIZE = "train_size must be a float in the open interval (0, 1) or None"
MSG_INVALID_SPLIT_RANDOM_STATE = "split_random_state must be a non-negative integer or None"

CMAP_COLORS = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]
MARKER_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:grey"
]
MARKER_STYLES = ["o", "s", "P", "v", "D", "X"]


class ProbaViz:
    """
    Visualizes class probabilities computed by a supervised ML model trained on
    classified samples with two numerical features.

    The class keeps mutable state (dataset/model) and uses lazy fitting.
    Any model or dataset update marks the internal state as dirty; fitting
    occurs on demand when plotting methods require predictions.
    """

    FS = 22
    _pr_f1_recall_grid: np.ndarray | None = None
    _pr_f1_precision_grid: np.ndarray | None = None
    _pr_f1_surface: np.ndarray | None = None
    _pr_f1_levels = np.array([0.2, 0.4, 0.6, 0.8])

    def __init__(
        self,
        model: Any = None,
        data: pd.DataFrame | None = None,
        target: Sequence | None = None,
        train_size: float | None = None,
        split_random_state: int | None = None,
        features: Sequence[str | int] | None = None,
        grid_res: Tuple[int, int] = (100, 100),
        *,
        auto_fit: bool = True,
    ):
        self._model: Any = None
        self._data: pd.DataFrame | None = None
        self._target: np.ndarray | None = None
        self._train_size: float | None = None
        self._split_random_state: int | None = None
        self._features: np.ndarray | None = None
        self._grid_res: Tuple[int, int] = (100, 100)
        self._classes: np.ndarray = np.array([])

        self._coord_dict: dict[str, np.ndarray] = {}
        self._mesh_entries: np.ndarray = np.empty((0, 2))
        self._xy: np.ndarray = np.empty((0, 2))
        self._train_xy: np.ndarray = np.empty((0, 2))
        self._test_xy: np.ndarray = np.empty((0, 2))
        self._train_target: np.ndarray = np.empty(0)
        self._test_target: np.ndarray = np.empty(0)
        self._model_target: np.ndarray = np.empty(0)
        self._model_train_target: np.ndarray = np.empty(0)
        self._model_test_target: np.ndarray = np.empty(0)
        self._train_predictions: np.ndarray | None = None
        self._test_predictions: np.ndarray | None = None
        self._train_predict_proba: np.ndarray | None = None
        self._test_predict_proba: np.ndarray | None = None
        self._target_label_encoder: LabelEncoder | None = None
        self._fit_warnings: list[str] = []
        self._prediction_cache_valid = False
        self._is_dirty = True
        self._is_fitted = False
        self.auto_fit = auto_fit

        self._ensure_pr_f1_cache()
        self.grid_res = grid_res

        if data is not None or target is not None or features is not None:
            if data is None or target is None or features is None:
                raise ValueError(MSG_MISSING_DATA)
            self.set_dataset(
                data,
                target,
                features,
                train_size=train_size,
                split_random_state=split_random_state,
                grid_res=grid_res,
            )

        self.model = model

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, new_model: Any) -> None:
        if new_model is not None:
            self._ensure_model_compatibility(new_model)
        self._model = new_model
        self._mark_dirty()

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError(MSG_MISSING_DATA)
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if self._target is None or self._features is None:
            raise ValueError("target and features must be set before data")
        self.set_dataset(
            value,
            self._target,
            self._features,
            train_size=self._train_size,
            split_random_state=self._split_random_state,
            grid_res=self._grid_res,
        )

    @property
    def target(self) -> np.ndarray:
        if self._target is None:
            raise ValueError(MSG_MISSING_DATA)
        return self._target

    @target.setter
    def target(self, value: Sequence) -> None:
        if self._data is None or self._features is None:
            raise ValueError("data and features must be set before target")
        self.set_dataset(
            self._data,
            value,
            self._features,
            train_size=self._train_size,
            split_random_state=self._split_random_state,
            grid_res=self._grid_res,
        )

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            raise ValueError(MSG_MISSING_DATA)
        return self._features

    @features.setter
    def features(self, value: Sequence[str | int]) -> None:
        if self._data is None or self._target is None:
            raise ValueError("data and target must be set before features")
        self.set_dataset(
            self._data,
            self._target,
            value,
            train_size=self._train_size,
            split_random_state=self._split_random_state,
            grid_res=self._grid_res,
        )

    @property
    def grid_res(self) -> Tuple[int, int]:
        return self._grid_res

    @grid_res.setter
    def grid_res(self, value: Tuple[int, int]) -> None:
        if len(value) != 2 or not all(isinstance(res, int) for res in value):
            raise TypeError("grid_res must contain two integers")
        self._grid_res = value
        if self._data is not None and self._target is not None and self._features is not None:
            self._build_mesh()
            self._mark_dirty()

    @property
    def train_size(self) -> float | None:
        return self._train_size

    @train_size.setter
    def train_size(self, value: float | None) -> None:
        validated = self._validate_train_size(value)
        if self._data is None or self._target is None or self._features is None:
            self._train_size = validated
            return
        self.set_dataset(
            self._data,
            self._target,
            self._features,
            train_size=validated,
            split_random_state=self._split_random_state,
            grid_res=self._grid_res,
        )

    @property
    def split_random_state(self) -> int | None:
        return self._split_random_state

    @split_random_state.setter
    def split_random_state(self, value: int | None) -> None:
        validated = self._validate_split_random_state(value)
        if self._data is None or self._target is None or self._features is None:
            self._split_random_state = validated
            return
        self.set_dataset(
            self._data,
            self._target,
            self._features,
            train_size=self._train_size,
            split_random_state=validated,
            grid_res=self._grid_res,
        )

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @property
    def fit_warnings(self) -> list[str]:
        return self._fit_warnings.copy()

    def _mark_dirty(self) -> None:
        self._fit_warnings = []
        self._is_dirty = True
        self._is_fitted = False
        self._prediction_cache_valid = False

    def _validate_dataset_inputs(
        self,
        data: pd.DataFrame,
        target: Sequence | NDArray,
        features: Sequence[str | int] | NDArray,
        grid_res: Tuple[int, int],
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.shape[0] != len(target):
            raise ValueError("data & target must have the same length")
        if len(features) != 2:
            raise ValueError("two features must be specified for visualization")
        if len(grid_res) != 2 or not all(isinstance(res, int) for res in grid_res):
            raise TypeError("grid_res must contain two integers")

    def _validate_train_size(self, train_size: float | None) -> float | None:
        if train_size is None:
            return None
        if not isinstance(train_size, Real) or isinstance(train_size, bool):
            raise TypeError(MSG_INVALID_TRAIN_SIZE)
        train_size_float = float(train_size)
        if not 0.0 < train_size_float < 1.0:
            raise ValueError(MSG_INVALID_TRAIN_SIZE)
        return train_size_float

    def _validate_split_random_state(self, value: int | None) -> int | None:
        if value is None:
            return None
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(MSG_INVALID_SPLIT_RANDOM_STATE)
        random_state = int(value)
        if random_state < 0:
            raise ValueError(MSG_INVALID_SPLIT_RANDOM_STATE)
        return random_state

    def _select_feature_columns(
        self,
        data: pd.DataFrame,
        features: Sequence[str | int] | NDArray,
    ) -> pd.DataFrame:
        try:
            selected = data.iloc[:, list(features)]
        except (IndexError, ValueError, TypeError):
            try:
                selected = data.loc[:, list(features)]
            except KeyError as exc:
                raise ValueError("features must reference existing data columns") from exc

        for feature in selected.columns:
            if not pd.api.types.is_numeric_dtype(selected[feature]) or pd.api.types.is_bool_dtype(selected[feature]):
                raise ValueError(f"feature {feature} is not numeric")

        return selected

    def _build_mesh(self) -> None:
        x_offset = np.ptp(self._xy[:, 0]) / 100
        y_offset = np.ptp(self._xy[:, 1]) / 100

        x_values = np.linspace(
            self._xy[:, 0].min() - x_offset,
            self._xy[:, 0].max() + x_offset,
            self._grid_res[0],
        )
        y_values = np.linspace(
            self._xy[:, 1].min() - y_offset,
            self._xy[:, 1].max() + y_offset,
            self._grid_res[1],
        )

        mesh_x, mesh_y = np.meshgrid(x_values, y_values)
        self._coord_dict = {"x": mesh_x, "y": mesh_y}
        self._mesh_entries = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    def _encode_targets(
        self,
        target_arr: np.ndarray,
        train_target_arr: np.ndarray,
        test_target_arr: np.ndarray,
    ) -> tuple[LabelEncoder | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if pd.api.types.is_numeric_dtype(target_arr) and not pd.api.types.is_bool_dtype(target_arr):
            classes = np.unique(target_arr)
            return None, target_arr, train_target_arr, test_target_arr, classes

        label_encoder = LabelEncoder()
        encoded_target_arr = label_encoder.fit_transform(target_arr)
        encoded_train_target_arr = label_encoder.transform(train_target_arr)
        encoded_test_target_arr = label_encoder.transform(test_target_arr)
        classes = label_encoder.classes_.copy()
        return (
            label_encoder,
            encoded_target_arr,
            encoded_train_target_arr,
            encoded_test_target_arr,
            classes,
        )

    def _decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if self._target_label_encoder is None:
            return predictions
        return self._target_label_encoder.inverse_transform(predictions)

    def _ensure_model_compatibility(self, model: Any) -> None:
        # XXX this check will create problems when working with other frameworks
        if not is_classifier(model):
            raise TypeError("model must be a scikit-learn classifier")

    def _ensure_fitted_for_plot(self, require_predict_proba: bool = False) -> None:
        if self._model is None:
            raise ValueError(MSG_MODEL_REQUIRED)
        self._ensure_model_compatibility(self._model)

        if require_predict_proba and not hasattr(self._model, "predict_proba"):
            raise AttributeError("model must implement predict_proba for contour plotting")

        if self._is_dirty:
            if not self.auto_fit:
                raise ValueError("model is dirty; enable auto_fit or call fit() before plotting")
            self.fit()

    def set_dataset(
        self,
        data: pd.DataFrame,
        target: Sequence | NDArray,
        features: Sequence[str | int] | NDArray | None = None,
        train_size: float | None = None,
        split_random_state: int | None = None,
        grid_res: Tuple[int, int] | None = None,
    ) -> None:
        """
        Update dataset-related state in one operation.
        """
        active_features: Sequence[str | int] | NDArray
        if features is None:
            if self._features is None:
                raise ValueError("features must be provided when setting dataset for the first time")
            active_features = list(self._features)
        else:
            active_features = features

        active_grid = self._grid_res if grid_res is None else grid_res

        active_train_size = self._validate_train_size(train_size)
        active_split_random_state = self._validate_split_random_state(split_random_state)
        self._validate_dataset_inputs(data, target, active_features, active_grid)
        selected_data = self._select_feature_columns(data, active_features)
        target_arr = np.asarray(target)
        xy = np.ascontiguousarray(selected_data.to_numpy())
        try:
            train_xy, test_xy, train_target_arr, test_target_arr = train_test_split(
                xy, target_arr, stratify=target_arr, train_size=active_train_size,
                random_state=active_split_random_state,
            )
        except ValueError as exc:
            raise ValueError(
                "Unable to perform strict stratified train/test split. "
                "Ensure each class has enough samples for both splits and that "
                "train_size is compatible with class frequencies."
            ) from exc

        (
            target_label_encoder,
            model_target_arr,
            model_train_target_arr,
            model_test_target_arr,
            classes,
        ) = self._encode_targets(target_arr, train_target_arr, test_target_arr)

        self._data = selected_data
        self._features = selected_data.columns.to_numpy()
        self._classes = classes
        self._xy = xy
        self._train_size = active_train_size
        self._split_random_state = active_split_random_state
        self._train_xy = train_xy
        self._test_xy = test_xy
        self._target = target_arr
        self._train_target = train_target_arr
        self._test_target = test_target_arr
        self._model_target = model_target_arr
        self._model_train_target = model_train_target_arr
        self._model_test_target = model_test_target_arr
        self._target_label_encoder = target_label_encoder
        self._train_predictions = None
        self._test_predictions = None
        self._train_predict_proba = None
        self._test_predict_proba = None
        self._prediction_cache_valid = False
        self._grid_res = active_grid
        self._build_mesh()
        self._mark_dirty()

    def update_params(self, **params: Any) -> None:
        """
        Update model hyperparameters and mark state dirty.
        """
        if self._model is None:
            raise ValueError(MSG_MODEL_REQUIRED)
        if not hasattr(self._model, "set_params"):
            raise AttributeError("model must implement set_params for parameter updates")
        self._model.set_params(**params)
        # XXX if model was fit before, it will remain fit while `_is_fitted` will be False
        self._mark_dirty()

    def fit(self, force: bool = False) -> None:
        """
        Fit the configured model on the current dataset.
        """
        if self._model is None:
            raise ValueError(MSG_MODEL_REQUIRED)
        if self._data is None or self._target is None:
            raise ValueError(MSG_MISSING_DATA)
        self._ensure_model_compatibility(self._model)

        if force or self._is_dirty:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                self._model.fit(self._train_xy, self._model_train_target)
            self._fit_warnings = [
                f"{warning.category.__name__}: {warning.message}"
                for warning in caught_warnings
            ]
            self._train_predictions = self._decode_predictions(self._model.predict(self._train_xy))
            self._test_predictions = self._decode_predictions(self._model.predict(self._test_xy))
            if hasattr(self._model, "predict_proba"):
                self._train_predict_proba = self._model.predict_proba(self._train_xy)
                self._test_predict_proba = self._model.predict_proba(self._test_xy)
            else:
                self._train_predict_proba = None
                self._test_predict_proba = None
            self._prediction_cache_valid = True
            self._is_fitted = True
            self._is_dirty = False

    def _get_train_predictions(self) -> np.ndarray:
        if self._train_predictions is None or not self._prediction_cache_valid:
            self._train_predictions = self._decode_predictions(self.model.predict(self._train_xy))
            self._prediction_cache_valid = True
        return self._train_predictions  # type: ignore

    def _get_test_predictions(self) -> np.ndarray:
        if self._test_predictions is None or not self._prediction_cache_valid:
            self._test_predictions = self._decode_predictions(self.model.predict(self._test_xy))
            self._prediction_cache_valid = True
        return self._test_predictions  # type: ignore

    def _get_train_predict_proba(self) -> np.ndarray:
        if self._train_predict_proba is None:
            self._train_predict_proba = self.model.predict_proba(self._train_xy)
        return self._train_predict_proba

    def _get_test_predict_proba(self) -> np.ndarray:
        if self._test_predict_proba is None:
            self._test_predict_proba = self.model.predict_proba(self._test_xy)
        return self._test_predict_proba

    def _get_split_targets_and_scores(
        self, data_split: Literal["train", "test"]
    ) -> tuple[np.ndarray, np.ndarray]:
        if data_split == "train":
            return self._train_target, self._get_train_predict_proba()
        if data_split == "test":
            return self._test_target, self._get_test_predict_proba()
        raise ValueError("data_split must be either 'train' or 'test'")

    def _compute_roc_curves(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> tuple[
        dict[Any, np.ndarray],
        dict[Any, np.ndarray],
        dict[Any, np.ndarray],
        dict[Any, float],
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        fpr: dict[Any, np.ndarray] = {}
        tpr: dict[Any, np.ndarray] = {}
        thresholds: dict[Any, np.ndarray] = {}
        roc_auc: dict[Any, float] = {}

        y_true_bin = np.column_stack([y_true == current_class for current_class in self.classes])
        y_true_bin = y_true_bin.astype(int)

        for index, current_class in enumerate(self.classes):
            fpr[current_class], tpr[current_class], thresholds[current_class] = roc_curve(
                y_true_bin[:, index], y_score[:, index]
            )
            roc_auc[current_class] = auc(fpr[current_class], tpr[current_class])

        micro_fpr, micro_tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        micro_auc = auc(micro_fpr, micro_tpr)

        all_fpr = np.unique(np.concatenate([fpr[current_class] for current_class in self.classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for current_class in self.classes:
            mean_tpr += np.interp(all_fpr, fpr[current_class], tpr[current_class])
        mean_tpr /= len(self.classes)
        macro_auc = auc(all_fpr, mean_tpr)

        return (
            fpr,
            tpr,
            thresholds,
            roc_auc,
            micro_fpr,
            micro_tpr,
            micro_auc,
            all_fpr,
            mean_tpr,
            macro_auc,
        )

    def _binarize_targets(self, y_true: np.ndarray) -> np.ndarray:
        y_true_bin = np.column_stack([y_true == current_class for current_class in self.classes])
        return y_true_bin.astype(int)

    def _compute_precision_recall_curves(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> tuple[
        dict[Any, np.ndarray],
        dict[Any, np.ndarray],
        dict[Any, np.ndarray],
        dict[Any, float],
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ]:
        precision: dict[Any, np.ndarray] = {}
        recall: dict[Any, np.ndarray] = {}
        thresholds: dict[Any, np.ndarray] = {}
        average_precision: dict[Any, float] = {}

        y_true_bin = self._binarize_targets(y_true)

        for index, current_class in enumerate(self.classes):
            precision[current_class], recall[current_class], thresholds[current_class] = (
                precision_recall_curve(y_true_bin[:, index], y_score[:, index])
            )
            average_precision[current_class] = average_precision_score(
                y_true_bin[:, index], y_score[:, index]
            )

        micro_precision, micro_recall, _ = precision_recall_curve(
            y_true_bin.ravel(), y_score.ravel()
        )
        micro_average_precision = average_precision_score(
            y_true_bin, y_score, average="micro"
        )

        macro_recall = np.linspace(0.0, 1.0, 200)
        mean_precision = np.zeros_like(macro_recall)
        for current_class in self.classes:
            class_recall = recall[current_class][::-1]
            class_precision = precision[current_class][::-1]
            mean_precision += np.interp(macro_recall, class_recall, class_precision)
        mean_precision /= len(self.classes)
        macro_average_precision = auc(macro_recall, mean_precision)
        prevalence = float(y_true_bin.mean())

        return (
            precision,
            recall,
            thresholds,
            average_precision,
            micro_precision,
            micro_recall,
            micro_average_precision,
            mean_precision,
            macro_recall,
            macro_average_precision,
            prevalence,
        )

    @classmethod
    def _ensure_pr_f1_cache(cls) -> None:
        if (
            cls._pr_f1_recall_grid is not None
            and cls._pr_f1_precision_grid is not None
            and cls._pr_f1_surface is not None
        ):
            return

        recall_axis = np.linspace(0.0, 1.0, 300)
        precision_axis = np.linspace(0.0, 1.0, 300)
        recall_grid, precision_grid = np.meshgrid(recall_axis, precision_axis)
        denominator = precision_grid + recall_grid
        f1_surface = np.divide(
            2.0 * precision_grid * recall_grid,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )

        cls._pr_f1_recall_grid = recall_grid
        cls._pr_f1_precision_grid = precision_grid
        cls._pr_f1_surface = f1_surface

    def _plot_pr_f1_contours(self, axes: plt.Axes) -> None:
        self._ensure_pr_f1_cache()
        contour_set = axes.contour(
            self._pr_f1_recall_grid,
            self._pr_f1_precision_grid,
            self._pr_f1_surface,
            levels=self._pr_f1_levels,
            colors="0.6",
            linestyles="--",
            linewidths=1.0,
            alpha=0.5,
            zorder=0,
        )
        axes.clabel(contour_set, inline=True, fontsize=max(10, self.FS // 2), fmt="F1=%.1f")

    def _style_roc_axes(self, axes: plt.Axes) -> None:
        axes.set_xlim(-0.1, 1.1)
        axes.set_ylim(-0.1, 1.1)
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("False Positive Rate")
        axes.set_ylabel("True Positive Rate")
        axes.grid(True, alpha=0.25)
        axes.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1.5, label="_nolegend_")

    def _style_pr_axes(self, axes: plt.Axes) -> None:
        axes.set_xlim(-0.1, 1.1)
        axes.set_ylim(-0.1, 1.1)
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("Recall")
        axes.set_ylabel("Precision")
        axes.grid(True, alpha=0.25)

    def plot(
        self,
        contour_on: bool = True,
        return_fig: bool = False,
        fig_size: Tuple[int, int] = (12, 6),
    ) -> Optional[plt.Figure]:
        """
        Plot train data and optional model probability contours.
        """
        if self._data is None or self._target is None:
            raise ValueError(MSG_MISSING_DATA)

        fig, axes = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
        axes.set_xlabel(self.data.columns[0], fontsize=self.FS)
        axes.set_ylabel(self.data.columns[1], fontsize=self.FS)
        axes.set_facecolor("#F0F2F6")

        cmap_cycle = cycle(CMAP_COLORS)
        m_color_cycle = cycle(MARKER_COLORS)
        m_style_cycle = cycle(MARKER_STYLES)

        pred_proba = None
        pred_class = None
        levels = np.arange(0.0, 1.05, 0.05)
        class_handles = []

        if contour_on:
            self._ensure_fitted_for_plot(require_predict_proba=True)
            pred_proba = self.model.predict_proba(self._mesh_entries)
            pred_class = np.argmax(pred_proba, axis=1)
            train_f1 = f1_score(
                self._train_target, self._get_train_predictions(), average="macro")
            test_f1 = f1_score(
                self._test_target, self._get_test_predictions(), average="macro")

            axes.set_facecolor("k")
            axes.text(
                1.04,
                0.00,
                (
                    f"Train F1\n(Macro):\n{train_f1:.2%}\n"
                    "$----$\n"
                    f"Test F1\n(Macro):\n{test_f1:.2%}"
                ),
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=axes.transAxes,
                fontsize=self.FS,
            )

        for index, current_class in enumerate(self.classes):
            if contour_on and pred_proba is not None and pred_class is not None:
                current_cmap = next(cmap_cycle)
                class_proba = np.where(pred_class == index, pred_proba[:, index], np.nan)

                if not np.isnan(class_proba).all():
                    cs0 = axes.contourf(
                        self._coord_dict["x"],
                        self._coord_dict["y"],
                        class_proba.reshape(self._coord_dict["x"].shape),
                        cmap=current_cmap,
                        alpha=1,
                        vmin=0,
                        vmax=1,
                        levels=levels,
                    )

                    contour_color = "w" if cs0.get_cmap().name == "Greys" else "k"
                    cs1 = axes.contour(cs0, levels=levels[::3], colors=contour_color)
                    axes.clabel(cs1, levels[::3], inline=True, fontsize=self.FS)

            marker_color = next(m_color_cycle)
            marker_style = next(m_style_cycle)
            train_scatter = axes.scatter(
                self._train_xy[self._train_target == current_class, 0],
                self._train_xy[self._train_target == current_class, 1],
                s=100,
                c=marker_color,
                marker=marker_style,
                edgecolor="k",
                linewidths=2,
                zorder=2,
                label=current_class,
            )
            class_handles.append(train_scatter)
            axes.scatter(
                self._test_xy[self._test_target == current_class, 0],
                self._test_xy[self._test_target == current_class, 1],
                s=100,
                c=marker_color,
                marker=marker_style,
                edgecolor="w",
                linewidths=2,
                zorder=2,
            )

        split_train_handle = (
            Line2D(
                [0], [0], marker="o", linestyle="None",
                markerfacecolor="#539ECD", markeredgecolor="k",
                markersize=21,
            ),
            Line2D(
                [0], [0], marker="o", linestyle="None",
                markerfacecolor="tab:blue", markeredgecolor="k",
                markeredgewidth=2.2, markersize=9,
            ),
        )
        split_test_handle = (
            Line2D(
                [0], [0], marker="o", linestyle="None",
                markerfacecolor="#539ECD", markeredgecolor="k",
                markersize=21,
            ),
            Line2D(
                [0], [0], marker="o", linestyle="None",
                markerfacecolor="tab:blue", markeredgecolor="w",
                markeredgewidth=2.2, markersize=9,
            ),
        )
        handles = [split_train_handle, split_test_handle, *class_handles,]
        labels = ["Train", "Test", *map(lambda h: h.get_label(), class_handles)]

        axes.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
            title="Subset / Class",
            borderaxespad=0,
            borderpad=0,
            handletextpad=1.0,
            handlelength=0.0,
            alignment="left",
            fontsize=self.FS,
            title_fontsize=self.FS,
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        )
        axes.tick_params(axis="both", which="major", labelsize=self.FS)

        if return_fig:
            return fig
        return None

    def replot(self, contour_on: bool = True, **params: Any) -> Optional[plt.Figure]:
        """
        Tuned for a widget use; adjusts passed hyperparameters  of the set
        supervised ML model and calls `plot` method.
        **Warning:** changes hyperparameters of the set model.
        """
        self.update_params(**params)
        return self.plot(contour_on=contour_on)

    def plot_matrices(
        self,
        return_fig: bool = False,
        data_split: Literal["train", "test"] = "train",
        mode: Literal["confusion", "error"] = "confusion",
        fig_size: Tuple[int, int] = (16, 9),
    ) -> Optional[plt.Figure]:
        """
        Plot raw and normalized confusion/error matrices.
        """
        self._ensure_fitted_for_plot(require_predict_proba=False)
        if data_split == "train":
            y_pred = self._get_train_predictions()
            y_true = self._train_target
        elif data_split == "test":
            y_pred = self._get_test_predictions()
            y_true = self._test_target
        else:
            raise ValueError("data_split must be either 'train' or 'test'")

        if mode == "confusion":
            sample_weight = None
            cmap: str = "Greys"
        elif mode == "error":
            sample_weight = y_pred != y_true
            cmap = "Reds"
        else:
            raise ValueError("mode must be either 'confusion' or 'error'")

        with plt.rc_context({"font.size": self.FS}):
            fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True)
            fig.tight_layout(pad=2)

            ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sample_weight,
                normalize=None,
                colorbar=False,
                cmap=cmap,
                ax=axes[0],
            )
            axes[0].set_title("Raw Counts")
            axes[0].set_xlabel(None)

            ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sample_weight,
                normalize="true",
                colorbar=False,
                cmap=cmap,
                ax=axes[1],
                values_format=".0%",
                im_kw={"vmin": 0, "vmax": 1},
            )
            axes[1].set_title("Normalized by Row")
            axes[1].set_xlabel(None)

            ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sample_weight,
                normalize="pred",
                colorbar=False,
                cmap=cmap,
                ax=axes[2],
                values_format=".0%",
                im_kw={"vmin": 0, "vmax": 1},
            )
            axes[2].set_title("Normalized by Column")

        if return_fig:
            return fig
        return None

    def plot_roc(
        self,
        return_fig: bool = False,
        data_split: Literal["train", "test"] = "train",
        mode: Literal["micro_macro", "class"] = "micro_macro",
        fig_size: Tuple[int, int] = (8, 8),
    ) -> Optional[plt.Figure]:
        """
        Plot split-specific ROC curves for micro/macro aggregates or per-class
        one-vs-rest comparisons.
        """
        self._ensure_fitted_for_plot(require_predict_proba=True)
        y_true, y_score = self._get_split_targets_and_scores(data_split)

        (
            fpr,
            tpr,
            thresholds,
            roc_auc,
            micro_fpr,
            micro_tpr,
            micro_auc,
            macro_fpr,
            macro_tpr,
            macro_auc,
        ) = self._compute_roc_curves(y_true, y_score)

        with plt.rc_context({"font.size": self.FS}):
            fig, axes = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
            self._style_roc_axes(axes)

            if mode == "micro_macro":
                axes.plot(
                    macro_fpr,
                    macro_tpr,
                    color="k",
                    linestyle="-",
                    linewidth=2.5,
                    label=f"Macro-average (AUC = {macro_auc:.3f})",
                )
                axes.plot(
                    micro_fpr,
                    micro_tpr,
                    color="k",
                    linestyle="--",
                    linewidth=2.5,
                    label=f"Micro-average (AUC = {micro_auc:.3f})",
                )
                axes.set_title("Micro / Macro ROC Curves")
            elif mode == "class":
                for index, current_class in enumerate(self.classes):
                    cmap_name = CMAP_COLORS[index % len(CMAP_COLORS)]
                    cmap = plt.get_cmap(cmap_name)
                    line_color = cmap(0.75)
                    marker_style = MARKER_STYLES[index % len(MARKER_STYLES)]

                    axes.plot(
                        fpr[current_class],
                        tpr[current_class],
                        color=line_color,
                        linewidth=2.0,
                        marker=marker_style,
                        markevery=max(1, len(fpr[current_class]) // 8),
                        label=f"{current_class} (AUC = {roc_auc[current_class]:.3f})",
                    )
                    axes.scatter(
                        fpr[current_class],
                        tpr[current_class],
                        c=np.clip(thresholds[current_class], 0.0, 1.0),
                        zorder=3,
                        cmap=cmap,
                        edgecolor="k",
                        vmin=0,
                        vmax=1,
                        marker=marker_style,
                        s=70,
                    )
                axes.set_title("Class-vs-Rest ROC Curves")
            else:
                raise ValueError("mode must be either 'micro_macro' or 'class'")

            axes.legend(loc="lower right", frameon=True, framealpha=.5)

        if return_fig:
            return fig
        return None

    def plot_pr(
        self,
        return_fig: bool = False,
        data_split: Literal["train", "test"] = "train",
        mode: Literal["micro_macro", "class"] = "micro_macro",
        fig_size: Tuple[int, int] = (8, 8),
    ) -> Optional[plt.Figure]:
        """
        Plot split-specific precision-recall curves for micro/macro aggregates
        or per-class one-vs-rest comparisons.
        """
        self._ensure_fitted_for_plot(require_predict_proba=True)
        y_true, y_score = self._get_split_targets_and_scores(data_split)

        (
            precision,
            recall,
            thresholds,
            average_precision,
            micro_precision,
            micro_recall,
            micro_average_precision,
            macro_precision,
            macro_recall,
            macro_average_precision,
            prevalence,
        ) = self._compute_precision_recall_curves(y_true, y_score)

        with plt.rc_context({"font.size": self.FS}):
            fig, axes = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
            self._style_pr_axes(axes)
            self._plot_pr_f1_contours(axes)

            if mode == "micro_macro":
                axes.plot(
                    macro_recall,
                    macro_precision,
                    color="k",
                    linestyle="-",
                    linewidth=2.5,
                    label=f"Macro-average (AP = {macro_average_precision:.3f})",
                )
                axes.plot(
                    micro_recall,
                    micro_precision,
                    color="k",
                    linestyle="--",
                    linewidth=2.5,
                    label=f"Micro-average (AP = {micro_average_precision:.3f})",
                )
                axes.axhline(
                    prevalence,
                    color="0.5",
                    linestyle=":",
                    linewidth=1.5,
                    label="_nolegend_",
                )
                axes.set_title("Micro / Macro Precision-Recall Curves")
            elif mode == "class":
                for index, current_class in enumerate(self.classes):
                    cmap_name = CMAP_COLORS[index % len(CMAP_COLORS)]
                    cmap = plt.get_cmap(cmap_name)
                    line_color = cmap(0.75)
                    marker_style = MARKER_STYLES[index % len(MARKER_STYLES)]

                    axes.plot(
                        recall[current_class],
                        precision[current_class],
                        color=line_color,
                        linewidth=2.0,
                        marker=marker_style,
                        markevery=max(1, len(recall[current_class]) // 8),
                        label=f"{current_class} (AP = {average_precision[current_class]:.3f})",
                    )
                    if len(thresholds[current_class]) > 0:
                        axes.scatter(
                            recall[current_class][1:],
                            precision[current_class][1:],
                            c=np.clip(thresholds[current_class], 0.0, 1.0),
                            zorder=3,
                            cmap=cmap,
                            edgecolor="k",
                            vmin=0,
                            vmax=1,
                            marker=marker_style,
                            s=70,
                        )
                axes.set_title("Class-vs-Rest Precision-Recall Curves")
            else:
                raise ValueError("mode must be either 'micro_macro' or 'class'")

            axes.legend(loc="lower left", frameon=True, framealpha=.5)

        if return_fig:
            return fig
        return None

    def get_classification_metrics(
        self, data_split: Literal["train", "test"] = "train"
    ) -> dict[str, pd.DataFrame]:
        """
        Compute split-specific classification metrics with class-wise and aggregate summaries.

        Formulas:
        - Class-wise precision/recall/F1/support come from
          ``precision_recall_fscore_support(y_true, y_pred, labels=self.classes, average=None)``.
        - Class-wise one-vs-rest log loss for class ``k`` is
          ``log_loss(1[y=k], p_k)``, where ``p_k`` is the predicted probability
          for class ``k`` and ``1[y=k]`` is the binary indicator target.
        - Class-wise one-vs-rest Brier score for class ``k`` is
          ``mean((p_k - 1[y=k])**2)``.
        - Aggregate ``log_loss`` is the multiclass cross-entropy
          ``log_loss(y_true, y_score, labels=self.classes)``.
        - Aggregate ``brier_score`` values are one-vs-rest summaries:
          ``micro`` is computed on flattened one-vs-rest targets/scores,
          ``macro`` is the unweighted mean of class-wise OVR Brier scores,
          and ``weighted`` is the support-weighted mean of class-wise OVR Brier
          scores.
        """
        self._ensure_fitted_for_plot(require_predict_proba=True)
        y_true, y_score = self._get_split_targets_and_scores(data_split)
        y_pred = (
            self._get_train_predictions()
            if data_split == "train"
            else self._get_test_predictions()
        )

        precision, recall, f1_values, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.classes, average=None, zero_division=0
        )
        y_true_bin = self._binarize_targets(y_true)

        class_log_loss_ovr: list[float] = []
        class_brier_ovr: list[float] = []
        for index, _ in enumerate(self.classes):
            class_log_loss_ovr.append(log_loss(y_true_bin[:, index], y_score[:, index]))
            class_brier_ovr.append(brier_score_loss(y_true_bin[:, index], y_score[:, index]))

        class_specific_df = pd.DataFrame(
            {
                "class": self.classes,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_values,
                "log_loss_ovr": class_log_loss_ovr,
                "brier_score_ovr": class_brier_ovr,
                "support": support.astype(int),
            }
        )

        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        total_support = int(np.sum(support))
        class_brier_arr = np.asarray(class_brier_ovr, dtype=float)
        class_support_arr = support.astype(float)
        weighted_brier = (
            float(np.average(class_brier_arr, weights=class_support_arr))
            if total_support > 0
            else float("nan")
        )
        aggregate_log_loss = float(log_loss(y_true, y_score, labels=list(self.classes)))
        micro_brier = float(brier_score_loss(y_true_bin.ravel(), y_score.ravel()))
        macro_brier = float(np.mean(class_brier_arr))

        aggregate_df = pd.DataFrame(
            {
                "aggregate": ["micro", "macro", "weighted"],
                "precision": [micro_precision, macro_precision, weighted_precision],
                "recall": [micro_recall, macro_recall, weighted_recall],
                "f1_score": [micro_f1, macro_f1, weighted_f1],
                "log_loss": [aggregate_log_loss, aggregate_log_loss, aggregate_log_loss],
                "brier_score": [micro_brier, macro_brier, weighted_brier],
                "support": [total_support, total_support, total_support],
            }
        )

        return {"class_specific_df": class_specific_df, "aggregate_df": aggregate_df}
