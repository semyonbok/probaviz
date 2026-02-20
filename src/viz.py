"""
Utilities to visualize classifier probabilities and training performance.
"""

from __future__ import annotations

from itertools import cycle
from typing import Any, Optional, Sequence, Tuple, Literal
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import ConfusionMatrixDisplay

plt.style.use("seaborn-v0_8-ticks")

# Validation and lifecycle error messages.
MSG_MISSING_DATA = "train_data, train_target, and features must all be set"
MSG_MODEL_REQUIRED = "a model must be set to compute this plot"
MSG_MODEL_CLASSIFIER = "model must be a scikit-learn classifier"

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

    def __init__(
        self,
        model: Any = None,
        train_data: pd.DataFrame | None = None,
        train_target: Sequence | None = None,
        features: Sequence[str | int] | None = None,
        grid_res: Tuple[int, int] = (100, 100),
        *,
        auto_fit: bool = True,
    ):
        self._model: Any = None
        self._train_data: pd.DataFrame | None = None
        self._train_target: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._grid_res: Tuple[int, int] = (100, 100)
        self._classes: np.ndarray = np.array([])

        self._coord_dict: dict[str, np.ndarray] = {}
        self._mesh_entries: np.ndarray = np.empty((0, 2))
        self._train_xy: np.ndarray = np.empty((0, 2))
        self._x_values: np.ndarray = np.empty(0)
        self._y_values: np.ndarray = np.empty(0)
        self._class_masks: dict[Any, np.ndarray] = {}
        self._train_predictions: np.ndarray | None = None
        self._prediction_cache_valid = False
        self._is_dirty = True
        self._is_fitted = False
        self.auto_fit = auto_fit

        self.grid_res = grid_res

        if train_data is not None or train_target is not None or features is not None:
            if train_data is None or train_target is None or features is None:
                raise ValueError(MSG_MISSING_DATA)
            self.set_dataset(train_data, train_target, features, grid_res=grid_res)

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
    def train_data(self) -> pd.DataFrame:
        if self._train_data is None:
            raise ValueError(MSG_MISSING_DATA)
        return self._train_data

    @train_data.setter
    def train_data(self, value: pd.DataFrame) -> None:
        if self._train_target is None or self._features is None:
            raise ValueError("train_target and features must be set before train_data")
        self.set_dataset(value, self._train_target, self._features, grid_res=self._grid_res)

    @property
    def train_target(self) -> np.ndarray:
        if self._train_target is None:
            raise ValueError(MSG_MISSING_DATA)
        return self._train_target

    @train_target.setter
    def train_target(self, value: Sequence) -> None:
        if self._train_data is None or self._features is None:
            raise ValueError("train_data and features must be set before train_target")
        self.set_dataset(self._train_data, value, self._features, grid_res=self._grid_res)

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            raise ValueError(MSG_MISSING_DATA)
        return self._features

    @features.setter
    def features(self, value: Sequence[str | int]) -> None:
        if self._train_data is None or self._train_target is None:
            raise ValueError("train_data and train_target must be set before features")
        self.set_dataset(self._train_data, self._train_target, value, grid_res=self._grid_res)

    @property
    def grid_res(self) -> Tuple[int, int]:
        return self._grid_res

    @grid_res.setter
    def grid_res(self, value: Tuple[int, int]) -> None:
        if len(value) != 2 or not all(isinstance(res, int) for res in value):
            raise TypeError("grid_res must contain two integers")
        self._grid_res = value
        if self._train_data is not None and self._train_target is not None and self._features is not None:
            self._build_mesh()
            self._mark_dirty()

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    def _mark_dirty(self) -> None:
        self._is_dirty = True
        self._is_fitted = False
        self._prediction_cache_valid = False

    def _validate_dataset_inputs(
        self,
        train_data: pd.DataFrame,
        train_target: Sequence | NDArray,
        features: Sequence[str | int] | NDArray,
        grid_res: Tuple[int, int],
    ) -> None:
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must be a pandas DataFrame")
        if train_data.shape[0] != len(train_target):
            raise ValueError("data & target must have the same length")
        if len(features) != 2:
            raise ValueError("two features must be specified for visualization")
        if len(grid_res) != 2 or not all(isinstance(res, int) for res in grid_res):
            raise TypeError("grid_res must contain two integers")

    def _select_feature_columns(
        self,
        train_data: pd.DataFrame,
        features: Sequence[str | int] | NDArray,
    ) -> pd.DataFrame:
        try:
            selected = train_data.iloc[:, list(features)]
        except (IndexError, ValueError, TypeError):
            try:
                selected = train_data.loc[:, list(features)]
            except KeyError as exc:
                raise ValueError("features must reference existing train_data columns") from exc

        for feature in selected.columns:
            if not pd.api.types.is_numeric_dtype(selected[feature]) or pd.api.types.is_bool_dtype(selected[feature]):
                raise ValueError(f"feature {feature} is not numeric")

        return selected

    def _build_mesh(self) -> None:
        x_offset = np.ptp(self._x_values) / 100
        y_offset = np.ptp(self._y_values) / 100

        x_values = np.linspace(
            self._x_values.min() - x_offset,
            self._x_values.max() + x_offset,
            self._grid_res[0],
        )
        y_values = np.linspace(
            self._y_values.min() - y_offset,
            self._y_values.max() + y_offset,
            self._grid_res[1],
        )

        mesh_x, mesh_y = np.meshgrid(x_values, y_values)
        self._coord_dict = {"x": mesh_x, "y": mesh_y}
        self._mesh_entries = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    def _ensure_model_compatibility(self, model: Any) -> None:
        # XXX this check will create problems when working with other frameworks
        if not is_classifier(model):
            raise TypeError(MSG_MODEL_CLASSIFIER)

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
        train_data: pd.DataFrame,
        train_target: Sequence | NDArray,
        features: Sequence[str | int] | NDArray | None = None,
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

        self._validate_dataset_inputs(train_data, train_target, active_features, active_grid)
        selected_data = self._select_feature_columns(train_data, active_features)
        train_target_arr = np.asarray(train_target)
        train_xy = np.ascontiguousarray(selected_data.to_numpy())

        self._train_data = selected_data
        self._train_target = train_target_arr
        self._features = selected_data.columns.to_numpy()
        self._classes = np.unique(train_target_arr)
        self._train_xy = train_xy
        self._x_values = train_xy[:, 0]
        self._y_values = train_xy[:, 1]
        self._class_masks = {
            current_class: train_target_arr == current_class
            for current_class in self._classes
        }
        self._train_predictions = None
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
        if self._train_data is None or self._train_target is None:
            raise ValueError(MSG_MISSING_DATA)
        self._ensure_model_compatibility(self._model)

        if force or self._is_dirty:
            self._model.fit(self._train_xy, self.train_target)
            self._train_predictions = self._model.predict(self._train_xy)
            self._prediction_cache_valid = True
            self._is_fitted = True
            self._is_dirty = False

    def _get_train_predictions(self) -> np.ndarray:
        if self._train_predictions is None or not self._prediction_cache_valid:
            self._train_predictions = self.model.predict(self._train_xy)
            self._prediction_cache_valid = True
        return self._train_predictions

    def plot(
        self,
        contour_on: bool = True,
        return_fig: bool = False,
        fig_size: Tuple[int, int] = (12, 6),
    ) -> Optional[plt.Figure]:
        """
        Plot train data and optional model probability contours.
        """
        if self._train_data is None or self._train_target is None:
            raise ValueError(MSG_MISSING_DATA)

        fig, axes = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
        axes.set_xlabel(self.train_data.columns[0], fontsize=self.FS)
        axes.set_ylabel(self.train_data.columns[1], fontsize=self.FS)

        cmap_cycle = cycle(CMAP_COLORS)
        m_color_cycle = cycle(MARKER_COLORS)
        m_style_cycle = cycle(MARKER_STYLES)

        pred_proba = None
        pred_class = None
        levels = np.arange(0.0, 1.05, 0.05)

        if contour_on:
            self._ensure_fitted_for_plot(require_predict_proba=True)
            pred_proba = self.model.predict_proba(self._mesh_entries)
            pred_class = np.argmax(pred_proba, axis=1)
            train_score = self.model.score(self._train_xy, self.train_target)

            axes.set_facecolor("k")
            axes.text(
                1.04,
                0.05,
                f"Train\nScore:\n{100 * train_score:.2f}%",
                verticalalignment="center",
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

            axes.scatter(
                self._x_values[self._class_masks[current_class]],
                self._y_values[self._class_masks[current_class]],
                s=100,
                c=next(m_color_cycle),
                marker=next(m_style_cycle),
                edgecolor="k",
                zorder=2,
                label=current_class,
            )

        axes.legend(
            loc="center left",
            bbox_to_anchor=(1.04, 0.5),
            title="Class",
            borderaxespad=0,
            borderpad=0,
            handletextpad=1.0,
            handlelength=0.0,
            alignment="left",
            fontsize=self.FS,
            title_fontsize=self.FS,
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
        mode: Literal["confusion", "error"] = "confusion",
        fig_size: Tuple[int, int] = (16, 9),
    ) -> Optional[plt.Figure]:
        """
        Plot raw and normalized confusion/error matrices.
        """
        self._ensure_fitted_for_plot(require_predict_proba=False)
        predictions = self._get_train_predictions()
        if mode == "confusion":
            sample_weight = None
            cmap: str = "Greys"
        elif mode == "error":
            sample_weight = predictions != self.train_target
            cmap = "Reds"
        else:
            raise ValueError("mode must be either 'confusion' or 'error'")

        with plt.rc_context({"font.size": self.FS}):
            fig, axes = plt.subplots(1, 3, figsize=fig_size, tight_layout=True, sharey=True)

            ConfusionMatrixDisplay.from_predictions(
                self.train_target,
                predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sample_weight,
                normalize=None,
                colorbar=False,
                cmap=cmap,
                ax=axes[0],
            )
            axes[0].set_title("Raw Counts")

            ConfusionMatrixDisplay.from_predictions(
                self.train_target,
                predictions,
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
            axes[1].set_ylabel(None)

            ConfusionMatrixDisplay.from_predictions(
                self.train_target,
                predictions,
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
            axes[2].set_ylabel(None)

        if return_fig:
            return fig
        return None
