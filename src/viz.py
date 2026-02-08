"""
Allows examining the classification performance of a supervised ML model.
"""
# %% set-up
from itertools import cycle
from typing import Any, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.base import is_classifier

plt.style.use("seaborn-v0_8-ticks")


# %% the class
class ProbaViz():
    """
    Visualizes class probabilities computed by a supervised ML model trained on
    classified samples with two numerical features.
    Supports more than two classes.

    ...

    Parameters
    ----------
    model : classifier
        An instance of a supervised ML model implementation with `fit`,
        `predict`, `predict_proba` `score`, `set_params` methods.
    train_data : pd.DataFrame of shape (n_samples, n_features)
        Data frame containing two numerical features used for model training.
    train_target : array-like of shape (n_samples,)
        Classes of samples from `train_data`.
    features : array-like of shape (2,)
        A sequence listing two numerical features to be used for model
        training; contains either `str` or `int` referring to either
        feature names or indexes in `train_data`.
    grid_res : tuple of int, optional, default=(100, 100)
        Resolution of the grid for plotting decision boundaries and
        probabilities.

    Methods
    -------
    set_model(model)
        Sets the supervised ML model, performance of which is examined.
    set_data(train_data, train_target, features)
        Sets data attributes related to the training dataset.
    plot()
        Draws scatter plot displaying the training data discriminated by class
        and contour plots with the height values corresponding to class
        probabilities computed by the set supervised ML model; contours are
        discriminated by class with the highest probability at a given pair of
        feature values; borders between contours show the decision boundaries.
    replot()
        Tuned for a widget use; adjusts passed hyperparameters  of the set
        supervised ML model and calls plot method.
        **Warning:** changes hyperparameters of the set model.
    plot_confusion_matrices()
        Plots two confusion matrices: one showing raw counts and the other
        showing row-normalized values
    plot_error_matrices()
        Plots two error matrices: one normalized by predicted values (columns)
        and another normalized by true class (rows)
    """
    FS = 22  # "xx-large"

    def __init__(
            self, model: Any,
            train_data: pd.DataFrame,
            train_target: Sequence,
            features: Sequence[str | int],
            grid_res: Tuple[int, int] = (100, 100)
    ):
        self._define_utilities()
        self.set_data(train_data, train_target, features, grid_res)
        self.set_model(model)

    def _define_utilities(self):
        self._asserts = {
            "a1": "data & target must have the same length",
            "a2": "two features must be specified for visualization",
            "a3": "two integers must be used to specify grid resolution",
            "a4": "feature {} is not numeric"
        }

        self._cmap_colors = [
            "Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"
        ]

        self._m_colors = [
            "tab:blue", "tab:orange", "tab:green",
            "tab:red", "tab:purple", "tab:grey"
        ]

        self._m_styles = ["o", "s", "P", "v", "D", "X"]

    def set_model(self, new_model: Any):
        """
        Sets the supervised ML model, performance of which is examined.

        Parameters
        ----------
        new_model : classifier
            An instance of a supervised ML model implementation with `fit`,
            `predict`, `predict_proba` methods and `class_` data
            attribute assigned during `fit` call.

        Returns
        -------
        `None`.
        """
        if new_model is not None:
            if is_classifier(new_model):
                new_model.fit(self.train_data.values, self.train_target)
        self.model = new_model

    def set_data(
            self,
            train_data: pd.DataFrame,
            train_target: Sequence,
            features: Sequence[str | int],
            grid_res: Tuple[int, int] = (100, 100)
    ):
        """
        Sets data attributes related to the training dataset.

        Parameters
        ----------
        train_data : pd.DataFrame of shape (n_samples, n_features)
            Data frame containing two numerical features used for model
            training.
        train_target : array-like of shape (n_samples,)
            Classes of samples from `train_data`.
        features : array-like of shape 2
            A sequence listing two numerical features to be used for model
            training; contains either `str` or `int` referring to either
            feature names or feature indexes in `train_data`.
        grid_res : tuple of int, optional, default=(100, 100)
            Resolution of the grid for plotting decision boundaries and
            probabilities.

        Returns
        -------
        `None`.
        """
        # input validation
        assert train_data.shape[0] == len(train_target), self._asserts["a1"]
        assert len(features) == 2, self._asserts["a2"]
        assert len(grid_res) == 2 and all(
            [isinstance(res, int) for res in grid_res]
        ), self._asserts["a3"]

        try:
            train_data = train_data.iloc[:, features]
        except IndexError:
            train_data = train_data.loc[:, features]  # KeyError possible

        for feature in train_data.columns:
            assert (
                pd.api.types.is_numeric_dtype(train_data[feature])
                and not pd.api.types.is_bool_dtype(train_data[feature])  # noqa
            ), self._asserts["a4"].format(feature)

        # define new entries for contour, ensure all data points will be seen
        coord_dict = {}
        for axis, feature in zip(["x", "y"], [0, 1]):
            offset = np.ptp(train_data.iloc[:, feature].values) / 100
            coord_dict[axis] = np.linspace(
                train_data.iloc[:, feature].min() - offset,
                train_data.iloc[:, feature].max() + offset,
                grid_res[feature]
            )
        coord_dict["x"], coord_dict["y"] = np.meshgrid(
            coord_dict["x"], coord_dict["y"]
        )

        # set data attributes
        self._coord_dict = coord_dict
        self._mesh_entries = np.append(
            coord_dict["x"].reshape(coord_dict["x"].size, 1),
            coord_dict["y"].reshape(coord_dict["y"].size, 1),
            axis=1
        )
        self.train_data = train_data
        self.features = train_data.columns.values
        self.train_target = train_target
        self.classes = np.unique(train_target)

    def plot(
            self, contour_on: bool = True, return_fig: bool = False,
            fig_size: Tuple[int, int] = (12, 6)
    ) -> Optional[plt.Figure]:
        """
        Draws scatter plot displaying the training data discriminated by class
        and contour plots with the height values corresponding to class
        probabilities computed by the set supervised ML model; contours are
        discriminated by class with the highest probability at a given pair of
        feature values; borders between contours show the decision boundaries.

        Parameters
        ----------
        contour_on : bool, optional
           If `True`, contour plots are drawn in addition to scatter plot; if
           `False`, only scatter plot is drawn; the default is `True`.
        return_fig : bool, optional
            If `True`, returns a `matplotlib.figure.Figure` instance; if
            `False`, returns `None`; the default is `False`.
        fig_size : tuple of int, optional, default=(12, 6)
            A tuple specifying the width and height of the figure in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The generated figure if `return_fig` is `True`; otherwise `None`.
        """
        # figure canvas and appearance
        fig, axes = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
        axes.set_xlabel(self.train_data.iloc[:, 0].name, fontsize=self.FS)
        axes.set_ylabel(self.train_data.iloc[:, 1].name, fontsize=self.FS)

        # engage utilities and combine data with target for scatter plot
        cmap_cycle = cycle(self._cmap_colors)
        m_color_cycle = cycle(self._m_colors)
        m_style_cycle = cycle(self._m_styles)

        # fit model, get predictions and train score
        if contour_on:
            pred_proba = self.model.predict_proba(self._mesh_entries)
            pred_class = np.argmax(pred_proba, axis=1)
            train_score = self.model.score(
                self.train_data.values, self.train_target
            )

            axes.set_facecolor("k")  # for better decision boundary display
            axes.text(
                1.04, 0.05, f"Train\nScore:\n{100 * train_score:.2f}%",
                verticalalignment="center", horizontalalignment="left",
                transform=axes.transAxes, fontsize=self.FS,
            )

        # iteratively plot contours and data points for every class
        for index, _ in enumerate(self.classes):
            if contour_on:
                levels = np.arange(0., 1.05, 0.05)
                current_cmap = next(cmap_cycle)
                class_proba = np.where(
                    pred_class == index, pred_proba[:, index], np.nan
                )

                if ~np.isnan(class_proba).all():  # skip contour if no class
                    # main filled contour
                    cs0 = axes.contourf(
                        self._coord_dict["x"], self._coord_dict["y"],
                        class_proba.reshape(self._coord_dict["x"].shape),
                        cmap=current_cmap, alpha=1, vmin=0, vmax=1,
                        levels=levels
                    )

                    # isolines
                    if cs0.get_cmap().name == "Greys":
                        current_icolor = "w"
                    else:
                        current_icolor = "k"
                    cs1 = axes.contour(
                        cs0, levels=levels[::4], colors=current_icolor
                    )
                    axes.clabel(cs1, levels[::4], inline=True, fontsize=self.FS)

            # data points and legend
            axes.scatter(
                self.train_data.columns[0], self.train_data.columns[1],
                data=self.train_data.loc[self.train_target == self.classes[index]],
                c=next(m_color_cycle), marker=next(m_style_cycle),
                edgecolor="k", zorder=2, label=self.classes[index]
            )

        axes.legend(
            loc="center left", bbox_to_anchor=(1.04, .5), title="Class",
            borderaxespad=0, borderpad=0, handletextpad=1., handlelength=0.,
            alignment="left", fontsize=self.FS, title_fontsize=self.FS,
        )
        axes.tick_params(axis='both', which='major', labelsize=self.FS)

        if return_fig:
            return fig
        return None

    def replot(self, contour_on: bool = True, **params):
        """
        Tuned for a widget use; adjusts passed hyperparameters  of the set
        supervised ML model and calls `plot` method.
        **Warning:** changes hyperparameters of the set model.

        Parameters
        ----------
        contour_on : bool, optional
            If `True`, contour plots are drawn in addition to scatter plot;
            if `False`, only scatter plot is drawn; the default is `True`.
        **params : kwargs
            Additional keyword arguments representing hyperparameters specific
            to the set supervised ML model.

        Returns
        -------
        `None`.
        """
        self.set_model(self.model.set_params(**params))
        self.plot(contour_on=contour_on)

    def plot_confusion_matrices(
            self, return_fig: bool = False, fig_size: Tuple[int, int] = (16, 9)
    ):
        """
        Plots three confusion matrices: the left one shows raw counts, the
        middle one shows row-normalized values (i.e., normalized by true class)
        and the right one shows column-normalized values (i.e., normalized by
        predictions).

        Parameters
        ----------
        return_fig : bool, optional
            If `True`, returns a `matplotlib.figure.Figure` instance; if
            `False`, returns `None`; the default is `False`.
        fig_size : tuple of int, optional, default=(16, 9)
            A tuple specifying the width and height of the figure in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The generated figure if `return_fig` is `True`; otherwise `None`.
        """
        # TODO prevent from plotting if model is `None`
        predictions = self.model.predict(self.train_data.values)

        with plt.rc_context({'font.size': self.FS}):
            fig, axes = plt.subplots(
                1, 3, figsize=fig_size, tight_layout=True, sharey=True
            )

            ConfusionMatrixDisplay.from_predictions(
                self.train_target, predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                normalize=None,
                colorbar=False,
                cmap="Greys",
                ax=axes[0]
            )
            axes[0].set_title("Raw Counts")

            ConfusionMatrixDisplay.from_predictions(
                self.train_target, predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                normalize="true",
                colorbar=False,
                cmap="Greys",
                ax=axes[1],
                values_format=".0%",
                im_kw={"vmin": 0, "vmax": 1}
            )
            axes[1].set_title("Normalized by Row")
            axes[1].set_ylabel(None)

            ConfusionMatrixDisplay.from_predictions(
                self.train_target, predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                normalize="pred",
                colorbar=False,
                cmap="Greys",
                ax=axes[2],
                values_format=".0%",
                im_kw={"vmin": 0, "vmax": 1}
            )
            axes[2].set_title("Normalized by Column")
            axes[2].set_ylabel(None)

        if return_fig:
            return fig

    def plot_error_matrices(
            self, return_fig: bool = False, fig_size: Tuple[int, int] = (16, 9)
    ):
        """
        Plots three error matrices: the left one shows raw counts, the middle
        one is normalized by true class (rows) and the right one is normalized
        by predicted values (columns). These matrices highlight the
        distribution of prediction errors made by the supervised ML model,
        helping to identify where the model struggles the most.

        Parameters
        ----------
        return_fig : bool, optional
            If `True`, returns a `matplotlib.figure.Figure` instance; if
            `False`, returns `None`; the default is `False`.
        fig_size : tuple of int, optional, default=(16, 9)
            A tuple specifying the width and height of the figure in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The generated figure if `return_fig` is `True`; otherwise `None`.
        """
        # TODO prevent from plotting if model is `None`
        predictions = self.model.predict(self.train_data.values)
        sw = predictions != self.train_target

        with plt.rc_context({'font.size': self.FS}):
            fig, axes = plt.subplots(
                1, 3, figsize=fig_size, tight_layout=True, sharey=True
            )

            ConfusionMatrixDisplay.from_predictions(
                self.train_target, predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sw,
                normalize=None,
                colorbar=False,
                cmap="Reds",
                ax=axes[0],
            )
            axes[0].set_title("Raw Counts")

            ConfusionMatrixDisplay.from_predictions(
                self.train_target, predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sw,
                normalize="true",
                colorbar=False,
                cmap="Reds",
                ax=axes[1],
                values_format=".0%",
                im_kw={"vmin": 0, "vmax": 1}
            )
            axes[1].set_title("Normalized by Row")
            axes[1].set_ylabel(None)

            ConfusionMatrixDisplay.from_predictions(
                self.train_target, predictions,
                display_labels=self.classes,
                xticks_rotation="vertical",
                sample_weight=sw,
                normalize="pred",
                colorbar=False,
                cmap="Reds",
                ax=axes[2],
                values_format=".0%",
                im_kw={"vmin": 0, "vmax": 1}
            )
            axes[2].set_title("Normalized by Column")
            axes[2].set_ylabel(None)

        if return_fig:
            return fig
