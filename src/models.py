# Model registry
from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    NearestCentroid
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

import src.widgets as wdg


@dataclass(frozen=True)
class Model():
    # TODO add a dedicated description to replace docstring in info pane
    factory: Callable[[], BaseEstimator]
    widgets: Callable[..., Dict]


MODELS: dict[str, Model] = {
    "Logistic Regression": Model(
        factory=lambda: LogisticRegression(),
        widgets=lambda *, hp_desc, **_: wdg.lr_widgets(hp_desc)
    ),
    "SGDClassifier": Model(
        factory=lambda: SGDClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.sgdc_widgets(hp_desc)
    ),
    "Linear Discriminant Analysis": Model(
        factory=lambda: LinearDiscriminantAnalysis(),
        widgets=lambda *, hp_desc, **_: wdg.lda_widgets(hp_desc)
    ),
    "Quadratic Discriminant Analysis": Model(
        factory=lambda: QuadraticDiscriminantAnalysis(),
        widgets=lambda *, hp_desc, **_: wdg.qda_widgets(hp_desc)
    ),
    "K Nearest Neighbors": Model(
        factory=lambda: KNeighborsClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.knc_widgets(hp_desc)
    ),
    "Radius Neighbors Classifier": Model(
        factory=lambda: RadiusNeighborsClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.rnc_widgets(hp_desc)
    ),
    "Nearest Centroid": Model(
        factory=lambda: NearestCentroid(),
        widgets=lambda *, hp_desc, **_: wdg.nc_widgets(hp_desc)
    ),
    "Decision Tree": Model(
        factory=lambda: DecisionTreeClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.dtc_widgets(hp_desc)
    ),
    "Extra Tree Classifier": Model(
        factory=lambda: ExtraTreeClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.etc_widgets(hp_desc)
    ),
    "Extra Trees Classifier": Model(
        factory=lambda: ExtraTreesClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.etsc_widgets(hp_desc)
    ),
    "Random Forest": Model(
        factory=lambda: RandomForestClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.rfc_widgets(hp_desc)
    ),
    "Bagging Classifier": Model(
        factory=lambda: BaggingClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.bc_widgets(hp_desc)
    ),
    "AdaBoosting": Model(
        factory=lambda: AdaBoostClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.abc_widgets(hp_desc)
    ),
    "Gradient Boosting": Model(
        factory=lambda: GradientBoostingClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.gbc_widgets(hp_desc)
    ),
    "Histogram Gradient Boosting": Model(
        factory=lambda: HistGradientBoostingClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.hgbc_widgets(hp_desc)
    )
}
