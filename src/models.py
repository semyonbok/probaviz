# Model registry
from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
    "K Nearest Neighbors": Model(
        factory=lambda: KNeighborsClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.knc_widgets(hp_desc)
    ),
    "Decision Tree": Model(
        factory=lambda: DecisionTreeClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.dtc_widgets(hp_desc)
    ),
    "Random Forest": Model(
        factory=lambda: RandomForestClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.rfc_widgets(hp_desc)
    ),
    "Gradient Boosting": Model(
        factory=lambda: GradientBoostingClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.gbc_widgets(hp_desc)
    )
}
