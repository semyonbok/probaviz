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
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    NearestCentroid
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    MultinomialNB,
    ComplementNB,
    CategoricalNB
)
# from sklearn.semi_supervised import LabelPropagation, LabelSpreading

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
    "Stochastic Gradient Descent": Model(
        factory=lambda: SGDClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.sgdc_widgets(hp_desc)
    ),
    "SVC": Model(
        factory=lambda: SVC(),
        widgets=lambda *, hp_desc, **_: wdg.svc_widgets(hp_desc)
    ),
    "NuSVC": Model(
        factory=lambda: NuSVC(),
        widgets=lambda *, hp_desc, **_: wdg.nsvc_widgets(hp_desc)
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
    "Radius Neighbors": Model(
        factory=lambda: RadiusNeighborsClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.rnc_widgets(hp_desc)
    ),
    "Nearest Centroid": Model(
        factory=lambda: NearestCentroid(),
        widgets=lambda *, hp_desc, **_: wdg.nc_widgets(hp_desc)
    ),
    "Gaussian Naive Bayes": Model(
        factory=lambda: GaussianNB(),
        widgets=lambda *, hp_desc, **_: wdg.gnb_widgets(hp_desc)
    ),
    "Bernoulli Naive Bayes": Model(
        factory=lambda: BernoulliNB(),
        widgets=lambda *, hp_desc, **_: wdg.bnb_widgets(hp_desc)
    ),
    "Multinomial Naive Bayes": Model(
        factory=lambda: MultinomialNB(),
        widgets=lambda *, hp_desc, **_: wdg.mnb_widgets(hp_desc)
    ),
    "Complement Naive Bayes": Model(
        factory=lambda: ComplementNB(),
        widgets=lambda *, hp_desc, **_: wdg.cnb_widgets(hp_desc)
    ),
    "Categorical Naive Bayes": Model(
        factory=lambda: CategoricalNB(),
        widgets=lambda *, hp_desc, **_: wdg.catnb_widgets(hp_desc)
    ),
    "Decision Tree": Model(
        factory=lambda: DecisionTreeClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.dtc_widgets(hp_desc)
    ),
    "Extra Tree": Model(
        factory=lambda: ExtraTreeClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.etc_widgets(hp_desc)
    ),
    "Extra Trees": Model(
        factory=lambda: ExtraTreesClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.etsc_widgets(hp_desc)
    ),
    "Random Forest": Model(
        factory=lambda: RandomForestClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.rfc_widgets(hp_desc)
    ),
    "Bagging": Model(
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
    ),
    "Multi-layer Perceptron": Model(
        factory=lambda: MLPClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.mlpc_widgets(hp_desc)
    ),
    "Gaussian Process": Model(
        factory=lambda: GaussianProcessClassifier(),
        widgets=lambda *, hp_desc, **_: wdg.gpc_widgets(hp_desc)
    )
}
