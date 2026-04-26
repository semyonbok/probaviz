from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
from typing import Mapping, Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)

MIN_CLASS_SAMPLES = 10
MAX_CLASS_SAMPLES = 150
DEFAULT_CLASS_SAMPLES = 50
DEFAULT_RANDOM_STATE = 17


@dataclass(frozen=True)
class SyntheticSpec:
    key: str
    label: str
    min_classes: int
    max_classes: int
    help_text: str
    fixed_classes: int | None = None

    @property
    def default_n_classes(self) -> int:
        if self.fixed_classes is not None:
            return self.fixed_classes
        return min(3, self.max_classes)


SYNTHETIC_SPECS: dict[str, SyntheticSpec] = {
    spec.label: spec
    for spec in (
        SyntheticSpec(
            key="classification",
            label="Classification",
            min_classes=2,
            max_classes=4,
            help_text=(
                "General-purpose classification data with tunable class separation, "
                "label noise, and cluster structure."
            ),
        ),
        SyntheticSpec(
            key="blobs",
            label="Blobs",
            min_classes=2,
            max_classes=6,
            help_text=(
                "Gaussian clusters with direct control over class count and cluster spread."
            ),
        ),
        SyntheticSpec(
            key="gaussian_quantiles",
            label="Gaussian Quantiles",
            min_classes=2,
            max_classes=6,
            help_text=(
                "Classes are formed by concentric quantile regions of a single Gaussian."
            ),
        ),
        SyntheticSpec(
            key="circles",
            label="Circles",
            min_classes=2,
            max_classes=2,
            fixed_classes=2,
            help_text="Two concentric circles for binary nonlinear classification.",
        ),
        SyntheticSpec(
            key="moons",
            label="Moons",
            min_classes=2,
            max_classes=2,
            fixed_classes=2,
            help_text="Two interleaving half-moons for binary nonlinear classification.",
        ),
    )
}


def get_synthetic_spec(label: str) -> SyntheticSpec:
    try:
        return SYNTHETIC_SPECS[label]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic dataset method: {label}") from exc


def get_synthetic_labels() -> list[str]:
    return list(SYNTHETIC_SPECS.keys())


def deserialize_synthetic_params(payload: str) -> dict[str, object]:
    params = json.loads(payload)
    class_counts = params.get("class_counts")
    if isinstance(class_counts, Mapping):
        params["class_counts"] = {int(class_id): count for class_id, count in class_counts.items()}
    return params


def normalize_class_counts(
    n_classes: int,
    class_counts: Mapping[int, int] | list[int] | tuple[int, ...],
) -> list[int]:
    if not isinstance(n_classes, Integral) or isinstance(n_classes, bool):
        raise TypeError("n_classes must be an integer")

    n_classes_int = int(n_classes)
    if n_classes_int < 2:
        raise ValueError("n_classes must be at least 2")

    if isinstance(class_counts, Mapping):
        keys = set(class_counts.keys())
        expected = set(range(n_classes_int))
        if keys != expected:
            raise ValueError("class_counts must define one entry for each class label")
        ordered_counts = [class_counts[class_id] for class_id in range(n_classes_int)]
    else:
        ordered_counts = list(class_counts)
        if len(ordered_counts) != n_classes_int:
            raise ValueError("class_counts must define one entry for each class label")

    normalized: list[int] = []
    for count in ordered_counts:
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise TypeError("class counts must be integers")
        count_int = int(count)
        if not MIN_CLASS_SAMPLES <= count_int <= MAX_CLASS_SAMPLES:
            raise ValueError(
                f"class counts must be between {MIN_CLASS_SAMPLES} and {MAX_CLASS_SAMPLES}"
            )
        normalized.append(count_int)

    return normalized


def validate_random_state(random_state: int | None) -> int | None:
    if random_state is None:
        return None
    if not isinstance(random_state, Integral) or isinstance(random_state, bool):
        raise TypeError("random_state must be a non-negative integer or None")
    random_state_int = int(random_state)
    if random_state_int < 0:
        raise ValueError("random_state must be a non-negative integer or None")
    return random_state_int


def build_synthetic_dataset(
    method_label: str, params: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    spec = get_synthetic_spec(method_label)
    random_state = validate_random_state(params.get("random_state"))  # type: ignore[arg-type]
    requested_n_classes = int(params["n_classes"])

    if spec.fixed_classes is not None and requested_n_classes != spec.fixed_classes:
        raise ValueError(f"{method_label} supports exactly {spec.fixed_classes} classes")

    n_classes = spec.fixed_classes or requested_n_classes

    if not spec.min_classes <= n_classes <= spec.max_classes:
        raise ValueError(
            f"{method_label} supports between {spec.min_classes} and {spec.max_classes} classes"
        )

    class_counts = normalize_class_counts(
        n_classes=n_classes,
        class_counts=params["class_counts"],  # type: ignore[arg-type]
    )
    generation_seed = random_state
    total_samples = MAX_CLASS_SAMPLES * n_classes

    if spec.key == "blobs":
        cluster_std = float(params["cluster_std"])
        center_box = (float(params["center_box_min"]), float(params["center_box_max"]))
        if center_box[0] >= center_box[1]:
            raise ValueError("center_box_min must be smaller than center_box_max")
        X, y = make_blobs(
            n_samples=[MAX_CLASS_SAMPLES] * n_classes,
            centers=None,
            n_features=2,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=generation_seed,
        )
    elif spec.key == "classification":
        n_clusters_per_class = int(params["n_clusters_per_class"])
        if n_classes * n_clusters_per_class > 4:
            raise ValueError(
                "make_classification with exactly 2 features requires "
                "n_classes * n_clusters_per_class <= 4"
            )
        X, y = make_classification(
            n_samples=total_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=float(params["class_sep"]),
            flip_y=float(params["flip_y"]),
            random_state=generation_seed,
        )
    elif spec.key == "gaussian_quantiles":
        X, y = make_gaussian_quantiles(
            n_samples=total_samples,
            n_features=2,
            n_classes=n_classes,
            cov=float(params["cov"]),
            random_state=generation_seed,
        )
    elif spec.key == "circles":
        X, y = make_circles(
            n_samples=total_samples,
            noise=float(params["noise"]),
            factor=float(params["factor"]),
            random_state=generation_seed,
        )
    elif spec.key == "moons":
        X, y = make_moons(
            n_samples=total_samples,
            noise=float(params["noise"]),
            random_state=generation_seed,
        )
    else:
        raise AssertionError(f"Unsupported synthetic method: {spec.key}")

    balanced_X, balanced_y = rebalance_exact_class_counts(
        X=X,
        y=np.asarray(y, dtype=int),
        class_counts=class_counts,
        random_state=generation_seed,
    )
    return (
        pd.DataFrame(balanced_X, columns=["Feature 1", "Feautre 2"]),
        pd.Series(balanced_y).map({i: f"class_{i}" for i in range(n_classes)})
    )


def rebalance_exact_class_counts(
    *,
    X: np.ndarray,
    y: np.ndarray,
    class_counts: list[int],
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    selected_indices: list[np.ndarray] = []

    for class_id, requested_count in enumerate(class_counts):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < requested_count:
            raise ValueError(f"Generated dataset does not contain enough samples for class {class_id}")
        chosen = rng.permutation(class_indices)[:requested_count]
        selected_indices.append(chosen)

    indices = np.concatenate(selected_indices)
    indices = rng.permutation(indices)
    return X[indices], y[indices]
