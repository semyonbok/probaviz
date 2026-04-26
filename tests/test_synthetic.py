from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import (  # noqa
    build_synthetic_dataset,
    deserialize_synthetic_params,
    normalize_class_counts,
)


@pytest.mark.parametrize(
    ("method_label", "params", "expected_counts"),
    [
        (
            "Blobs",
            {
                "n_classes": 3,
                "class_counts": {0: 10, 1: 40, 2: 130},
                "cluster_std": 1.0,
                "center_box_min": -10.0,
                "center_box_max": 10.0,
                "random_state": 17,
            },
            {0: 10, 1: 40, 2: 130},
        ),
        (
            "Classification",
            {
                "n_classes": 4,
                "class_counts": {0: 10, 1: 20, 2: 30, 3: 40},
                "class_sep": 1.0,
                "flip_y": 0.0,
                "n_clusters_per_class": 1,
                "random_state": 17,
            },
            {0: 10, 1: 20, 2: 30, 3: 40},
        ),
        (
            "Gaussian Quantiles",
            {
                "n_classes": 5,
                "class_counts": {0: 10, 1: 20, 2: 30, 3: 40, 4: 50},
                "cov": 1.0,
                "random_state": 17,
            },
            {0: 10, 1: 20, 2: 30, 3: 40, 4: 50},
        ),
        (
            "Circles",
            {
                "n_classes": 2,
                "class_counts": {0: 10, 1: 130},
                "noise": 0.1,
                "factor": 0.5,
                "random_state": 17,
            },
            {0: 10, 1: 130},
        ),
        (
            "Moons",
            {
                "n_classes": 2,
                "class_counts": {0: 130, 1: 10},
                "noise": 0.1,
                "random_state": 17,
            },
            {0: 130, 1: 10},
        ),
    ],
)
def test_build_synthetic_dataset_returns_exact_requested_class_counts(
    method_label,
    params,
    expected_counts,
):
    data, target = build_synthetic_dataset(method_label, params)

    assert data.shape == (sum(expected_counts.values()), 2)
    assert list(data.columns) == ["Feature 1", "Feature 2"]

    unique, counts = np.unique(target, return_counts=True)
    observed = {str(class_id): int(count) for class_id, count in zip(unique, counts)}
    expected = {f"class_{class_id}": count for class_id, count in expected_counts.items()}
    assert observed == expected


@pytest.mark.parametrize(
    ("method_label", "params"),
    [
        (
            "Classification",
            {
                "n_classes": 3,
                "class_counts": {0: 20, 1: 25, 2: 30},
                "class_sep": 1.0,
                "flip_y": 0.0,
                "n_clusters_per_class": 1,
                "random_state": 17,
            },
        ),
        (
            "Gaussian Quantiles",
            {
                "n_classes": 6,
                "class_counts": {0: 10, 1: 15, 2: 20, 3: 25, 4: 30, 5: 35},
                "cov": 1.0,
                "random_state": 17,
            },
        ),
    ],
)
def test_build_synthetic_dataset_is_reproducible(method_label, params):
    data_a, target_a = build_synthetic_dataset(method_label, params)
    data_b, target_b = build_synthetic_dataset(method_label, params)

    assert np.array_equal(data_a.to_numpy(), data_b.to_numpy())
    assert np.array_equal(target_a, target_b)


def test_normalize_class_counts_requires_exact_class_coverage():
    with pytest.raises(ValueError, match="one entry for each class label"):
        normalize_class_counts(3, {0: 10, 1: 20})


def test_normalize_class_counts_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="between 10 and 150"):
        normalize_class_counts(2, {0: 9, 1: 20})


def test_deserialize_synthetic_params_restores_integer_class_count_keys():
    params = deserialize_synthetic_params(
        '{"n_classes": 2, "class_counts": {"0": 10, "1": 20}, "noise": 0.1, "random_state": 17}'
    )

    assert params["class_counts"] == {0: 10, 1: 20}


def test_make_classification_rejects_invalid_cluster_and_class_combo():
    with pytest.raises(ValueError, match="n_classes \\* n_clusters_per_class <= 4"):
        build_synthetic_dataset(
            "Classification",
            {
                "n_classes": 3,
                "class_counts": {0: 10, 1: 10, 2: 10},
                "class_sep": 1.0,
                "flip_y": 0.0,
                "n_clusters_per_class": 2,
                "random_state": 17,
            },
        )


def test_fixed_binary_generators_reject_invalid_n_classes():
    with pytest.raises(ValueError, match="supports exactly 2 classes"):
        build_synthetic_dataset(
            "Circles",
            {
                "n_classes": 3,
                "class_counts": {0: 10, 1: 10},
                "noise": 0.1,
                "factor": 0.5,
                "random_state": 17,
            },
        )
