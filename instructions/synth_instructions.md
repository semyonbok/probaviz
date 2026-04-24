## Synthetic Dataset Generators

### Summary
The synthetic dataset feature is implemented for the existing single-label, 2D ProbaViz flow.

Supported `sklearn.datasets` generators:
- `make_classification`
- `make_blobs`
- `make_gaussian_quantiles`
- `make_circles`
- `make_moons`

Excluded:
- `make_multilabel_classification` because the app is single-label end to end
- `make_hastie_10_2` because the app generates exactly 2 numerical features

### Product Rules
- All synthetic datasets must have exactly 2 generated numerical features.
- Users specify sample counts separately for each class.
- Each class count must be between `10` and `150`.
- Class labels are contiguous integers starting at `0`.
- Generation is deterministic via explicit `random_state`.

### Generator Constraints
- `make_blobs`: supports `2..6` classes.
- `make_gaussian_quantiles`: supports `2..6` classes.
- `make_classification`: supports `2..4` classes in practice because the app uses exactly 2 features and enforces `n_classes * n_clusters_per_class <= 4`.
- `make_circles`: fixed to 2 classes.
- `make_moons`: fixed to 2 classes.

### Generation Strategy
- Synthetic generation lives in `src/synthetic.py`.
- Widget collection lives in `src/widgets.py`.
- `app.py` passes a JSON-serialized synthetic config into `process_synth(...)`.
- Deserialization must restore integer keys for `class_counts`.

Fixed-size generation rule:
- Generate a raw dataset with `150 * n_classes` samples.
- For `make_blobs`, use `n_samples=[150] * n_classes`.
- For binary fixed-class generators, this yields `300` raw samples.
- After generation, deterministically downsample each class to the requested count using the same `random_state`.

Failure rule:
- If a generated dataset does not contain enough samples for any requested class, raise a clear error.
- There is no retry or adaptive over-generation loop.

### UI Rules
- Synthetic methods are first-class dataset sources in the sidebar.
- Each method renders only its own curated parameter controls.
- Common controls:
  - per-class sample counts
- Method-specific controls:
  - `make_classification`: `n_classes`, `class_sep`, `flip_y`, `n_clusters_per_class`
  - `make_blobs`: `n_classes`, `cluster_std`, `center_box_min`, `center_box_max`
  - `make_gaussian_quantiles`: `n_classes`, `cov`
  - `make_circles`: `noise`, `factor`
  - `make_moons`: `noise`
- For `make_classification`, when `n_classes > 2`, `n_clusters_per_class` must be forced to `1` and disabled in the UI to avoid invalid Streamlit state and invalid sklearn parameter combinations.
- For fixed-binary methods, always render only `Class 0 samples` and `Class 1 samples`.

### Testing Expectations
- Each supported method returns exactly 2 feature columns.
- Generated datasets must match the requested per-class counts exactly.
- Generation must be reproducible for identical config plus `random_state`.
- Invalid `class_counts` configs must fail validation cleanly.
- Fixed-binary generators must reject invalid `n_classes`.
- Synthetic config JSON round-trip must preserve integer class labels in `class_counts`.
