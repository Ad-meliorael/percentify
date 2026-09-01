# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CHANGELOG.md` to track changes across versions ([#36](https://github.com/Ad-meliorael/percentify/issues/36))

---

## [1.0.2] – 2026-07-13

### Added
- `correlate()` – returns Pearson / Spearman / Kendall correlation with automatic p-value formatting.
- `skew_report()` – scans a DataFrame for skewed columns and suggests appropriate transformations.
- `bootstrap_ci()` – computes a bootstrap confidence interval for any statistic.
- `permutation_test()` – non-parametric permutation test for two-sample comparisons.
- `effect_size()` – computes Cohen's d (continuous) and Cohen's h (binary) per group pair.
- `pca_loadings()` – returns the feature loading matrix alongside the existing `pca_variance()` summary.
- `imbalance()` – reports class-balance ratios for a categorical column.
- `profiling.py` / `ProfileReport` – automated data-quality profiler that surfaces missingness, constants, ID-like columns, duplicate rows, dtype mismatches, high cardinality, sentinel values, collinearity, and potential target-leakage findings.

### Fixed
- `correlate()`: small p-values no longer round to `0.00`; they are now displayed in scientific notation (e.g. `3.45e-12`). ([PR #35](https://github.com/Ad-meliorael/percentify/pull/35))
- `split()`: rounding no longer silently drops or gains units when splitting a total into weighted parts. ([PR #32](https://github.com/Ad-meliorael/percentify/pull/32))
- General bug fixes across `stats.py` to stabilise the new functions.

### Changed
- Full Polars compatibility enforced on every function via the `@_backend_aware` decorator — all functions now accept and return Polars objects transparently without any manual conversion.
- Python version constraint updated to `>=3.10`; confirmed support for 3.10 – 3.13.
- Added `scipy` as a formal dependency.

---

## [1.0.0] – 2026-07-09

### Added
- `difference()` – returns the absolute and percentage difference between two values or Series.
- `split()` – splits a total into weighted parts that always sum correctly.
- `display()` – formats a number as a percentage string (e.g. `display(0.4)` → `"40%"`).
- `vif()` – Variance Inflation Factor for multicollinearity detection.
- `pca_variance()` – cumulative variance explained per principal component.
- MkDocs-based documentation site with API reference pages.

### Changed
- Major internal restructuring: all functions consolidated into `stats.py`.
- Removed legacy math-only helpers that duplicated standard library behaviour.
- `pca_variance()` renamed from `variance_explained` for readability.
- `change()` now accepts Series and DataFrames in addition to scalar values.
- Project version bumped to `1.0.0`, signalling a stable public API.

### Fixed
- Bugs in `change()` for edge-case inputs (zero baseline, mixed-sign values).

---

## [0.3.0] – 2026-05-09

### Added
- `outliers()` – IQR-based outlier detection returning flagged rows.
- `cv()` – coefficient of variation for Series and DataFrames.
- `missing()` – missing-value summary with percentages per column.
- `imbalance()` (preliminary) – class-count and ratio report for categorical data.

### Changed
- Public method names cleaned up for consistency (e.g. snake_case throughout).
- Project description in `pyproject.toml` updated.
- README rewritten to reflect the full current API.

---

## [0.2.0] – 2026-04-10

### Added
- Additional percentage helper utilities.
- Initial `display()` helper (hardcoded suffix, multiply option).

### Changed
- Package description revised for clarity.
- README formatting and wording improved.

---

## [0.1.4] – 2025-12-19

### Added
- Initial public release.
- `change()` – percentage change between two numeric values.
- `vif()` – early implementation of VIF-based multicollinearity check.
- Basic `README.md` and `pyproject.toml` scaffold.

---

[Unreleased]: https://github.com/Ad-meliorael/percentify/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/Ad-meliorael/percentify/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/Ad-meliorael/percentify/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/Ad-meliorael/percentify/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ad-meliorael/percentify/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/Ad-meliorael/percentify/releases/tag/v0.1.4
