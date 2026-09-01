import pytest
import numpy as np

pl = pytest.importorskip("polars")
import pandas as pd  # noqa: E402
from percentify import (  # noqa: E402
    change, vif, missing, cv, outliers,
    pca_variance, pca_loadings, imbalance, difference, split, display,
    correlate, skew_report, bootstrap_ci, permutation_test, effect_size,
)


# ===== DataFrame in -> polars DataFrame out =====

def test_vif_polars():
    np.random.seed(0)
    base = np.random.randn(80)
    df = pl.DataFrame({
        "a": base,
        "b": base * 2 + np.random.randn(80) * 0.1,
        "c": np.random.randn(80),
    })
    result = vif(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "VIF"]
    assert set(result["feature"].to_list()) == {"a", "b", "c"}


def test_missing_polars_values():
    df = pl.DataFrame({"salary": [1.0, None, 3.0, None], "age": [1.0, 2.0, None, 4.0]})
    result = missing(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["column", "missing_pct", "has_missing"]
    d = dict(zip(result["column"].to_list(), result["missing_pct"].to_list()))
    assert d["salary"] == 50.0
    assert d["age"] == 25.0
    flags = dict(zip(result["column"].to_list(), result["has_missing"].to_list()))
    assert flags == {"salary": True, "age": True}


def test_cv_polars_dataframe():
    result = cv(pl.DataFrame({"a": [10.0, 20, 30], "b": [1.0, 2, 3]}))
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "cv"]


def test_outliers_polars_dataframe():
    df = pl.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 100], "b": [1.0, 2, 3, 4, 5, 6, 7]})
    result = outliers(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "outlier_pct"]


def test_pca_variance_polars():
    np.random.seed(0)
    base = np.random.randn(80)
    df = pl.DataFrame({
        "a": base + np.random.randn(80) * 0.3,
        "b": base + np.random.randn(80) * 0.3,
        "c": np.random.randn(80),
    })
    result = pca_variance(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["component", "variance_explained", "cumulative"]


def test_pca_loadings_polars():
    np.random.seed(1)
    base = np.random.randn(80)
    df = pl.DataFrame({
        "a": base + np.random.randn(80) * 0.05,
        "b": base + np.random.randn(80) * 0.05,
        "c": np.random.randn(80),
    })
    result = pca_loadings(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "PC1", "PC2", "PC3"]


# ===== Series in -> scalar out =====

def test_cv_polars_series_returns_float():
    assert isinstance(cv(pl.Series([10.0, 20, 30, 40, 50])), float)


def test_outliers_polars_series_returns_float():
    assert isinstance(outliers(pl.Series([1.0, 2, 3, 4, 5, 6, 100])), float)


# ===== Series in -> polars Series out =====

def test_change_polars_period_over_period():
    result = change(pl.Series([100.0, 150, 90, 135]))
    assert isinstance(result, pl.Series)
    vals = result.to_list()
    assert vals[1] == 50.0
    assert vals[2] == -40.0


def test_change_polars_two_columns():
    result = change(pl.Series([100.0, 200, 50]), pl.Series([150.0, 150, 100]))
    assert isinstance(result, pl.Series)
    assert result.to_list() == [50.0, -25.0, 100.0]


def test_difference_polars_two_columns():
    result = difference(pl.Series([10.0, 50]), pl.Series([20.0, 50]))
    assert isinstance(result, pl.Series)
    assert result.to_list() == [66.67, 0.0]


def test_split_polars_series():
    result = split(200, pl.Series([1, 3]))
    assert isinstance(result, pl.Series)
    assert result.to_list() == [50.0, 150.0]


def test_display_polars_series():
    result = display(pl.Series([0.25, 0.5]), multiply=True)
    assert isinstance(result, pl.Series)
    assert result.to_list() == ["25.0%", "50.0%"]


# ===== imbalance: polars DataFrame + summary preserved =====

def test_imbalance_polars_with_summary():
    result = imbalance(pl.Series(["No"] * 850 + ["Yes"] * 150))
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["class", "count", "pct"]
    summary = result.attrs["summary"]
    assert summary["majority_class"] == "No"
    assert summary["minority_class"] == "Yes"
    assert summary["imbalance_ratio"] == 5.67


# ===== inferential functions =====

def test_correlate_polars_dataframe():
    np.random.seed(0)
    base = np.random.randn(100)
    df = pl.DataFrame({"a": base, "b": base + np.random.randn(100) * 0.01, "c": np.random.randn(100)})
    result = correlate(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature_1", "feature_2", "r", "p"]


def test_correlate_polars_two_series_tuple():
    r, p = correlate(pl.Series(range(50)), pl.Series(range(50)))
    assert isinstance(r, float)
    assert r == 1.0


def test_correlate_polars_log_p_dataframe():
    np.random.seed(0)
    base = np.random.randn(200)
    df = pl.DataFrame({"a": base, "b": base * 2 + np.random.randn(200) * 0.1,
                       "c": np.random.randn(200)})
    result = correlate(df, log_p=True)
    assert isinstance(result, pl.DataFrame)          # polars in -> polars out
    assert result.columns == ["feature_1", "feature_2", "r", "p", "log10_p"]


def test_correlate_polars_log_p_two_series():
    np.random.seed(0)
    a = np.random.randn(200)
    b = 0.44 * a + np.sqrt(1 - 0.44 ** 2) * np.random.randn(200)
    r, p, log10_p = correlate(pl.Series(a), pl.Series(b), log_p=True)
    assert isinstance(log10_p, float)
    assert log10_p < 0


def test_skew_report_polars():
    np.random.seed(0)
    df = pl.DataFrame({"income": np.random.exponential(1, 300), "sym": np.random.randn(300)})
    result = skew_report(df)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "skew", "kurtosis", "outlier_pct", "suggested_transform"]


def test_bootstrap_ci_polars_series():
    result = bootstrap_ci(pl.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]), random_state=0)
    assert isinstance(result, tuple)
    assert result[0] <= result[1]


def test_permutation_test_polars():
    result = permutation_test(pl.Series([1.0, 2, 3, 4]), pl.Series([5.0, 6, 7, 8]), random_state=0)
    assert isinstance(result, float)


def test_effect_size_polars():
    np.random.seed(0)
    df = pl.DataFrame({
        "g": ["A"] * 50 + ["B"] * 50,
        "v": np.concatenate([np.random.randn(50), np.random.randn(50) + 1]),
    })
    result = effect_size(df, group="g", value="v")
    assert isinstance(result, pl.DataFrame)
    assert "cohen_d" in result.columns


# ===== regression: pandas in still returns pandas =====

def test_pandas_input_unchanged():
    result = missing(pd.DataFrame({"a": [1.0, None], "b": [1.0, 2.0]}))
    assert isinstance(result, pd.DataFrame)


def test_scalar_input_unchanged():
    assert change(100, 150) == 50.0
    assert display(0.45, multiply=True) == "45.0%"
