import pytest
import numpy as np
import pandas as pd
from percentify import (
    change, vif, missing, cv, outliers, pca_variance, pca_loadings, imbalance,
    difference, split, display, PercentifyWarning,
)


# ===== Fixtures =====

@pytest.fixture
def independent_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })


@pytest.fixture
def collinear_df():
    np.random.seed(42)
    x = np.random.randn(100)
    return pd.DataFrame({
        "x": x,
        "y": x * 2 + np.random.randn(100) * 0.01,
        "z": np.random.randn(100),
    })


def _to_map(df, key, val):
    return dict(zip(df[key], df[val]))


# ===== change =====

def test_change_two_scalars_increase():
    assert change(100, 150) == 50.0


def test_change_two_scalars_decrease():
    assert change(200, 150) == -25.0


def test_change_zero_old():
    assert change(0, 100) == 0.0


def test_change_negative_old():
    assert change(-100, -50) == 50.0


def test_change_custom_decimals():
    assert change(3, 7, 4) == 133.3333


def test_change_scalar_missing_new_raises():
    with pytest.raises(ValueError):
        change(100)


def test_change_series():
    result = change(pd.Series([100, 150, 90]))
    assert isinstance(result, pd.Series)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 50.0
    assert result.iloc[2] == -40.0


def test_change_series_custom_decimals():
    result = change(pd.Series([3, 7]), decimals=4)
    assert result.iloc[1] == 133.3333


def test_change_dataframe():
    df = pd.DataFrame({"rev": [100, 150, 90], "cost": [10, 20, 10]})
    result = change(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["rev", "cost"]
    assert result["rev"].iloc[1] == 50.0
    assert result["cost"].iloc[1] == 100.0


def test_change_dataframe_ignores_non_numeric():
    df = pd.DataFrame({"rev": [100, 150], "name": ["a", "b"]})
    assert list(change(df).columns) == ["rev"]


def test_change_non_numeric_series_warns():
    with pytest.warns(PercentifyWarning):
        result = change(pd.Series(["a", "b", "c"]))
    assert result.isna().all()


def test_change_two_columns():
    df = pd.DataFrame({"old": [100, 200, 50], "new": [150, 150, 100]})
    result = change(df["old"], df["new"])
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 50.0
    assert result.iloc[1] == -25.0
    assert result.iloc[2] == 100.0


def test_change_two_columns_zero_old():
    result = change(pd.Series([0, 100]), pd.Series([50, 150]))
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 50.0


def test_change_two_columns_non_numeric_warns():
    with pytest.warns(PercentifyWarning):
        result = change(pd.Series(["a", "b"]), pd.Series([1, 2]))
    assert result.isna().all()


# ===== vif =====

def test_vif_returns_dataframe(independent_df):
    result = vif(independent_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "VIF"]


def test_vif_returns_all_columns(independent_df):
    result = vif(independent_df)
    assert set(result["feature"]) == {"a", "b", "c"}


def test_vif_independent_features_low(independent_df):
    result = vif(independent_df)
    assert (result["VIF"] < 5.0).all()


def test_vif_collinear_features_high(collinear_df):
    vals = _to_map(vif(collinear_df), "feature", "VIF")
    assert vals["x"] > 10
    assert vals["y"] > 10


def test_vif_sorted_descending(collinear_df):
    vifs = vif(collinear_df)["VIF"].tolist()
    assert vifs == sorted(vifs, reverse=True)


def test_vif_flag_filters(independent_df):
    result = vif(independent_df, flag=5.0)
    assert result.empty


def test_vif_flag_returns_collinear(collinear_df):
    features = set(vif(collinear_df, flag=5.0)["feature"])
    assert "x" in features
    assert "y" in features
    assert "z" not in features


def test_vif_ignores_non_numeric():
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    assert set(vif(df)["feature"]) == {"a", "b"}


def test_vif_all_categorical_warns():
    df = pd.DataFrame({"city": ["NY", "LA"], "team": ["A", "B"]})
    with pytest.warns(PercentifyWarning, match="[Nn]umeric"):
        result = vif(df)
    assert result.empty
    assert list(result.columns) == ["feature", "VIF"]


def test_vif_single_numeric_warns():
    df = pd.DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    with pytest.warns(PercentifyWarning):
        result = vif(df)
    assert result.empty


def test_vif_custom_decimals(independent_df):
    for val in vif(independent_df, decimals=4)["VIF"]:
        parts = str(float(val)).split(".")
        if len(parts) == 2:
            assert len(parts[1]) <= 4


# ===== missing =====

def test_missing_returns_dataframe():
    df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
    result = missing(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["column", "missing_pct"]


def test_missing_basic():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["b"] == 40.0
    assert vals["a"] == 20.0
    assert vals["c"] == 0.0


def test_missing_sorted_descending():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3, 4, 5],
        "c": [1, 2, 3, 4, 5],
    })
    assert missing(df)["column"].tolist() == ["b", "a", "c"]


def test_missing_no_nulls():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert (missing(df)["missing_pct"] == 0.0).all()


def test_missing_all_null():
    df = pd.DataFrame({"a": [None, None, None]})
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["a"] == 100.0


def test_missing_empty_df():
    df = pd.DataFrame({"a": [], "b": []})
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["a"] == 0.0


def test_missing_includes_non_numeric():
    df = pd.DataFrame({
        "name": ["Alice", None, "Charlie"],
        "age": [25, None, 35],
    })
    vals = _to_map(missing(df), "column", "missing_pct")
    assert vals["name"] == 33.33
    assert vals["age"] == 33.33


# ===== cv =====

def test_cv_series_returns_float():
    result = cv(pd.Series([10, 20, 30, 40, 50]))
    assert isinstance(result, float)
    assert result > 0


def test_cv_dataframe_returns_dataframe():
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
    result = cv(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "cv"]


def test_cv_dataframe_values():
    df = pd.DataFrame({
        "a": [10, 20, 30, 40, 50],
        "b": [100, 100, 100, 100, 100],
    })
    vals = _to_map(cv(df), "feature", "cv")
    assert vals["a"] > 0
    assert vals["b"] == 0.0


def test_cv_zero_mean_series_warns():
    with pytest.warns(PercentifyWarning):
        result = cv(pd.Series([-1, 0, 1]))
    assert result == float("inf")


def test_cv_zero_mean_dataframe():
    df = pd.DataFrame({"a": [-1, 0, 1], "b": [10, 20, 30]})
    vals = _to_map(cv(df), "feature", "cv")
    assert vals["a"] == float("inf")
    assert vals["b"] > 0


def test_cv_non_numeric_series_warns():
    with pytest.warns(PercentifyWarning):
        result = cv(pd.Series(["cat", "dog", "bird"]))
    assert np.isnan(result)


def test_cv_ignores_non_numeric():
    df = pd.DataFrame({"a": [10, 20, 30], "name": ["x", "y", "z"]})
    assert set(cv(df)["feature"]) == {"a"}


def test_cv_custom_decimals():
    result = cv(pd.Series([10, 20, 30, 40, 50]), decimals=4)
    parts = str(result).split(".")
    if len(parts) == 2:
        assert len(parts[1]) <= 4


# ===== outliers =====

def test_outliers_series_returns_float():
    result = outliers(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]))
    assert isinstance(result, float)
    assert result > 0


def test_outliers_no_outliers():
    assert outliers(pd.Series([1, 2, 3, 4, 5])) == 0.0


def test_outliers_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    result = outliers(df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["feature", "outlier_pct"]
    vals = _to_map(result, "feature", "outlier_pct")
    assert vals["a"] > 0
    assert vals["b"] == 0.0


def test_outliers_all_nan():
    assert outliers(pd.Series([np.nan, np.nan, np.nan])) == 0.0


def test_outliers_non_numeric_series_warns():
    with pytest.warns(PercentifyWarning):
        result = outliers(pd.Series(["cat", "dog"]))
    assert np.isnan(result)


def test_outliers_ignores_non_numeric():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "name": ["x", "y", "z", "w", "v"]})
    assert set(outliers(df)["feature"]) == {"a"}


def test_outliers_custom_multiplier():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
    assert outliers(s, multiplier=1.0) >= outliers(s, multiplier=3.0)


# ===== pca_variance =====

def test_pca_variance_returns_dataframe(independent_df):
    result = pca_variance(independent_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["component", "variance_explained", "cumulative"]


def test_pca_variance_all_components(independent_df):
    result = pca_variance(independent_df)
    assert result["component"].tolist() == ["PC1", "PC2", "PC3"]


def test_pca_variance_sums_to_100(independent_df):
    result = pca_variance(independent_df, decimals=None)
    assert pytest.approx(result["variance_explained"].sum(), abs=0.01) == 100.0


def test_pca_variance_cumulative_ends_at_100(independent_df):
    result = pca_variance(independent_df, decimals=None)
    assert pytest.approx(result["cumulative"].iloc[-1], abs=0.01) == 100.0


def test_pca_variance_sorted_descending(independent_df):
    values = pca_variance(independent_df)["variance_explained"].tolist()
    assert values == sorted(values, reverse=True)


def test_pca_variance_correlated_features():
    np.random.seed(42)
    x = np.random.randn(100)
    df = pd.DataFrame({
        "a": x,
        "b": x + np.random.randn(100) * 0.01,
        "c": np.random.randn(100),
    })
    vals = _to_map(pca_variance(df), "component", "variance_explained")
    assert vals["PC1"] > 50


def test_pca_variance_n_components(independent_df):
    result = pca_variance(independent_df, n_components=2)
    assert result["component"].tolist() == ["PC1", "PC2"]


def test_pca_variance_too_few_columns_warns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.warns(PercentifyWarning):
        result = pca_variance(df)
    assert result.empty


def test_pca_variance_ignores_non_numeric():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["foo"] * 50,
    })
    assert len(pca_variance(df)) == 2


def _scaled_signal_df():
    """Two columns sharing a signal but on wildly different scales, plus noise."""
    np.random.seed(0)
    base = np.random.randn(300)
    return pd.DataFrame({
        "small": base + np.random.randn(300) * 0.3,
        "huge": (base + np.random.randn(300) * 0.3) * 50000,
        "indep": np.random.randn(300),
    })


def test_pca_variance_standardize_default_ignores_scale():
    # Default standardize=True: the huge-scale column must not hijack PC1.
    pc1 = pca_variance(_scaled_signal_df())["variance_explained"].iloc[0]
    assert pc1 < 90


def test_pca_variance_standardize_false_scale_dominates():
    # Covariance-based: the huge-scale column swamps everything.
    pc1 = pca_variance(_scaled_signal_df(), standardize=False)["variance_explained"].iloc[0]
    assert pc1 > 99


def test_pca_variance_standardize_drops_constant_column():
    np.random.seed(0)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "const": [5.0] * 100,
    })
    result = pca_variance(df)
    assert result["component"].tolist() == ["PC1", "PC2"]


def test_pca_variance_standardize_all_constant_warns():
    df = pd.DataFrame({"a": [5, 5, 5], "b": [1, 1, 1]})
    with pytest.warns(PercentifyWarning):
        result = pca_variance(df)
    assert result.empty


# ===== pca_loadings =====

def test_pca_loadings_returns_dataframe(independent_df):
    result = pca_loadings(independent_df)
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["feature", "PC1", "PC2", "PC3"]


def test_pca_loadings_feature_rows(independent_df):
    assert set(pca_loadings(independent_df)["feature"]) == {"a", "b", "c"}


def test_pca_loadings_columns_are_unit_norm(independent_df):
    result = pca_loadings(independent_df, decimals=None)
    for pc in ["PC1", "PC2", "PC3"]:
        assert abs((result[pc] ** 2).sum() - 1.0) < 1e-9


def test_pca_loadings_shared_signal_loads_together():
    np.random.seed(1)
    base = np.random.randn(200)
    df = pd.DataFrame({
        "a": base + np.random.randn(200) * 0.05,
        "b": base + np.random.randn(200) * 0.05,   # shares a signal with a
        "c": np.random.randn(200),                  # independent
    })
    load = dict(zip(pca_loadings(df)["feature"], pca_loadings(df)["PC1"]))
    assert abs(load["a"]) > 0.5
    assert abs(load["b"]) > 0.5
    assert (load["a"] > 0) == (load["b"] > 0)   # a and b load with the same sign
    assert abs(load["c"]) < 0.3


def test_pca_loadings_n_components(independent_df):
    result = pca_loadings(independent_df, n_components=2)
    assert result.columns.tolist() == ["feature", "PC1", "PC2"]


def test_pca_loadings_ignores_non_numeric():
    np.random.seed(1)
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "name": ["x"] * 50,
    })
    assert set(pca_loadings(df)["feature"]) == {"a", "b"}


def test_pca_loadings_too_few_columns_warns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.warns(PercentifyWarning):
        result = pca_loadings(df)
    assert result.empty


def test_pca_loadings_custom_decimals(independent_df):
    for val in pca_loadings(independent_df, decimals=3)["PC1"]:
        parts = str(float(val)).split(".")
        if len(parts) == 2:
            assert len(parts[1]) <= 3


# ===== imbalance =====

def test_imbalance_basic():
    s = pd.Series(["No"] * 85 + ["Yes"] * 15)
    result = imbalance(s)
    assert list(result.columns) == ["class", "count", "pct"]
    vals = dict(zip(result["class"], result["pct"]))
    assert vals["No"] == 85.0
    assert vals["Yes"] == 15.0


def test_imbalance_counts():
    result = imbalance(pd.Series(["No"] * 85 + ["Yes"] * 15))
    vals = dict(zip(result["class"], result["count"]))
    assert vals["No"] == 85
    assert vals["Yes"] == 15


def test_imbalance_sorted_descending():
    s = pd.Series(["a"] * 10 + ["b"] * 50 + ["c"] * 40)
    assert imbalance(s)["count"].tolist() == [50, 40, 10]


def test_imbalance_summary_attrs():
    s = pd.Series(["No"] * 850 + ["Yes"] * 150)
    summary = imbalance(s).attrs["summary"]
    assert summary["n_classes"] == 2
    assert summary["majority_class"] == "No"
    assert summary["minority_class"] == "Yes"
    assert summary["imbalance_ratio"] == 5.67
    assert abs(summary["entropy_pct"] - 61.0) < 0.5


def test_imbalance_balanced():
    s = pd.Series(["x"] * 50 + ["y"] * 50)
    summary = imbalance(s).attrs["summary"]
    assert summary["imbalance_ratio"] == 1.0
    assert abs(summary["entropy_pct"] - 100.0) < 0.01


def test_imbalance_ignores_nulls():
    s = pd.Series(["a", "b", None, "a", None])
    result = imbalance(s)
    vals = dict(zip(result["class"], result["count"]))
    assert vals["a"] == 2
    assert vals["b"] == 1
    assert result["count"].sum() == 3


def test_imbalance_empty_warns():
    with pytest.warns(PercentifyWarning):
        result = imbalance(pd.Series([], dtype=object))
    assert result.empty


def test_imbalance_dataframe_raises():
    with pytest.raises(TypeError):
        imbalance(pd.DataFrame({"a": [1, 2]}))


# ===== difference =====

def test_difference_two_scalars():
    assert difference(10, 20) == 66.67


def test_difference_same():
    assert difference(50, 50) == 0.0


def test_difference_both_zero():
    assert difference(0, 0) == 0.0


def test_difference_order_independent():
    assert difference(10, 20) == difference(20, 10)


def test_difference_two_columns():
    result = difference(pd.Series([10, 50, 100]), pd.Series([20, 50, 300]))
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 66.67
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 100.0


def test_difference_non_numeric_warns():
    with pytest.warns(PercentifyWarning):
        result = difference(pd.Series(["a", "b"]), pd.Series([1, 2]))
    assert result.isna().all()


# ===== split =====

def test_split_list():
    assert split(200, [1, 3]) == [50.0, 150.0]


def test_split_equal():
    result = split(100, [1, 1, 1])
    assert result == [33.33, 33.33, 33.34]
    assert sum(result) == 100


def test_split_series_returns_series():
    result = split(200, pd.Series([1, 3]))
    assert isinstance(result, pd.Series)
    assert result.tolist() == [50.0, 150.0]


def test_split_series_preserves_index():
    weights = pd.Series([1, 3], index=["a", "b"])
    result = split(200, weights)
    assert result["a"] == 50.0
    assert result["b"] == 150.0


def test_split_empty_raises():
    with pytest.raises(ValueError):
        split(100, [])


def test_split_zero_sum_raises():
    with pytest.raises(ValueError):
        split(100, [0, 0])


def test_split_non_numeric_warns():
    with pytest.warns(PercentifyWarning):
        result = split(100, pd.Series(["a", "b"]))
    assert result.isna().all()


# ===== display =====

def test_display_scalar():
    assert display(25.0) == "25.0%"


def test_display_multiply():
    assert display(0.45, multiply=True) == "45.0%"


def test_display_custom_suffix():
    assert display(50, suffix=" percent") == "50.0 percent"


def test_display_series():
    result = display(pd.Series([0.25, 0.5]), multiply=True)
    assert isinstance(result, pd.Series)
    assert result.tolist() == ["25.0%", "50.0%"]


def test_display_series_no_multiply():
    result = display(pd.Series([25.0, 33.3]))
    assert result.tolist() == ["25.0%", "33.3%"]


def test_display_non_numeric_warns():
    with pytest.warns(PercentifyWarning):
        result = display(pd.Series(["a", "b"]))
    assert result.isna().all()
