import pytest
import numpy as np
import pandas as pd
from percentify import (
    correlate, skew_report, bootstrap_ci, permutation_test, effect_size,
    PercentifyWarning,
)


# ===== correlate =====

def test_correlate_two_series_returns_tuple():
    r, p = correlate(pd.Series(range(50)), pd.Series(range(50)))
    assert r == 1.0
    assert p < 0.05


def test_correlate_dataframe_tidy_and_sorted():
    np.random.seed(0)
    base = np.random.randn(100)
    df = pd.DataFrame({
        "a": base,
        "b": base + np.random.randn(100) * 0.01,   # near-perfect with a
        "c": np.random.randn(100),
    })
    result = correlate(df)
    assert list(result.columns) == ["feature_1", "feature_2", "r", "p"]
    assert {result.iloc[0]["feature_1"], result.iloc[0]["feature_2"]} == {"a", "b"}
    assert result["r"].abs().tolist() == sorted(result["r"].abs().tolist(), reverse=True)


def test_correlate_small_p_is_not_rounded_to_zero():
    # A p-value below the decimals resolution must keep significant figures
    # instead of collapsing to a misleading 0.0.
    np.random.seed(0)
    n = 200
    a = np.random.randn(n)
    b = 0.44 * a + np.sqrt(1 - 0.44 ** 2) * np.random.randn(n)
    r, p = correlate(pd.Series(a), pd.Series(b))
    assert 0 < p < 0.01          # genuinely tiny, and never exactly zero
    assert r == round(r, 2)      # r still honours decimals


def test_correlate_underflowing_p_is_never_exactly_zero():
    # With a large sample the true p is ~1e-2000, so scipy underflows to a
    # literal 0.0. We must report a vanishingly small number, not "impossible".
    np.random.seed(0)
    n = 40000
    a = np.random.randn(n)
    b = 0.44 * a + np.sqrt(1 - 0.44 ** 2) * np.random.randn(n)
    _, p = correlate(pd.Series(a), pd.Series(b))
    assert p > 0.0
    assert p < 1e-300


def test_correlate_dataframe_p_column_never_zero():
    np.random.seed(0)
    base = np.random.randn(5000)
    df = pd.DataFrame({"a": base, "b": base * 2 + np.random.randn(5000) * 0.01})
    result = correlate(df)
    assert (result["p"] > 0).all()


def test_correlate_ordinary_p_still_rounds_normally():
    # Values above the resolution keep the familiar 2-decimal formatting.
    np.random.seed(3)
    r, p = correlate(pd.Series(np.random.randn(50)), pd.Series(np.random.randn(50)))
    assert p == round(p, 2)


def test_correlate_spearman():
    r, p = correlate(pd.Series([1, 2, 3, 4, 5]), pd.Series([1, 4, 9, 16, 25]), method="spearman")
    assert r == 1.0   # perfectly monotonic


def test_correlate_too_few_pairs_warns():
    with pytest.warns(PercentifyWarning):
        r, p = correlate(pd.Series([1.0, 2.0]), pd.Series([2.0, 4.0]))
    assert np.isnan(r)


def test_correlate_single_numeric_column_warns():
    with pytest.warns(PercentifyWarning):
        result = correlate(pd.DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]}))
    assert result.empty


def test_correlate_small_p_value_not_rounded_to_zero():
    np.random.seed(0)
    x = pd.Series(np.random.randn(200))
    y = pd.Series(x * 0.6 + np.random.randn(200) * 0.5)

    r, p = correlate(x, y)
    assert p > 0

    result = correlate(pd.DataFrame({"a": x, "b": y}))
    assert (result["p"] > 0).all()


# ===== skew_report =====

def test_skew_report_columns():
    np.random.seed(0)
    df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
    assert list(skew_report(df).columns) == [
        "feature", "skew", "kurtosis", "outlier_pct", "suggested_transform"]


def test_skew_report_right_skew_suggests_log1p():
    np.random.seed(0)
    row = skew_report(pd.DataFrame({"income": np.random.exponential(1, 500)})).iloc[0]
    assert row["skew"] > 1
    assert row["suggested_transform"] == "log1p"


def test_skew_report_symmetric_suggests_none():
    np.random.seed(0)
    row = skew_report(pd.DataFrame({"sym": np.random.randn(1000)})).iloc[0]
    assert row["suggested_transform"] == "none"


def test_skew_report_sorted_by_abs_skew():
    np.random.seed(0)
    df = pd.DataFrame({"sym": np.random.randn(500), "skewed": np.random.exponential(1, 500)})
    assert skew_report(df)["feature"].iloc[0] == "skewed"


def test_skew_report_no_numeric_warns():
    with pytest.warns(PercentifyWarning):
        result = skew_report(pd.DataFrame({"a": ["x", "y", "z"]}))
    assert result.empty


# ===== bootstrap_ci =====

def test_bootstrap_ci_returns_ordered_tuple():
    lo, hi = bootstrap_ci(list(range(100)), random_state=0)
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo < hi


def test_bootstrap_ci_contains_true_mean():
    np.random.seed(0)
    lo, hi = bootstrap_ci(np.random.randn(500) + 5.0, random_state=0)
    assert lo < 5.0 < hi


def test_bootstrap_ci_reproducible():
    data = list(range(50))
    assert bootstrap_ci(data, random_state=1) == bootstrap_ci(data, random_state=1)


def test_bootstrap_ci_custom_statistic():
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], statistic=np.median, random_state=0)
    assert lo <= hi


def test_bootstrap_ci_too_few_warns():
    with pytest.warns(PercentifyWarning):
        result = bootstrap_ci([1.0])
    assert np.isnan(result[0])


# ===== permutation_test =====

def test_permutation_test_returns_p_value():
    p = permutation_test([1, 2, 3, 4], [1, 2, 3, 4], random_state=0)
    assert isinstance(p, float)
    assert 0 < p <= 1


def test_permutation_test_flags_real_difference():
    np.random.seed(0)
    a = np.random.randn(100)
    b = np.random.randn(100) + 2.0
    assert permutation_test(a, b, random_state=0) < 0.05


def test_permutation_test_small_p_survives_rounding():
    # The floor is 1/(n_permutations+1); with 100k shuffles that is ~1e-5,
    # which plain 4-decimal rounding would have flattened to 0.0.
    np.random.seed(0)
    a = np.random.randn(200)
    b = np.random.randn(200) + 3
    p = permutation_test(a, b, n_permutations=100000, random_state=0)
    assert p > 0.0
    assert p < 0.001


def test_permutation_test_reproducible():
    a, b = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
    assert permutation_test(a, b, random_state=1) == permutation_test(a, b, random_state=1)


def test_permutation_test_too_few_warns():
    with pytest.warns(PercentifyWarning):
        result = permutation_test([1.0], [2.0, 3.0])
    assert np.isnan(result)


# ===== effect_size =====

def test_effect_size_numeric_columns():
    np.random.seed(0)
    df = pd.DataFrame({
        "g": ["A"] * 50 + ["B"] * 50,
        "v": np.concatenate([np.random.randn(50), np.random.randn(50) + 1]),
    })
    assert list(effect_size(df, group="g", value="v").columns) == [
        "comparison", "cohen_d", "hedges_g", "mean_diff", "interpretation"]


def test_effect_size_large_difference():
    np.random.seed(0)
    df = pd.DataFrame({
        "g": ["A"] * 100 + ["B"] * 100,
        "v": np.concatenate([np.random.randn(100), np.random.randn(100) + 3]),
    })
    assert effect_size(df, group="g", value="v")["interpretation"].iloc[0] == "large"


def test_effect_size_binary_outcome():
    df = pd.DataFrame({
        "variant": ["A"] * 100 + ["B"] * 100,
        "conv": [1] * 20 + [0] * 80 + [1] * 40 + [0] * 60,
    })
    result = effect_size(df, group="variant", value="conv")
    assert list(result.columns) == ["comparison", "cohen_h", "lift_pct", "interpretation"]
    assert result["cohen_h"].iloc[0] > 0


def test_effect_size_missing_column_warns():
    with pytest.warns(PercentifyWarning):
        result = effect_size(pd.DataFrame({"g": ["A", "B"], "v": [1, 2]}), group="g", value="nope")
    assert result.empty


def test_effect_size_not_two_groups_warns():
    df = pd.DataFrame({"g": ["A", "B", "C"] * 10, "v": range(30)})
    with pytest.warns(PercentifyWarning):
        result = effect_size(df, group="g", value="v")
    assert result.empty
