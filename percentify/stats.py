import functools
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd


class PercentifyWarning(UserWarning):
    """Raised when percentify handles a situation gracefully instead of erroring."""


def _warn(message: str) -> None:
    warnings.warn(message, PercentifyWarning, stacklevel=3)


def _round(value: float, decimals: Optional[int]) -> float:
    if decimals is None:
        return value
    return round(value, decimals)


def _round_p(value: float, decimals: Optional[int]) -> float:
    """Round a p-value without collapsing a small but non-zero result to 0.

    Plain rounding turns p=0.0001 into 0.0 at the default 2 decimals, which
    reads as "exactly zero" instead of "very small". When that happens, keep
    enough decimals to show the first two significant digits instead.

    A p-value of exactly 0.0 is passed through untouched. That is not a claim
    that p is zero: it means the true value underflowed double precision
    (scipy returns a literal 0.0 once p drops below about 1e-308). Use
    ``log_p=True`` on ``correlate`` to recover the magnitude in log space.
    """
    if decimals is None or not np.isfinite(value):
        return _round(value, decimals)
    rounded = round(value, decimals)
    if rounded != 0 or value == 0:
        return rounded
    magnitude = int(np.floor(np.log10(abs(value))))
    return round(value, -magnitude + 1)


# Safety cap for the incomplete-beta series below; convergence normally needs
# far fewer terms (a few hundred), so this only guards a pathological input.
_LOG_P_MAX_TERMS = 100_000


def _log10_p_from_r(r: float, n: int, p: Optional[float] = None) -> float:
    """log10 of the two-sided p-value for a correlation of ``r`` over ``n`` pairs.

    Returns log10 of scipy's own p-value whenever that is representable. Below
    the double-precision floor (about 1e-308) scipy returns a literal 0.0, so
    the magnitude is computed directly in log space instead of being lost.

    The two-sided p-value is the regularized incomplete beta
    ``I_x(df/2, 1/2)`` with ``x = 1 - r**2``. That is evaluated through its
    hypergeometric series rather than a leading-order tail approximation, so
    the result stays accurate for any x: agreement with an arbitrary-precision
    reference is within about 1e-9 log10 units.

    For ``method="spearman"`` the same transform is applied to rho. That
    matches the asymptotic scipy itself uses, and like scipy it is an
    approximation for small n or heavily tied data.
    """
    if p is not None and np.isfinite(p) and p > 0:
        return float(np.log10(p))

    from scipy import special

    r = abs(float(r))
    df = n - 2
    if not np.isfinite(r) or df < 1:
        return float("nan")
    if r >= 1.0:
        return float("-inf")  # a perfect correlation drives p below any bound
    if r == 0.0:
        return 0.0  # p == 1, and the series below is not needed

    a, b = df / 2.0, 0.5
    x = (1.0 - r) * (1.0 + r)  # == 1 - r**2, without the cancellation near |r|=1

    # I_x(a, b) = x**a / (a * B(a, b)) * 2F1(a, 1 - b; a + 1; x)
    term = total = 1.0
    for k in range(_LOG_P_MAX_TERMS):
        term *= (a + k) * (1.0 - b + k) * x / ((a + 1.0 + k) * (1.0 + k))
        total += term
        if term < 1e-17 * total:
            break

    log_ibeta = (a * np.log(x) - np.log(a) - special.betaln(a, b)
                 + np.log(total))
    return float(log_ibeta / np.log(10.0))


def _is_polars(obj) -> bool:
    """True if obj is a polars DataFrame/Series, without importing polars."""
    return type(obj).__module__.split(".", 1)[0] == "polars"


def _to_pandas(obj):
    """Convert a polars DataFrame/Series to pandas; leave anything else as-is."""
    return obj.to_pandas() if _is_polars(obj) else obj


def _to_polars(result):
    """Convert a pandas result back to polars, carrying any .attrs summary."""
    import polars as pl

    if isinstance(result, pd.DataFrame):
        out = pl.from_pandas(result)
        if getattr(result, "attrs", None):
            out.attrs = dict(result.attrs)
        return out
    if isinstance(result, pd.Series):
        return pl.from_pandas(result)
    return result


def _backend_aware(func):
    """Let a pandas-based function accept and return polars objects.

    When any argument is a polars DataFrame/Series, inputs are converted to
    pandas, the function runs, and the result is converted back to polars
    (polars in -> polars out). Pure-pandas calls pass straight through and
    never import polars, so polars stays an optional dependency.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        polars_in = (any(_is_polars(a) for a in args)
                     or any(_is_polars(v) for v in kwargs.values()))
        if not polars_in:
            return func(*args, **kwargs)
        args = tuple(_to_pandas(a) for a in args)
        kwargs = {k: _to_pandas(v) for k, v in kwargs.items()}
        return _to_polars(func(*args, **kwargs))
    return wrapper


@_backend_aware
def change(old, new=None, decimals: Optional[int] = 2):
    """
    Percentage change.

    - Two numbers: percentage change from ``old`` to ``new``.
    - Two columns/Series: element-wise percentage change from ``old`` to ``new``.
    - One Series: period-over-period percentage change down the column.
    - One DataFrame: period-over-period change for every numeric column.

    Args:
        old: The original value(s) - a number, Series, or DataFrame.
        new: The new value(s) - a number or Series. Omit for period-over-period.
        decimals: Number of decimal places to round to. If None, no rounding.

    Returns:
        float, Series, or DataFrame, matching the input.
    """
    # Two columns / Series: element-wise change from `old` to `new`.
    if isinstance(old, pd.Series) and new is not None:
        new_s = new if isinstance(new, pd.Series) else pd.Series(new, index=old.index)
        if not (pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new_s)):
            _warn("change expects numeric columns, but got non-numeric data. "
                  "Returning NaN. Encode or select numeric columns.")
            return pd.Series([float("nan")] * len(old), index=old.index)
        old_f = old.astype(float)
        new_f = new_s.astype(float)
        result = (new_f - old_f) / old_f.abs() * 100.0
        result = result.where(old_f != 0, 0.0)  # old == 0 -> 0.0 (safe division)
        return result if decimals is None else result.round(decimals)

    # One Series: period-over-period change down the column.
    if isinstance(old, pd.Series):
        if not pd.api.types.is_numeric_dtype(old):
            _warn(f"change expects numeric data, but got a non-numeric Series "
                  f"(dtype: {old.dtype}). Returning NaN. Encode or select a numeric column.")
            return pd.Series([float("nan")] * len(old), index=old.index)
        result = old.pct_change(fill_method=None) * 100.0
        return result if decimals is None else result.round(decimals)

    # One DataFrame: period-over-period change for every numeric column.
    if isinstance(old, pd.DataFrame):
        numeric = old.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            _warn("Numeric columns required: no numeric columns found for change.")
            return numeric
        result = numeric.pct_change(fill_method=None) * 100.0
        return result if decimals is None else result.round(decimals)

    if new is None:
        raise ValueError(
            "change(old, new) needs two numbers. Pass a Series or DataFrame instead "
            "for period-over-period percentage change."
        )

    old = float(old)
    if old == 0:
        return 0.0
    value = (float(new) - old) / abs(old) * 100.0
    return _round(value, decimals)


@_backend_aware
def vif(df: pd.DataFrame, decimals: Optional[int] = 2, flag: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor for each numeric column in a DataFrame.

    VIF = 1 / (1 - R²), where R² comes from regressing each feature
    against all other features. VIF > 5 suggests moderate multicollinearity,
    VIF > 10 suggests severe multicollinearity.

    Non-numeric columns are ignored. If fewer than 2 numeric columns are
    available, a warning is raised and an empty DataFrame is returned instead
    of an error — encode categoricals first.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to. If None, raw floats.
        flag: If set, only return rows with VIF above this threshold.

    Returns:
        DataFrame with columns ["feature", "VIF"], sorted highest VIF first.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"vif expects a pandas DataFrame, got {type(df).__name__}.")

    empty = pd.DataFrame(columns=["feature", "VIF"])
    numeric = df.select_dtypes(include=[np.number])
    n_numeric = numeric.shape[1]

    if n_numeric < 2:
        if n_numeric == 0:
            _warn("Numeric columns required: no numeric columns found. VIF measures "
                  "multicollinearity between numeric features - encode any "
                  "categorical/text columns first.")
        else:
            _warn("Numeric columns required: VIF needs at least 2 numeric columns to "
                  f"measure multicollinearity, but found only {n_numeric}.")
        return empty

    numeric = numeric.dropna()
    if numeric.shape[0] < 2:
        _warn("Not enough data: VIF needs at least 2 complete (non-null) rows.")
        return empty

    cols = numeric.columns.tolist()
    X = numeric.values.astype(float)

    vifs = []
    for i in range(len(cols)):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        others_with_intercept = np.hstack([np.ones((others.shape[0], 1)), others])

        coeffs, _, _, _ = np.linalg.lstsq(others_with_intercept, y, rcond=None)

        y_pred = others_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_sq = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
        vif_val = float("inf") if r_sq >= 1.0 else 1.0 / (1.0 - r_sq)

        if decimals is not None:
            vif_val = round(vif_val, decimals)
        vifs.append(vif_val)

    result = pd.DataFrame({"feature": cols, "VIF": vifs})
    result = result.sort_values("VIF", ascending=False).reset_index(drop=True)

    if flag is not None:
        result = result[result["VIF"] > flag].reset_index(drop=True)

    return result


@_backend_aware
def missing(df: pd.DataFrame, decimals: Optional[int] = 2) -> pd.DataFrame:
    """
    Calculate the percentage of missing values for each column.

    Works on every column (numeric and text), so nothing is dropped.

    Args:
        df: DataFrame to profile.
        decimals: Number of decimal places to round to.

    Returns:
        DataFrame with columns ["column", "missing_pct", "has_missing"],
        sorted highest first. ``has_missing`` is calculated before percentage
        rounding, so even a tiny non-zero amount of missing data remains visible.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"missing expects a pandas DataFrame, got {type(df).__name__}.")

    total = len(df)
    rows = []
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        pct = 0.0 if total == 0 else missing_count / total * 100.0
        rows.append((col, _round(pct, decimals), missing_count > 0, missing_count))

    result = pd.DataFrame(
        rows,
        columns=["column", "missing_pct", "has_missing", "_missing_count"],
    )
    result = result.sort_values(
        "_missing_count", ascending=False, kind="stable",
    ).drop(columns="_missing_count")
    return result.reset_index(drop=True)


@_backend_aware
def cv(data: Union[pd.Series, pd.DataFrame], decimals: Optional[int] = 2) -> Union[float, pd.DataFrame]:
    """
    Calculate the coefficient of variation (CV = std / mean * 100).

    Args:
        data: A Series (returns a single float) or DataFrame (returns a DataFrame
            of all numeric columns).
        decimals: Number of decimal places to round to.

    Returns:
        float for a Series, or a DataFrame with columns ["feature", "cv"]
        sorted highest first. Non-numeric input is handled with a warning.
    """
    if isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            _warn(f"cv expects numeric data, but got a non-numeric Series (dtype: "
                  f"{data.dtype}). Returning NaN. Encode or select a numeric column.")
            return float("nan")
        mean = data.mean()
        if mean == 0:
            _warn("Coefficient of variation is undefined when the mean is zero. "
                  "Returning inf.")
            return float("inf")
        return _round(data.std() / abs(mean) * 100.0, decimals)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"cv expects a Series or DataFrame, got {type(data).__name__}.")

    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        _warn("Numeric columns required: no numeric columns found for cv.")
        return pd.DataFrame(columns=["feature", "cv"])

    rows = []
    for col in numeric.columns:
        mean = numeric[col].mean()
        if mean == 0:
            rows.append((col, float("inf")))
        else:
            rows.append((col, _round(numeric[col].std() / abs(mean) * 100.0, decimals)))

    result = pd.DataFrame(rows, columns=["feature", "cv"])
    return result.sort_values("cv", ascending=False).reset_index(drop=True)


@_backend_aware
def outliers(
    data: Union[pd.Series, pd.DataFrame], decimals: Optional[int] = 2, multiplier: float = 1.5
) -> Union[float, pd.DataFrame]:
    """
    Calculate the percentage of outliers using the IQR method.

    An outlier is any value below Q1 - multiplier*IQR or above Q3 + multiplier*IQR.

    Args:
        data: A Series (returns a single float) or DataFrame (returns a DataFrame
            of all numeric columns).
        decimals: Number of decimal places to round to.
        multiplier: IQR multiplier for the outlier bounds (default: 1.5).

    Returns:
        float for a Series, or a DataFrame with columns ["feature", "outlier_pct"]
        sorted highest first. Non-numeric input is handled with a warning.
    """
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return 0.0
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        count = ((s < lower) | (s > upper)).sum()
        return count / len(s) * 100.0

    if isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            _warn(f"outliers expects numeric data, but got a non-numeric Series "
                  f"(dtype: {data.dtype}). Returning NaN. Encode or select a numeric column.")
            return float("nan")
        return _round(_calc(data), decimals)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"outliers expects a Series or DataFrame, got {type(data).__name__}.")

    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        _warn("Numeric columns required: no numeric columns found for outliers.")
        return pd.DataFrame(columns=["feature", "outlier_pct"])

    rows = [(col, _round(_calc(numeric[col]), decimals)) for col in numeric.columns]
    result = pd.DataFrame(rows, columns=["feature", "outlier_pct"])
    return result.sort_values("outlier_pct", ascending=False).reset_index(drop=True)


@_backend_aware
def pca_variance(
    df: pd.DataFrame, decimals: Optional[int] = 2, n_components: Optional[int] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Calculate the percentage of variance explained by each principal component.

    Performs PCA via eigendecomposition. By default every column is standardized
    to unit variance first (correlation-based PCA), so that a column measured in
    large units (e.g. dollars) cannot dominate the result purely because of its
    scale. Set ``standardize=False`` for covariance-based PCA on the raw values.

    Non-numeric columns are ignored. If fewer than 2 usable numeric columns are
    available, a warning is raised and an empty DataFrame is returned.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to.
        n_components: Number of components to return. If None, returns all.
        standardize: If True (default), scale each column to unit variance
            before the decomposition. Constant (zero-variance) columns are
            dropped when standardizing, since they cannot be scaled.

    Returns:
        DataFrame with columns ["component", "variance_explained", "cumulative"].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"pca_variance expects a pandas DataFrame, got {type(df).__name__}.")

    empty = pd.DataFrame(columns=["component", "variance_explained", "cumulative"])
    numeric = df.select_dtypes(include=[np.number]).dropna()

    if numeric.shape[1] < 2:
        _warn("Numeric columns required: pca_variance needs at least "
              "2 numeric columns.")
        return empty

    if numeric.shape[0] < 2:
        _warn("Not enough data: need at least 2 complete (non-null) rows.")
        return empty

    X = numeric.values.astype(float)

    if standardize:
        keep = X.std(axis=0, ddof=1) > 0
        if keep.sum() < 2:
            _warn("Numeric columns required: after dropping constant (zero-variance) "
                  "columns, fewer than 2 columns remain to standardize for PCA.")
            return empty
        X = X[:, keep]
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    eigenvalues = eigenvalues[::-1]
    total = eigenvalues.sum()

    if total == 0:
        _warn("All numeric columns are constant - there is no variance to explain.")
        return empty

    ratios = eigenvalues / total * 100.0
    if n_components is not None:
        ratios = ratios[:n_components]

    cumulative = np.cumsum(ratios)
    return pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(ratios))],
        "variance_explained": [_round(r, decimals) for r in ratios],
        "cumulative": [_round(c, decimals) for c in cumulative],
    })


@_backend_aware
def pca_loadings(
    df: pd.DataFrame, decimals: Optional[int] = 2, n_components: Optional[int] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Principal component loadings: how much each feature contributes to each PC.

    Returns the eigenvectors of the (optionally standardized) covariance matrix
    as a feature x component table. Read down a column to see what a component
    is made of. Columns are standardized first by default (matching
    pca_variance); pass standardize=False for covariance-based loadings.

    Non-numeric columns are ignored. If fewer than 2 usable numeric columns are
    available, a warning is raised and an empty DataFrame is returned.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to.
        n_components: Number of components (columns) to return. If None, all.
        standardize: If True (default), scale each column to unit variance first.

    Returns:
        DataFrame with a "feature" column followed by PC1, PC2, ... loadings.
        Note: the sign of a component is arbitrary, so only the relative signs
        and magnitudes within a column are meaningful.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"pca_loadings expects a pandas DataFrame, got {type(df).__name__}.")

    empty = pd.DataFrame(columns=["feature"])
    numeric = df.select_dtypes(include=[np.number]).dropna()

    if numeric.shape[1] < 2:
        _warn("Numeric columns required: pca_loadings needs at least 2 numeric columns.")
        return empty

    if numeric.shape[0] < 2:
        _warn("Not enough data: need at least 2 complete (non-null) rows.")
        return empty

    cols = numeric.columns.tolist()
    X = numeric.values.astype(float)

    if standardize:
        keep = X.std(axis=0, ddof=1) > 0
        if keep.sum() < 2:
            _warn("Numeric columns required: after dropping constant (zero-variance) "
                  "columns, fewer than 2 columns remain to standardize for PCA.")
            return empty
        cols = [c for c, k in zip(cols, keep) if k]
        X = X[:, keep]
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(X, rowvar=False))

    if eigenvalues.sum() == 0:
        _warn("All numeric columns are constant - there are no components to describe.")
        return empty

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    result = pd.DataFrame({"feature": cols})
    for i in range(eigenvectors.shape[1]):
        result[f"PC{i + 1}"] = [_round(float(v), decimals) for v in eigenvectors[:, i]]
    return result


@_backend_aware
def imbalance(data, decimals: Optional[int] = 2) -> pd.DataFrame:
    """
    Summarize class imbalance in a categorical target column.

    Args:
        data: A Series - the target/label column.
        decimals: Number of decimal places to round percentages to.

    Returns:
        DataFrame with columns ["class", "count", "pct"], most frequent first.
        Headline metrics are attached to ``result.attrs["summary"]``:
        n_classes, majority_class, minority_class, imbalance_ratio, entropy_pct.
        (imbalance_ratio = majority count / minority count; entropy_pct = 100 for
        a perfectly balanced target, approaching 0 as one class dominates.)
    """
    if isinstance(data, pd.DataFrame):
        raise TypeError(
            "imbalance expects a single column (Series), not a DataFrame. "
            "Pass one column, e.g. df['target']."
        )

    s = pd.Series(data).dropna()
    if len(s) == 0:
        _warn("No data: the target column is empty or all-null.")
        return pd.DataFrame(columns=["class", "count", "pct"])

    counts = s.value_counts()
    n = int(counts.sum())
    probs = counts.to_numpy(dtype=float) / n

    result = pd.DataFrame({
        "class": list(counts.index),
        "count": counts.to_numpy(dtype=int),
        "pct": [_round(p * 100.0, decimals) for p in probs],
    })

    entropy = -(probs * np.log(probs)).sum()
    max_entropy = np.log(len(counts))
    entropy_pct = (entropy / max_entropy * 100.0) if max_entropy > 0 else 0.0
    minority_count = counts.iloc[-1]
    ratio = counts.iloc[0] / minority_count if minority_count != 0 else float("inf")

    result.attrs["summary"] = {
        "n_classes": int(len(counts)),
        "majority_class": counts.index[0],
        "minority_class": counts.index[-1],
        "imbalance_ratio": _round(float(ratio), decimals),
        "entropy_pct": _round(float(entropy_pct), decimals),
    }
    return result


def _mean_diff(x, y):
    return float(np.mean(x) - np.mean(y))


def _suggest_transform(skew, series):
    if not np.isfinite(skew) or abs(skew) <= 0.5:
        return "none"
    if skew > 0.5 and series.min() >= 0:
        return "log1p"
    return "yeo-johnson"


def _interpret_d(d):
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _interpret_h(h):
    if h < 0.2:
        return "small"
    if h < 0.5:
        return "medium"
    return "large"


@_backend_aware
def correlate(a, b=None, method: str = "pearson", decimals: Optional[int] = 2,
              log_p: bool = False):
    """
    Correlation with p-values, the piece pandas' df.corr() leaves out.

    - Two Series: returns a (correlation, p_value) tuple.
    - A DataFrame: returns a tidy table of every numeric pair, strongest first,
      with columns ["feature_1", "feature_2", "r", "p"].

    Args:
        a: A Series (with b) or a DataFrame (matrix mode).
        b: The second Series for a pairwise correlation.
        method: "pearson" (linear) or "spearman" (rank / monotonic).
        decimals: Number of decimal places to round to. A p-value smaller than
            this resolution keeps two significant figures instead of rounding
            to 0.0, so a tiny p reads as 1.7e-31 rather than a false zero.
        log_p: Also report log10 of the p-value. Adds a "log10_p" column in
            matrix mode, and returns (r, p, log10_p) for two Series.

    Returns:
        A (r, p) tuple for two Series, or a DataFrame for a DataFrame.
        With log_p=True, a (r, p, log10_p) tuple or an extra "log10_p" column.

    Note:
        On a large sample almost any r is "significant", so p mostly tells you
        the correlation is not exactly zero. Read r for the strength.

        On a large sample p can underflow to a literal 0.0, which means "below
        1e-308", not "zero". log_p=True recovers the magnitude: a p reported as
        0.0 may be log10_p = -1967.54, i.e. about 1e-1968.
    """
    from scipy import stats

    corr_fns = {"pearson": stats.pearsonr, "spearman": stats.spearmanr}
    if method not in corr_fns:
        raise ValueError("method must be 'pearson' or 'spearman'.")
    corr_fn = corr_fns[method]

    if isinstance(a, pd.Series):
        if b is None:
            raise ValueError("correlate needs a second Series, or pass a DataFrame.")
        pair = pd.DataFrame({"a": a, "b": b}).apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            _warn("correlate needs at least 3 complete numeric pairs. Returning NaN.")
            nan = float("nan")
            return (nan, nan, nan) if log_p else (nan, nan)
        r, p = corr_fn(pair["a"].to_numpy(), pair["b"].to_numpy())
        out = (_round(float(r), decimals), _round_p(float(p), decimals))
        if log_p:
            log10_p = _log10_p_from_r(float(r), len(pair), float(p))
            return out + (_round(log10_p, decimals),)
        return out

    if not isinstance(a, pd.DataFrame):
        raise TypeError(f"correlate expects a Series or DataFrame, got {type(a).__name__}.")

    columns = ["feature_1", "feature_2", "r", "p"] + (["log10_p"] if log_p else [])
    numeric = a.select_dtypes(include=[np.number])
    empty = pd.DataFrame(columns=columns)
    cols = numeric.columns.tolist()
    if len(cols) < 2:
        _warn("Numeric columns required: correlate needs at least 2 numeric columns.")
        return empty

    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            pair = numeric[[c1, c2]].dropna()
            x, y = pair[c1].to_numpy(), pair[c2].to_numpy()
            if len(pair) < 3 or np.std(x) == 0 or np.std(y) == 0:
                continue
            r, p = corr_fn(x, y)
            row = (c1, c2, _round(float(r), decimals), _round_p(float(p), decimals))
            if log_p:
                log10_p = _log10_p_from_r(float(r), len(pair), float(p))
                row += (_round(log10_p, decimals),)
            rows.append(row)

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.loc[result["r"].abs().sort_values(ascending=False).index].reset_index(drop=True)


@_backend_aware
def skew_report(df: pd.DataFrame, decimals: Optional[int] = 2) -> pd.DataFrame:
    """
    Distribution shape per numeric column: skew, kurtosis, outlier percentage,
    and a suggested transform. Most-skewed first.

    Args:
        df: DataFrame with numeric columns.
        decimals: Number of decimal places to round to.

    Returns:
        DataFrame with columns
        ["feature", "skew", "kurtosis", "outlier_pct", "suggested_transform"].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"skew_report expects a pandas DataFrame, got {type(df).__name__}.")

    numeric = df.select_dtypes(include=[np.number])
    empty = pd.DataFrame(
        columns=["feature", "skew", "kurtosis", "outlier_pct", "suggested_transform"])
    if numeric.shape[1] == 0:
        _warn("Numeric columns required: no numeric columns found for skew_report.")
        return empty

    out_df = outliers(numeric, decimals=decimals)
    out_map = dict(zip(out_df["feature"], out_df["outlier_pct"]))

    rows = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 3:
            sk = kt = float("nan")
        else:
            sk, kt = float(s.skew()), float(s.kurt())
        rows.append({
            "feature": col,
            "skew": _round(sk, decimals),
            "kurtosis": _round(kt, decimals),
            "outlier_pct": out_map.get(col, 0.0),
            "suggested_transform": _suggest_transform(sk, s),
        })

    result = pd.DataFrame(rows)
    order = result["skew"].abs().sort_values(ascending=False, na_position="last").index
    return result.loc[order].reset_index(drop=True)


@_backend_aware
def bootstrap_ci(data, statistic=None, ci: float = 95, n_resamples: int = 1000,
                 decimals: Optional[int] = 2, random_state=None):
    """
    Bootstrap confidence interval for a statistic, with no distribution assumed.

    Resamples the data with replacement and reads the percentiles of the
    resampled statistic.

    Args:
        data: A Series or array of values.
        statistic: Function applied to each resample (default: mean).
        ci: Confidence level as a percentage (default 95).
        n_resamples: Number of bootstrap resamples.
        decimals: Number of decimal places to round to.
        random_state: Seed for reproducibility.

    Returns:
        A (low, high) tuple.
    """
    if statistic is None:
        statistic = np.mean

    values = pd.to_numeric(pd.Series(data), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        _warn("bootstrap_ci needs at least 2 numeric values. Returning (NaN, NaN).")
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(random_state)
    n = len(values)
    resampled = np.array([statistic(values[rng.integers(0, n, n)]) for _ in range(n_resamples)])

    alpha = (100 - ci) / 2
    low, high = np.percentile(resampled, [alpha, 100 - alpha])
    return (_round(float(low), decimals), _round(float(high), decimals))


@_backend_aware
def permutation_test(a, b, statistic=None, n_permutations: int = 1000,
                     decimals: Optional[int] = 4, random_state=None) -> float:
    """
    Permutation test for a difference between two samples. Returns a two-sided
    p-value (a number, not a verdict).

    Args:
        a, b: The two samples (Series or arrays).
        statistic: Function of (group_a, group_b) measuring the effect
            (default: difference in means).
        n_permutations: Number of label shuffles.
        decimals: Number of decimal places to round to. A p-value smaller than
            this resolution keeps two significant figures instead of rounding
            to 0.0. The smallest p this test can report is
            1 / (n_permutations + 1); raise n_permutations to resolve further.
        random_state: Seed for reproducibility.

    Returns:
        float: the p-value.
    """
    if statistic is None:
        statistic = _mean_diff

    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        _warn("permutation_test needs at least 2 numeric values per group. Returning NaN.")
        return float("nan")

    observed = abs(statistic(a, b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(random_state)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        if abs(statistic(combined[:n_a], combined[n_a:])) >= observed:
            count += 1
    return _round_p(float((count + 1) / (n_permutations + 1)), decimals)


@_backend_aware
def effect_size(df: pd.DataFrame, group: str, value: str, decimals: Optional[int] = 2) -> pd.DataFrame:
    """
    Effect size between two groups: the practical size of a difference, not just
    whether it is statistically significant.

    Detects the outcome type automatically:
    - Numeric value: Cohen's d, Hedges' g, and the mean difference.
    - Binary value (two levels): Cohen's h and the percentage lift.

    Args:
        df: DataFrame containing the group and value columns.
        group: Column naming the two groups to compare.
        value: The outcome column.
        decimals: Number of decimal places to round to.

    Returns:
        A one-row DataFrame of effect-size metrics with an interpretation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"effect_size expects a pandas DataFrame, got {type(df).__name__}.")
    for col in (group, value):
        if col not in df.columns:
            _warn(f"Column {col!r} not found. Returning empty.")
            return pd.DataFrame()

    data = df[[group, value]].dropna()
    levels = data[group].unique().tolist()
    if len(levels) != 2:
        _warn(f"effect_size compares exactly 2 groups, but {group!r} has {len(levels)}. "
              "Returning empty.")
        return pd.DataFrame()

    g1, g2 = levels
    a = data[data[group] == g1][value]
    b = data[data[group] == g2][value]
    if len(a) < 2 or len(b) < 2:
        _warn("effect_size needs at least 2 observations per group. Returning empty.")
        return pd.DataFrame()

    # Binary outcome (two levels): Cohen's h + lift.
    if data[value].nunique() == 2:
        positive = sorted(data[value].unique().tolist())[-1]
        p1, p2 = float((a == positive).mean()), float((b == positive).mean())
        h = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
        lift = (p1 - p2) / p2 * 100.0 if p2 != 0 else float("inf")
        return pd.DataFrame([{
            "comparison": f"{g1} vs {g2}",
            "cohen_h": _round(float(h), decimals),
            "lift_pct": _round(float(lift), decimals),
            "interpretation": _interpret_h(h),
        }])

    if not pd.api.types.is_numeric_dtype(data[value]):
        _warn(f"value column {value!r} is neither numeric nor binary; "
              "effect_size needs one of those. Returning empty.")
        return pd.DataFrame()

    # Numeric outcome: Cohen's d, Hedges' g, mean difference.
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    mean_diff = float(np.mean(x) - np.mean(y))
    pooled = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    d = mean_diff / pooled if pooled != 0 else 0.0
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    return pd.DataFrame([{
        "comparison": f"{g1} vs {g2}",
        "cohen_d": _round(float(d), decimals),
        "hedges_g": _round(float(g), decimals),
        "mean_diff": _round(mean_diff, decimals),
        "interpretation": _interpret_d(abs(d)),
    }])


@_backend_aware
def difference(a, b, decimals: Optional[int] = 2):
    """
    Symmetric percentage difference between two values or two columns.

    Unlike ``change`` (directional, old -> new), ``difference`` is
    order-independent: it measures how far apart two values are, using their
    average as the denominator.

    - Two numbers: a float.
    - Two Series/columns: element-wise Series.

    Args:
        a, b: Two numbers, or two Series/columns to compare element-wise.
        decimals: Number of decimal places to round to. If None, no rounding.

    Returns:
        float for two numbers, a Series for two columns.
    """
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        idx = a.index if isinstance(a, pd.Series) else b.index
        a_s = a if isinstance(a, pd.Series) else pd.Series(a, index=idx)
        b_s = b if isinstance(b, pd.Series) else pd.Series(b, index=idx)
        if not (pd.api.types.is_numeric_dtype(a_s) and pd.api.types.is_numeric_dtype(b_s)):
            _warn("difference expects numeric columns, but got non-numeric data. "
                  "Returning NaN. Encode or select numeric columns.")
            return pd.Series([float("nan")] * len(a_s), index=a_s.index)
        a_f = a_s.astype(float)
        b_f = b_s.astype(float)
        avg = (a_f.abs() + b_f.abs()) / 2.0
        result = (a_f - b_f).abs() / avg * 100.0
        result = result.where(avg != 0, 0.0)  # both zero -> 0.0
        return result if decimals is None else result.round(decimals)

    a = float(a)
    b = float(b)
    avg = (abs(a) + abs(b)) / 2.0
    if avg == 0:
        return 0.0
    return _round(abs(a - b) / avg * 100.0, decimals)


@_backend_aware
def split(total, weights, decimals: Optional[int] = 2):
    """
    Distribute a total across weights, proportionally.

    - A list/array of weights: a list of allocations.
    - A Series of weights: a Series of allocations (aligned to its index).

    Args:
        total: The total amount to distribute.
        weights: List/array or Series of weights.
        decimals: Number of decimal places to round to. If None, no rounding.

    Returns:
        list for list weights, Series for Series weights.

    Raises:
        ValueError: If weights is empty or the weights sum to zero.
    """
    is_series = isinstance(weights, pd.Series)
    w = weights if is_series else pd.Series(list(weights))

    if len(w) == 0:
        raise ValueError("`weights` must not be empty.")

    if not pd.api.types.is_numeric_dtype(w):
        _warn("split expects numeric weights, but got non-numeric data. "
              "Returning NaN. Encode or select numeric weights.")
        nan = pd.Series([float("nan")] * len(w), index=w.index)
        return nan if is_series else nan.tolist()

    weight_sum = float(w.sum())
    if weight_sum == 0:
        raise ValueError("Sum of `weights` must not be zero.")

    shares = w.astype(float) / weight_sum * float(total)
    if decimals is not None:
        shares = shares.round(decimals)
        remainder = round(float(total), decimals) - float(shares.sum())
        if remainder:
            shares.iloc[-1] = round(shares.iloc[-1] + remainder, decimals)
    return shares if is_series else shares.tolist()


@_backend_aware
def display(value, decimals: Optional[int] = 2, suffix: str = "%", multiply: bool = False):
    """
    Format a number or a numeric column as percentage strings.

    - A single number: a string, e.g. "45.0%".
    - A Series/column: a Series of formatted strings.

    Args:
        value: A number or Series to format.
        decimals: Number of decimal places to round to. If None, no rounding.
        suffix: The string appended to each value (default "%").
        multiply: If True, multiply by 100 first (0.45 -> "45.0%").

    Returns:
        str for a number, a Series of strings for a Series.
    """
    if isinstance(value, pd.Series):
        if not pd.api.types.is_numeric_dtype(value):
            _warn(f"display expects numeric data, but got a non-numeric Series "
                  f"(dtype: {value.dtype}). Returning NaN. Encode or select a numeric column.")
            return pd.Series([float("nan")] * len(value), index=value.index)
        v = value.astype(float)
        if multiply:
            v = v * 100.0
        if decimals is not None:
            v = v.round(decimals)
        return v.map(lambda x: f"{x}{suffix}" if pd.notna(x) else x)

    v = float(value)
    if multiply:
        v *= 100.0
    return f"{_round(v, decimals)}{suffix}"
