"""Diagnostic profiling for Percentify.

Unlike descriptive profilers that dump statistics for every column, ``profiler``
is a *diagnostician*: it runs a registry of checks, surfaces the problems that
actually matter, ranks them worst-first, and tells you how to fix each one.

    from percentify import profiler

    report = profiler(df)                # pandas or polars in, report out
    report                               # pretty summary (repr / notebook HTML)
    report.errors                        # just the blocking issues
    assert not report.errors             # drop straight into CI

Design notes
------------
* Backend is detected from the input type, never a flag: pandas, polars, or
  anything convertible all work. Analysis currently runs on a pandas view; the
  returned report is backend-agnostic. Native-polars checks can replace the
  conversion later without changing the public API.
* Every check is a plain function ``fn(df, target) -> list[Finding]`` registered
  in ``CHECKS``. Adding a diagnostic is a one-function change — a natural
  good-first-issue.
* No heavy dependencies: numpy + pandas only, matching the rest of the library.
  Notebook rendering uses ``_repr_html_``; the terminal uses plain unicode.
"""

from __future__ import annotations

import html
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .stats import PercentifyWarning, imbalance, missing

__all__ = ["profiler", "ProfileReport", "Finding"]

# Severity ordering and the penalty each level deducts from the health score.
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_SEVERITY_PENALTY = {"error": 12, "warning": 4, "info": 1}
_SEVERITY_MARK = {"error": "\u2717", "warning": "\u26a0", "info": "\u2139"}  # ✗ ⚠ ℹ

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█

# "All-nines" style magic numbers that are almost always disguised missing data.
# Deliberately conservative so real values (0, -1, small negatives) don't trip it.
_SENTINELS = frozenset({-999, 999, -9999, 9999, -99999, 99999, -999999})


@dataclass(frozen=True)
class Finding:
    """A single diagnostic result."""

    column: str
    severity: str  # "error" | "warning" | "info"
    code: str  # short machine-readable id, e.g. "id_like"
    message: str  # human explanation of what's wrong
    suggestion: str = ""  # the recommended fix

    def __str__(self) -> str:
        mark = _SEVERITY_MARK.get(self.severity, "-")
        tail = f"  \u2192 {self.suggestion}" if self.suggestion else ""
        return f"{mark} {self.column:<22} {self.message}{tail}"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _to_pandas(data) -> pd.DataFrame:
    """Return a pandas DataFrame regardless of the input backend.

    Detects polars without importing it unless needed, so polars is never a
    hard dependency.
    """
    if isinstance(data, pd.DataFrame):
        return data
    module = type(data).__module__.split(".")[0]
    if module == "polars":
        return data.to_pandas()
    if isinstance(data, pd.Series):
        return data.to_frame()
    # Last resort: let pandas try (dict, records, numpy array, ...).
    return pd.DataFrame(data)


def _sparkline(series: pd.Series, bins: int = 8) -> str:
    """A tiny unicode histogram for a numeric column."""
    values = pd.to_numeric(series, errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return " " * bins
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        return _SPARK[0] * bins
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    if counts.max() == 0:
        return " " * bins
    idx = np.floor(counts / counts.max() * (len(_SPARK) - 1)).astype(int)
    return "".join(_SPARK[i] for i in idx)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _is_textual_dtype(dt) -> bool:
    """True for object/categorical/string columns (incl. pandas 3.0 ``str``)."""
    if (pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt)
            or pd.api.types.is_datetime64_any_dtype(dt)
            or pd.api.types.is_timedelta64_dtype(dt)):
        return False
    return (isinstance(dt, pd.CategoricalDtype) or dt == object
            or pd.api.types.is_string_dtype(dt))


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _is_textual_dtype(df[c].dtype)]


def _format_missing(pct, has_missing) -> tuple[str, str]:
    """Format missingness consistently for terminal and notebook reports."""
    return f"{float(pct):.2f}%", "Yes" if bool(has_missing) else "No"


def _encode_target(target: pd.Series) -> tuple[np.ndarray, bool]:
    """Return (numeric codes, is_categorical)."""
    if pd.api.types.is_numeric_dtype(target) and not pd.api.types.is_bool_dtype(target):
        return target.to_numpy(dtype=float), False
    codes, _ = pd.factorize(target)
    return codes.astype(float), True


def _summarize(df: pd.DataFrame):
    """Per-column (dtype, missing %, missing flag, cardinality) summary."""
    rows, sparks = [], {}
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "cardinality": int(df[col].nunique(dropna=True)),
        })
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            sparks[col] = _sparkline(df[col])
    summary = pd.DataFrame(rows, columns=["column", "dtype", "cardinality"])
    summary = summary.merge(missing(df), on="column", how="left", sort=False)
    cols = ["column", "dtype", "missing_pct", "has_missing", "cardinality"]
    return summary[cols], sparks


def _target_info(series: pd.Series):
    """Return (inferred_role, balance_or_shape) for a target column.

    Many distinct values means a continuous target, not classification. That
    holds for numeric columns and for "scattered" continuous data that arrived
    as object/string (numbers stored as text): we coerce to numeric so those
    report a skew instead of a meaningless class breakdown.
    """
    is_bool = pd.api.types.is_bool_dtype(series)
    is_num = pd.api.types.is_numeric_dtype(series) and not is_bool
    is_text = (pd.api.types.is_object_dtype(series)
               or pd.api.types.is_string_dtype(series))
    n_distinct = int(series.nunique(dropna=True))

    if (is_num or is_text) and n_distinct > 20:
        # Coerce object/string to numbers so scattered data stored as text is
        # scored as regression rather than crashing on .skew() or falling
        # through to a bogus class breakdown.
        numeric = series if is_num else pd.to_numeric(series, errors="coerce")
        clean = numeric.dropna()
        non_null = int(series.notna().sum())
        mostly_numeric = bool(non_null) and clean.size / non_null >= 0.9
        if is_num or mostly_numeric:
            skew = float(clean.skew()) if clean.size >= 3 else float("nan")
            return "regression", (f"skew {skew:+.2f}" if np.isfinite(skew) else "n/a")
        # High-cardinality text that is not numeric: not a clean class target.
        return "regression", f"{n_distinct} distinct values, further diagnostic required"

    result = imbalance(series)
    summary = result.attrs.get("summary", {})
    if summary.get("n_classes", 0) < 2:
        return "classification", "single class" if summary.get("n_classes") == 1 else "n/a"
    pct = dict(zip(result["class"].tolist(), result["pct"].tolist()))
    maj, minr = summary["majority_class"], summary["minority_class"]
    balance = (f"majority '{maj}' {pct.get(maj, 0):.0f}%, "
               f"minority '{minr}' {pct.get(minr, 0):.0f}%, "
               f"ratio {summary['imbalance_ratio']:.1f}x")
    return "classification", balance


def _summarize_target(targets: dict) -> pd.DataFrame:
    """Target-specific summary: role, missing, distinct, and balance/shape."""
    rows = []
    for name, series in targets.items():
        role, balance = _target_info(series)
        rows.append({
            "column": name,
            "dtype": str(series.dtype),
            "inferred_role": role,
            "cardinality": int(series.nunique(dropna=True)),
            "balance": balance,
        })
    summary = pd.DataFrame(
        rows,
        columns=["column", "dtype", "inferred_role", "cardinality", "balance"],
    )
    summary = summary.merge(
        missing(pd.DataFrame(targets)), on="column", how="left", sort=False,
    )
    cols = [
        "column", "dtype", "inferred_role", "missing_pct", "has_missing",
        "cardinality", "balance",
    ]
    return summary[cols]


# --------------------------------------------------------------------------- #
# Checks — each takes (df, targets_dict) and returns a list[Finding]
# --------------------------------------------------------------------------- #
def check_missing(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    if len(df) == 0:
        return out
    # Reuse the package's missing() so there is a single source of truth.
    for _, row in missing(df).iterrows():
        col, pct = row["column"], row["missing_pct"]
        if pct >= 100.0:
            out.append(Finding(col, "error", "all_missing",
                               "column is entirely missing",
                               "drop the column"))
        elif pct >= 40.0:
            out.append(Finding(col, "warning", "high_missing",
                               f"{pct:.0f}% missing",
                               "impute or drop"))
    return out


def check_constant(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    for col in df.columns:
        if df[col].isna().all():
            continue  # already reported by check_missing
        if df[col].nunique(dropna=True) <= 1:
            out.append(Finding(col, "warning", "constant",
                               "only one distinct value",
                               "drop, carries no information"))
    return out


def check_id_like(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    if n < 20:
        return out
    for col in df.columns:
        nun = df[col].nunique(dropna=True)
        if nun / n >= 0.95 and (pd.api.types.is_integer_dtype(df[col])
                                or _is_textual_dtype(df[col].dtype)):
            out.append(Finding(col, "warning", "id_like",
                               f"identifier-like ({nun}/{n} unique)",
                               "further diagnostic required"))
    return out


def check_duplicate_rows(df: pd.DataFrame, target) -> list[Finding]:
    n = len(df)
    if n == 0:
        return []
    dups = int(df.duplicated().sum())
    if dups > 0:
        return [Finding("<rows>", "warning", "duplicate_rows",
                        f"{dups} duplicate rows ({dups / n * 100:.0f}%)",
                        "deduplicate")]
    return []


_SYMBOLS = re.compile(r"[,$%\s\u20ac\u00a3]")  # thousands sep, $, %, whitespace, € £
_DATEISH = re.compile(r"[-/:]")


def check_dtype_mismatch(df: pd.DataFrame, target) -> list[Finding]:
    # Regex runs in plain Python (not the pandas string accessor) so behaviour
    # is identical whether the column is object, pandas ``str`` or arrow-backed.
    out: list[Finding] = []
    for col in _text_columns(df):
        vals = [str(v) for v in df[col].dropna().head(200).tolist()]
        if len(vals) < 5:
            continue
        # numbers stored as text?
        stripped = [_SYMBOLS.sub("", v) for v in vals]
        numeric = pd.to_numeric(pd.Series(stripped), errors="coerce")
        if numeric.notna().mean() >= 0.9:
            out.append(Finding(col, "warning", "numeric_as_text",
                               "numbers stored as strings",
                               "cast to a numeric dtype"))
            continue
        # dates stored as text? Require a date-ish separator so bare ints
        # like "12345" aren't misread as timestamps.
        if np.mean([bool(_DATEISH.search(v)) for v in vals]) >= 0.8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(pd.Series(vals), errors="coerce",
                                        format="mixed")
            if parsed.notna().mean() >= 0.9:
                out.append(Finding(col, "warning", "date_as_text",
                                   "dates stored as strings",
                                   "cast to a datetime dtype"))
    return out


def check_categorical_consistency(df: pd.DataFrame, target) -> list[Finding]:
    """Detect values that are the same category under different spellings."""
    out: list[Finding] = []
    for col in _text_columns(df):
        raw = df[col].dropna().astype(str)
        if raw.empty:
            continue
        norm = raw.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        groups = raw.groupby(norm).agg(lambda s: sorted(set(s)))
        offenders = groups[groups.map(len) > 1]
        if not offenders.empty:
            variants = offenders.iloc[0]
            example = " / ".join(f'"{v}"' for v in variants[:3])
            extra = f" (+{len(offenders) - 1} more)" if len(offenders) > 1 else ""
            out.append(Finding(col, "warning", "inconsistent_categories",
                               f"same category, different spellings: {example}{extra}",
                               "normalize casing/whitespace"))
    return out


def check_high_cardinality(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    if n < 20:
        return out
    for col in _text_columns(df):
        nun = df[col].nunique(dropna=True)
        if 50 <= nun < 0.95 * n:  # many levels, but not identifier-like
            out.append(Finding(col, "info", "high_cardinality",
                               f"{nun} categories",
                               "will expand heavily under one-hot encoding"))
    return out


def check_sentinel_missing(df: pd.DataFrame, target) -> list[Finding]:
    out: list[Finding] = []
    for col in _numeric_columns(df):
        s = df[col].dropna()
        if len(s) < 20:
            continue
        for sentinel in _SENTINELS:
            share = (s == sentinel).mean()
            if share >= 0.01:
                out.append(Finding(col, "warning", "sentinel_missing",
                                   f"value {sentinel:g} in {share * 100:.0f}% of rows "
                                   "looks like a hidden missing code",
                                   "replace with NaN before analysis"))
                break
    return out


def check_collinearity(df: pd.DataFrame, target) -> list[Finding]:
    """Flag pairs of near-duplicate numeric columns (|r| >= 0.95)."""
    cols = _numeric_columns(df)
    if len(cols) < 2:
        return []
    corr = df[cols].corr(numeric_only=True).abs()
    out: list[Finding] = []
    seen: set[frozenset] = set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= 0.95 and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                out.append(Finding(f"{a} \u27f7 {b}", "warning", "collinear",
                                   f"correlation {r:.2f}, redundant pair",
                                   "drop one, or check vif()"))
    return out


def check_leakage(df: pd.DataFrame, targets) -> list[Finding]:
    """Flag features that predict a target almost perfectly."""
    if not targets:
        return []
    out: list[Finding] = []
    n = len(df)
    numeric_cols = set(_numeric_columns(df))
    text_cols = set(_text_columns(df))
    named = len(targets) > 1  # name the target only when there is more than one

    for tname, tseries in targets.items():
        y, y_is_cat = _encode_target(tseries)
        label = f" '{tname}'" if named else ""
        for col in df.columns:
            s = df[col]
            # Numeric feature: near-perfect linear association with the target.
            if col in numeric_cols:
                mask = s.notna().to_numpy() & np.isfinite(y)
                if mask.sum() < 10:
                    continue
                x = s.to_numpy(dtype=float)[mask]
                yy = y[mask]
                if np.std(x) == 0 or np.std(yy) == 0:
                    continue
                r = abs(np.corrcoef(x, yy)[0, 1])
                if np.isfinite(r) and r > 0.98:
                    out.append(Finding(col, "error", "leakage",
                                       f"predicts the target{label} (|r|={r:.2f})",
                                       "likely leakage, confirm it is known at predict time"))
            # Categorical feature: does it almost perfectly determine the target?
            elif y_is_cat and col in text_cols:
                nun = s.nunique(dropna=True)
                if nun <= 1 or nun >= 0.5 * n:
                    continue  # constant or id-like handled elsewhere
                tmp = pd.DataFrame({"f": s.astype("object"), "y": y}).dropna()
                if len(tmp) < 10:
                    continue
                purity = (tmp.groupby("f")["y"]
                          .agg(lambda g: g.value_counts(normalize=True).max())
                          * tmp.groupby("f").size() / len(tmp)).sum()
                if purity > 0.995:
                    out.append(Finding(col, "error", "leakage",
                                       f"almost perfectly determines the target{label} "
                                       f"(purity {purity:.2f})",
                                       "likely leakage, confirm it is known at predict time"))
    return out


def check_target_imbalance(df: pd.DataFrame, targets) -> list[Finding]:
    if not targets:
        return []
    out: list[Finding] = []
    named = len(targets) > 1
    for tname, tseries in targets.items():
        if pd.api.types.is_numeric_dtype(tseries) and tseries.nunique(dropna=True) > 20:
            continue  # treat as continuous
        # Reuse the package's imbalance() rather than re-deriving the class balance.
        result = imbalance(tseries)
        summary = result.attrs.get("summary", {})
        if summary.get("n_classes", 0) < 2:
            continue
        minority = summary["minority_class"]
        min_pct = dict(zip(result["class"].tolist(), result["pct"].tolist())).get(minority)
        if min_pct is not None and min_pct < 5.0:
            col_label = f"<target: {tname}>" if named else "<target>"
            out.append(Finding(col_label, "info", "imbalance",
                               f"class '{minority}' is only {min_pct:.1f}% of rows",
                               "consider resampling or class weights"))
    return out


# Order here is only the registry order; findings are re-sorted by severity.
CHECKS: list[Callable[[pd.DataFrame, dict], list[Finding]]] = [
    check_missing,
    check_constant,
    check_id_like,
    check_duplicate_rows,
    check_dtype_mismatch,
    check_categorical_consistency,
    check_high_cardinality,
    check_sentinel_missing,
    check_collinearity,
    check_leakage,
    check_target_imbalance,
]


# --------------------------------------------------------------------------- #
# Report object
# --------------------------------------------------------------------------- #
@dataclass
class ProfileReport:
    """The result of :func:`profiler`. Also a plain object you can act on."""

    n_rows: int
    n_cols: int
    findings: list[Finding]
    summary: pd.DataFrame
    target: list = field(default_factory=list)
    target_summary: Optional[pd.DataFrame] = None
    _sparklines: dict = field(default_factory=dict, repr=False)

    # -- filtered views ---------------------------------------------------- #
    @property
    def issues(self) -> list[Finding]:
        return self.findings

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "info"]

    @property
    def health(self) -> int:
        penalty = sum(_SEVERITY_PENALTY[f.severity] for f in self.findings)
        return int(max(0, 100 - penalty))

    def to_frame(self) -> pd.DataFrame:
        """Findings as a tidy DataFrame (empty if the data is clean)."""
        return pd.DataFrame(
            [(f.severity, f.code, f.column, f.message, f.suggestion)
             for f in self.findings],
            columns=["severity", "code", "column", "message", "suggestion"],
        )

    # -- rendering --------------------------------------------------------- #
    def _header(self) -> str:
        e, w, i = len(self.errors), len(self.warnings), len(self.infos)
        parts = [f"{self.n_rows:,} rows \u00d7 {self.n_cols} cols"]
        if self.target:
            label = "target" if len(self.target) == 1 else "targets"
            parts.append(f"{label}: {', '.join(map(str, self.target))}")
        parts.append(f"health {self.health}/100")
        parts.append(f"\u2717 {e} errors  \u26a0 {w} warnings  \u2139 {i} info")
        return "  \u00b7  ".join(parts)

    def _text_table(self, summary) -> list:
        lines = [f"{'Column':<20} {'Type':<8} {'Distribution':<14} "
                 f"{'Missing':>9} {'Has Missing':>11} {'Distinct':>10}"]
        for _, row in summary.iterrows():
            spark = self._sparklines.get(row["column"], "")
            miss, has_missing = _format_missing(
                row["missing_pct"], row["has_missing"],
            )
            lines.append(
                f"{str(row['column'])[:20]:<20} {str(row['dtype'])[:8]:<8} "
                f"{spark:<14} {miss:>9} {has_missing:>11} "
                f"{row['cardinality']:>10,}"
            )
        return lines

    def _target_text_table(self, summary) -> list:
        lines = [f"{'Column':<16} {'Type':<8} {'Inferred Role':<15} "
                 f"{'Missing':>9} {'Has Missing':>11} {'Distinct':>9}  "
                 "Balance / shape"]
        for _, row in summary.iterrows():
            miss, has_missing = _format_missing(
                row["missing_pct"], row["has_missing"],
            )
            lines.append(
                f"{str(row['column'])[:16]:<16} {str(row['dtype'])[:8]:<8} "
                f"{str(row['inferred_role']):<15} {miss:>9} "
                f"{has_missing:>11} {row['cardinality']:>9,}  {row['balance']}"
            )
        return lines

    def __str__(self) -> str:
        lines = [self._header(), "", "FINDINGS"]
        if self.findings:
            lines.append(f"{'Severity':<9} {'Column':<20} {'Issue':<40} Suggested fix")
            for f in self.findings:
                lines.append(
                    f"{f.severity:<9} {str(f.column)[:20]:<20} "
                    f"{str(f.message)[:40]:<40} {f.suggestion}"
                )
        else:
            lines.append("none, the data looks clean")
        lines.append("")
        if self.target_summary is not None and not self.target_summary.empty:
            lines.append("TARGET")
            lines.extend(self._target_text_table(self.target_summary))
            lines.append("")
        lines.append("COLUMNS")
        lines.extend(self._text_table(self.summary))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def _html_summary_table(self, summary) -> str:
        th = ("text-align:left;padding:3px 12px;border-bottom:1px solid #8884;"
              "font-weight:600;color:#888;font-size:11px")
        rows = ""
        for _, r in summary.iterrows():
            spark = self._sparklines.get(r["column"], "")
            miss, has_missing = _format_missing(
                r["missing_pct"], r["has_missing"],
            )
            rows += (
                "<tr>"
                f'<td style="font-family:monospace;padding:2px 12px">{html.escape(str(r["column"]))}</td>'
                f'<td style="padding:2px 12px;color:#888">{html.escape(str(r["dtype"]))}</td>'
                f'<td style="font-family:monospace;padding:2px 12px">{spark}</td>'
                f'<td style="padding:2px 12px">{miss}</td>'
                f'<td style="padding:2px 12px">{has_missing}</td>'
                f'<td style="padding:2px 12px">{r["cardinality"]:,}</td>'
                "</tr>"
            )
        return (
            '<table style="border-collapse:collapse;margin-bottom:12px">'
            f'<thead><tr><th style="{th}">Column</th><th style="{th}">Type</th>'
            f'<th style="{th}">Distribution</th><th style="{th}">Missing</th>'
            f'<th style="{th}">Has Missing</th>'
            f'<th style="{th}">Distinct</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )

    def _target_html_table(self, summary) -> str:
        th = ("text-align:left;padding:3px 12px;border-bottom:1px solid #8884;"
              "font-weight:600;color:#888;font-size:11px")
        rows = ""
        for _, r in summary.iterrows():
            miss, has_missing = _format_missing(
                r["missing_pct"], r["has_missing"],
            )
            rows += (
                "<tr>"
                f'<td style="font-family:monospace;padding:2px 12px">{html.escape(str(r["column"]))}</td>'
                f'<td style="padding:2px 12px;color:#888">{html.escape(str(r["dtype"]))}</td>'
                f'<td style="padding:2px 12px">{html.escape(str(r["inferred_role"]))}</td>'
                f'<td style="padding:2px 12px">{miss}</td>'
                f'<td style="padding:2px 12px">{has_missing}</td>'
                f'<td style="padding:2px 12px">{r["cardinality"]:,}</td>'
                f'<td style="padding:2px 12px">{html.escape(str(r["balance"]))}</td>'
                "</tr>"
            )
        return (
            '<table style="border-collapse:collapse;margin-bottom:12px">'
            f'<thead><tr><th style="{th}">Column</th><th style="{th}">Type</th>'
            f'<th style="{th}">Inferred Role</th><th style="{th}">Missing</th>'
            f'<th style="{th}">Has Missing</th>'
            f'<th style="{th}">Distinct</th><th style="{th}">Balance / shape</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
        )

    def _repr_html_(self) -> str:
        colors = {"error": "#d33", "warning": "#e69500", "info": "#3a7"}
        th = ("text-align:left;padding:3px 12px;border-bottom:1px solid #8884;"
              "font-weight:600;color:#888;font-size:11px")

        find_rows = ""
        for f in self.findings:
            c = colors.get(f.severity, "#888")
            find_rows += (
                "<tr>"
                f'<td style="color:{c};font-weight:600;padding:2px 12px">{f.severity}</td>'
                f'<td style="font-family:monospace;padding:2px 12px">{html.escape(str(f.column))}</td>'
                f'<td style="padding:2px 12px">{html.escape(str(f.message))}</td>'
                f'<td style="padding:2px 12px;color:#888">{html.escape(str(f.suggestion))}</td>'
                "</tr>"
            )
        if not find_rows:
            find_rows = ('<tr><td colspan="4" style="padding:6px 12px;color:#3a7">'
                         "no issues found \u2713</td></tr>")
        findings_table = (
            '<table style="border-collapse:collapse;margin-bottom:12px">'
            f'<thead><tr><th style="{th}">Severity</th><th style="{th}">Column</th>'
            f'<th style="{th}">Issue</th><th style="{th}">Suggested fix</th></tr></thead>'
            f'<tbody>{find_rows}</tbody></table>'
        )

        target_html = ""
        if self.target_summary is not None and not self.target_summary.empty:
            target_html = (
                '<div style="color:#888;font-size:11px;margin-bottom:2px">TARGET</div>'
                + self._target_html_table(self.target_summary)
            )

        return f"""
<div style="font-family:-apple-system,sans-serif;font-size:13px">
  <div style="font-weight:600;margin-bottom:8px">{self._header()}</div>
  <div style="color:#888;font-size:11px;margin-bottom:2px">FINDINGS</div>
  {findings_table}
  {target_html}
  <div style="color:#888;font-size:11px;margin-bottom:2px">COLUMNS</div>
  {self._html_summary_table(self.summary)}
</div>"""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def profiler(data, target=None) -> ProfileReport:
    """Diagnose a DataFrame: rank its problems worst-first, with fixes.

    Parameters
    ----------
    data
        A pandas or polars DataFrame (or anything convertible to one).
    target
        Optional column name, or a list of names for multi-target frames.
        Each target is set aside from the feature diagnostics and gets leakage
        and class-imbalance checks run against it.

    Returns
    -------
    ProfileReport
        Renders as a compact summary in notebooks/terminals and exposes
        ``.errors``, ``.warnings``, ``.issues`` and ``.to_frame()`` for
        programmatic use.
    """
    df = _to_pandas(data)

    # Normalise target(s) to a list of names, then set them aside.
    if target is None:
        wanted = []
    elif isinstance(target, str):
        wanted = [target]
    else:
        wanted = list(target)

    targets = {}
    for name in wanted:
        if name in df.columns:
            targets[name] = df[name]
        else:
            warnings.warn(f"target {name!r} not found in columns; ignoring",
                          PercentifyWarning, stacklevel=2)
    if targets:
        df = df.drop(columns=list(targets))

    findings: list[Finding] = []
    for check in CHECKS:
        try:
            findings.extend(check(df, targets))
        except Exception as exc:  # a broken check must never sink the report
            warnings.warn(f"{check.__name__} failed: {exc}", PercentifyWarning, stacklevel=2)

    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.column))

    summary, sparks = _summarize(df)
    target_summary = _summarize_target(targets) if targets else None

    return ProfileReport(
        n_rows=len(df), n_cols=df.shape[1], findings=findings,
        summary=summary, target=list(targets), target_summary=target_summary,
        _sparklines=sparks,
    )
