import pytest
import numpy as np
import pandas as pd
from percentify import profiler, ProfileReport, Finding, PercentifyWarning


@pytest.fixture
def messy_df():
    np.random.seed(0)
    base = np.random.randn(200)
    return pd.DataFrame({
        "id": range(200),                       # id-like
        "empty": [None] * 200,                   # entirely missing
        "const": [7] * 200,                      # constant
        "x": base,
        "x_dup": base * 1.0,                     # collinear with x
        "churn": ["No"] * 195 + ["Yes"] * 5,     # 2.5% minority target
    })


def test_profiler_returns_report(messy_df):
    assert isinstance(profiler(messy_df), ProfileReport)


def test_profiler_all_missing_is_error(messy_df):
    codes = {(f.column, f.code) for f in profiler(messy_df).errors}
    assert ("empty", "all_missing") in codes


def test_profiler_flags_constant_and_id(messy_df):
    warn_codes = {f.code for f in profiler(messy_df).warnings}
    assert "constant" in warn_codes
    assert "id_like" in warn_codes
    assert "collinear" in warn_codes


def test_profiler_target_imbalance(messy_df):
    infos = {f.code for f in profiler(messy_df, target="churn").infos}
    assert "imbalance" in infos


def test_profiler_to_frame(messy_df):
    frame = profiler(messy_df).to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["severity", "code", "column", "message", "suggestion"]


def test_profiler_findings_are_findings(messy_df):
    assert all(isinstance(f, Finding) for f in profiler(messy_df).findings)


def test_profiler_clean_data_full_health():
    clean = pd.DataFrame({
        "a": [x % 10 for x in range(30)],
        "b": [x % 7 for x in range(30)],
    })
    report = profiler(clean)
    assert report.findings == []
    assert report.health == 100


def test_profiler_health_drops_with_issues(messy_df):
    assert profiler(messy_df).health < 100


def test_profiler_target_not_found_warns(messy_df):
    with pytest.warns(PercentifyWarning):
        profiler(messy_df, target="does_not_exist")


def test_profiler_html_escapes_column_names():
    evil = pd.DataFrame({"<script>": [None] * 200, "ok": range(200)})
    rendered = profiler(evil)._repr_html_()
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered


def test_profiler_composes_missing_high_warning():
    # 60% missing should surface as a high_missing warning (via missing()).
    df = pd.DataFrame({"col": [1.0, 2.0] + [None] * 3})
    codes = {f.code for f in profiler(df).warnings}
    assert "high_missing" in codes


def test_profiler_displays_precise_missing_pct_and_flag():
    n = 50_000
    df = pd.DataFrame({
        "twenty_missing": [None] * 20 + [1.0] * (n - 20),
        "one_missing": [None] + [1.0] * (n - 1),
        "clean": [1.0] * n,
    })
    report = profiler(df)
    summary = report.summary.set_index("column")

    assert summary.loc["twenty_missing", "missing_pct"] == 0.04
    assert bool(summary.loc["twenty_missing", "has_missing"])
    assert summary.loc["one_missing", "missing_pct"] == 0.0
    assert bool(summary.loc["one_missing", "has_missing"])
    assert not bool(summary.loc["clean", "has_missing"])

    text = str(report)
    twenty_line = next(line for line in text.splitlines() if line.startswith("twenty_missing"))
    one_line = next(line for line in text.splitlines() if line.startswith("one_missing"))
    clean_line = next(line for line in text.splitlines() if line.startswith("clean"))
    assert "Has Missing" in text
    assert "0.04%" in twenty_line and "Yes" in twenty_line
    assert "0.00%" in one_line and "Yes" in one_line
    assert "0.00%" in clean_line and "No" in clean_line

    rendered = report._repr_html_()
    assert "Has Missing" in rendered
    assert "0.04%" in rendered
    assert ">Yes<" in rendered
    assert ">No<" in rendered


def test_profiler_polars_input(messy_df):
    pl = pytest.importorskip("polars")
    report = profiler(pl.from_pandas(messy_df), target="churn")
    assert isinstance(report, ProfileReport)
    assert any(f.code == "all_missing" for f in report.errors)


# ===== target visibility and list targets =====

def test_profiler_single_target_visible():
    np.random.seed(0)
    df = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50), "target": np.random.randn(50)})
    report = profiler(df, target="target")
    assert report.target == ["target"]
    assert "target" not in report.summary["column"].tolist()          # excluded from features
    assert "target" in report.target_summary["column"].tolist()       # shown in TARGET section
    assert "target: target" in str(report)                            # named in the header


def test_profiler_target_uses_missing_report_and_two_decimals():
    n = 1_000
    target = np.arange(n, dtype=float)
    target[0] = np.nan
    report = profiler(
        pd.DataFrame({"feature": np.arange(n) % 17, "target": target}),
        target="target",
    )
    row = report.target_summary.set_index("column").loc["target"]

    assert row["missing_pct"] == 0.1
    assert bool(row["has_missing"])
    target_line = next(line for line in str(report).splitlines() if line.startswith("target"))
    assert "0.10%" in target_line
    assert "Yes" in target_line


def test_profiler_accepts_target_list():
    np.random.seed(0)
    df = pd.DataFrame({
        "a": np.random.randn(60),
        "price": np.random.randn(60),
        "label": np.random.choice(["x", "y"], 60),
    })
    report = profiler(df, target=["price", "label"])
    assert report.target == ["price", "label"]
    assert "price" not in report.summary["column"].tolist()
    assert "label" not in report.summary["column"].tolist()
    assert set(report.target_summary["column"]) == {"price", "label"}
    assert "targets: price, label" in str(report)


def test_profiler_multi_target_imbalance_names_target():
    np.random.seed(0)
    n = 400
    df = pd.DataFrame({
        "a": np.random.randn(n),
        "y1": np.random.randn(n),                       # continuous, no imbalance
        "y2": ["yes"] * 8 + ["no"] * (n - 8),           # 2% minority
    })
    infos = profiler(df, target=["y1", "y2"]).infos
    assert any(f.code == "imbalance" and "y2" in f.column for f in infos)


def test_profiler_no_target_has_no_target_summary():
    report = profiler(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert report.target == []
    assert report.target_summary is None


def test_profiler_polars_target_list():
    pl = pytest.importorskip("polars")
    np.random.seed(0)
    df = pl.DataFrame({
        "a": np.random.randn(60),
        "price": np.random.randn(60),
        "label": np.random.choice(["x", "y"], 60),
    })
    report = profiler(df, target=["price", "label"])
    assert report.target == ["price", "label"]
    assert set(report.target_summary["column"]) == {"price", "label"}
    assert "price" not in report.summary["column"].tolist()


def test_profiler_target_role_and_balance():
    np.random.seed(0)
    df = pd.DataFrame({
        "a": np.random.randn(300),
        "price": np.random.lognormal(10, 0.5, 300),   # many distinct -> regression
        "sold": ["yes"] * 30 + ["no"] * 270,           # two classes -> classification
    })
    ts = profiler(df, target=["price", "sold"]).target_summary
    role = dict(zip(ts["column"], ts["inferred_role"]))
    balance = dict(zip(ts["column"], ts["balance"]))
    assert role["price"] == "regression"
    assert role["sold"] == "classification"
    assert "skew" in balance["price"]        # regression reports skew
    assert "ratio" in balance["sold"]        # classification reports the imbalance ratio
    assert "majority" in balance["sold"]     # verbose: the top class is included


def test_profiler_object_scattered_target_is_regression():
    # Scattered continuous data that arrived as object/string must be scored as
    # regression (via numeric coercion), not crash on .skew() or fall through
    # to a meaningless class breakdown.
    np.random.seed(0)
    values = np.random.lognormal(8, 0.4, 300).round(2)
    df = pd.DataFrame({
        "a": np.random.randn(300),
        "amount": pd.Series([str(v) for v in values], dtype=object),  # many distinct
    })
    assert df["amount"].dtype == object
    ts = profiler(df, target="amount").target_summary
    row = ts[ts["column"] == "amount"].iloc[0]
    assert row["inferred_role"] == "regression"
    assert "skew" in row["balance"]           # skew computed on the coerced numbers


def test_profiler_object_highcardinality_text_target_no_crash():
    # Genuinely non-numeric, high-cardinality text: must not crash and must not
    # be summarised as a fake two-class imbalance.
    df = pd.DataFrame({
        "a": range(60),
        "note": [f"free text {i}" for i in range(60)],   # 60 distinct, non-numeric
    })
    ts = profiler(df, target="note").target_summary
    row = ts[ts["column"] == "note"].iloc[0]
    assert "further diagnostic required" in row["balance"]
    assert "majority" not in row["balance"]
