<p align="center">
  <img src="https://raw.githubusercontent.com/data-centt/percentify/main/asset/log.png" alt="Percentify logo" height="140">
</p>

[![PyPI version](https://img.shields.io/pypi/v/percentify.svg?style=flat&color=blue)](https://pypi.org/project/percentify/)
[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-green?style=flat)](https://pypi.org/project/percentify/)
[![License](https://img.shields.io/pypi/l/percentify.svg?style=flat&color=orange)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-percentify-14b8a6)](https://data-centt.github.io/percentify/)
[![Build Status](https://github.com/data-centt/percentify/actions/workflows/python-app.yml/badge.svg)](https://github.com/data-centt/percentify/actions/workflows/python-app.yml)
[![Polars](https://img.shields.io/badge/Polars-supported-cd792c?style=flat)](https://data-centt.github.io/percentify/)

**80% of the checks you run on every dataset. 20% of the code.**

Exploratory stats and data-quality diagnostics for pandas and **Polars** DataFrames. One call each.

> [!TIP]
> **⚡ Polars is first-class, not an afterthought.** Pass a Polars DataFrame or Series and get the same kind straight back, with no flag and no manual conversion. Every function works on both backends, which is what sets Percentify apart from the pandas-only tools.

Where a function wraps an existing library (pandas, scipy, statsmodels, scikit-learn), it names it, so you always know where to dig deeper.

## ⭐  `profiler`

**pandas `.describe()` tells you what your data _is_. `profiler()` tells you what to _do_ about it:** every issue ranked worst-first, each with its fix.

```python
from percentify import profiler

report = profiler(df, target="churn")

report.to_frame()          # every finding, ranked worst-first, with a suggested fix
report.errors              # just the blocking issues
report.health              # a 0 to 100 data-health score
assert not report.errors   # drop it straight into a CI data-quality gate
```

Point it at any messy DataFrame, pandas or Polars, and see what it flags before you model. [Try it →](https://data-centt.github.io/percentify/documentation/#profiler)

# 📖 Documentation

**Full guides → [data-centt.github.io/percentify](https://data-centt.github.io/percentify/)**

# 📦 Installation

```bash
pip install percentify
```

Requires Python 3.10+, `numpy`, and `pandas` 2.0+.

## How to use percentify

Import the function that matches the question you want to answer, pass in a pandas or Polars object, and use the returned DataFrame or scalar directly in your notebook, report, or pipeline.

```python
from percentify import missing, profiler

missing(df)                  # quick column-level check
profiler(df, target="churn")  # ranked data-quality issues and fixes
```

## Quick example

```python
import pandas as pd
from percentify import missing

df = pd.DataFrame({
    "salary": [50000, None, 60000, None],
    "age":    [25, 30, None, 40],
    "city":   ["NY", "LA", "SF", "LA"],
})

missing(df)
#    column  missing_pct
# 0  salary         50.0
# 1     age         25.0
# 2    city          0.0
```

One import, one line. A clean, sorted DataFrame you can read or feed into the next step.

# 🤝 Contributing

Contributions are welcome, provided they align with the repository’s guiding principles. Please review the [contributing](https://github.com/data-centt/percentify/blob/main/CONTRIBUTING.md) guidelines before submitting.


# More Examples

These are short, recipe-style examples that go beyond the one-liner above and are intentionally not covered in the [documentation](https://data-centt.github.io/percentify/documentation/). The docs show each function in isolation; these show how to chain them into a real workflow.

### A 30-second data-quality gate

Drop this into CI to block training on a dataset that isn't ready:

```python
import pandas as pd
from percentify import profiler

df = pd.read_parquet("train.parquet")
report = profiler(df, target="label")

assert report.errors.empty, report.to_frame()
assert report.health >= 80, f"health too low: {report.health}"
```

#### Rank correlations by significance

Pull the pairs that are both strong *and* unlikely to be noise:

```python
import pandas as pd
from percentify import correlate

df = pd.DataFrame({
    "x":   range(50),
    "y":   [v * 0.9 + (v % 3) for v in range(50)],
    "z":   [v * 0.05 for v in range(50)],
    "w":   [v % 7 for v in range(50)],
})

print(correlate(df).sort_values("p_value").head(5))
```

#### Build a transform pipeline from `skew_report`

Let `skew_report` tell you what to apply, then apply it:

```python
import numpy as np
import pandas as pd
from percentify import skew_report

df = pd.DataFrame({
    "income": [30_000, 35_000, 1_200_000, 40_000, 28_000],
    "visits": [1, 1, 1, 50, 2],
})

plan = skew_report(df)
print(plan[["feature", "skew", "suggested_transform"]])
#    feature   skew suggested_transform
# 0   income  2.27              log1p
# 1   visits  2.19              log1p

df["income_log"] = np.log1p(df["income"])     # numpy / pandas, not percentify
df["visits_log"] = np.log1p(df["visits"])
```

#### Interpret PCA with both calls

Variance tells you *how much* of the signal each axis carries; loadings tell you *what it means*:

```python
import pandas as pd
from percentify import pca_variance, pca_loadings

df = pd.DataFrame({
    "height_cm": [160, 170, 180, 175, 165],
    "weight_kg": [55,  68,  82,  74,  60],
    "age":       [25,  35,  45,  30,  28],
})

print(pca_variance(df))    # PC1 carries most of the variance
print(pca_loadings(df))    # PC1 = (height, weight) with similar signs
```

#### Drop collinear columns before modelling

Use `vif` with a threshold to get a drop-list you can feed straight into `df.drop`:

```python
import pandas as pd
from percentify import vif

df = pd.DataFrame({
    "price":  [10, 12, 11, 13, 9,  14, 8,  12],
    "cost":   [ 6,  7,  7,  8, 5,   8, 4,   7],   # tracks price
    "margin": [ 4,  5,  4,  5, 4,   6, 4,   5],   # = price - cost
    "stock":  [100, 80, 90, 70, 110, 60, 120, 85],
})

to_drop = vif(df, flag=5.0)["feature"].tolist()
print(to_drop)                # e.g. ['cost', 'margin']
clean = df.drop(columns=to_drop)
```

#### Month-over-month KPI table

`change` over a DataFrame applies period-over-period growth to every numeric column at once:

```python
import pandas as pd
from percentify import change

kpis = pd.DataFrame({
    "revenue": [100, 120, 150, 135, 180],
    "signups": [400, 420, 470, 460, 510],
}, index=["Jan", "Feb", "Mar", "Apr", "May"])

print(change(kpis))
```

- [Worked examples for every function](https://data-centt.github.io/percentify/documentation/)
- [Project documentation](https://data-centt.github.io/percentify/)

# What's inside

| Function | What it answers |
|---|---|
| `profiler` | What is wrong with this dataset, and how do I fix it? |
| `change` | Growth as numbers, columns, or a whole series |
| `vif` | Which features are collinear? |
| `missing` | How much of each column is missing? |
| `cv` | How variable is each column, relative to its mean? |
| `outliers` | What percentage of each column are outliers? |
| `pca_variance` | How much variance does each principal component explain? |
| `pca_loadings` | What does each principal component consist of? |
| `imbalance` | How skewed are the classes in a target column? |
| `correlate` | Which features move together, and is it significant? |
| `skew_report` | How skewed is each column, and what transform helps? |
| `bootstrap_ci` | What is the confidence interval for a statistic? |
| `permutation_test` | Are two groups really different? (a p-value) |
| `effect_size` | How big is the difference, not just whether it is significant? |
| `difference` | How far apart are two values or columns? |
| `split` | How does a total divide across weights or groups? |
| `display` | Format numbers or a column as clean "%" strings |

→ See the **[documentation](https://data-centt.github.io/percentify/)** for a worked, real-output example of every function.
