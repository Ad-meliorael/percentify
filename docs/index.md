<p align="center">
  <img src="asset/log.png" alt="Percentify" width="480">
</p>



**80% of the checks you run on every dataset. 20% of the code.**

Exploratory stats and data-quality diagnostics for pandas and **Polars** DataFrames. One call each.

!!! tip "⚡ Polars is first-class, not an afterthought"
    Pass a Polars DataFrame or Series and get the same kind straight back, with no flag and no manual conversion. Every function works on both backends, which is what sets Percentify apart from the pandas-only tools.

Where a function wraps an existing library (pandas, scipy, statsmodels, scikit-learn), it names it, so you always know where to dig deeper.

---

## Install

```bash
pip install percentify
```

```python
import percentify as pcy
```

Requires Python 3.10+, `numpy`, and `pandas` 2.0+.

## A quick taste

```python
import pandas as pd
from percentify import missing

df = pd.DataFrame({
    "salary": [50000, None, 60000, None],
    "age":    [25, 30, None, 40],
    "city":   ["NY", "LA", "SF", "LA"],
})

missing(df)
```

```text
   column  missing_pct  has_missing
0  salary         50.0         True
1     age         25.0         True
2    city          0.0        False
```

The boolean flag is calculated before percentage rounding, so tiny non-zero
amounts of missing data remain visible even when the percentage is `0.00`.

[Read the full documentation →](documentation.md){ .md-button .md-button--primary }

---

## What's inside

| Function | What it answers |
|---|---|
| `profiler` | What is wrong with this dataset, and how do I fix it? |
| `change` | How much did a value grow, across numbers, columns, or a whole series? |
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
| `difference` | How far apart are two values or columns (regardless of direction)? |
| `split` | How does a total divide across weights or groups? |
| `display` | Format numbers or a column as clean "%" strings for reports. |
