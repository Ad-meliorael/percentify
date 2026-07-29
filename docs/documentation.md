# Documentation

Every example on this page is real: the output shown is exactly what the function returns.

## Installation

```bash
pip install percentify
```

Percentify requires Python 3.10+, `numpy`, and `pandas` 2.0+.

```python
from percentify import change, vif, missing, cv, outliers, pca_variance, pca_loadings
```

---

## Conventions

A few rules hold across the whole library, so you always know what to expect:

- **DataFrame in → DataFrame out.** Pass a DataFrame and you get a clean DataFrame back. Pass a single `Series` (where it makes sense) and you get a single number.
- **Sorted worst-first.** Results come back ordered by what usually matters most: most collinear, most missing, most variable, most outliers.
- **`decimals` everywhere.** Every function accepts a `decimals` argument (default `2`). Pass `decimals=None` for full precision.
- **Numeric-only functions ignore text columns.** They quietly skip non-numeric columns instead of crashing.
- **Friendly warnings, not tracebacks.** Hand a function something it can't use (e.g. an all-text DataFrame) and you get a `PercentifyWarning` explaining why, plus an empty result, never a raw NumPy/pandas stack trace. See [Warnings](#warnings-and-non-numeric-data).

---

## Polars support

Every function accepts **polars** DataFrames and Series as well as pandas, and hands back the same kind you passed in: polars in, polars out. Detection is automatic from the input type, so there is nothing to configure.

```python
import polars as pl
from percentify import missing

df = pl.DataFrame({"salary": [50000, None, 60000, None], "age": [25, 30, None, 40]})

missing(df)   # returns a polars DataFrame
```

polars stays optional: it is only imported when you actually pass a polars object, so pandas-only users pay nothing. Conversion goes through Arrow, so `pyarrow` must be installed for the polars path (it usually already is in a polars setup).

---

## `profiler`

The flagship, a one-call **data diagnostician**. pandas `.describe()` tells you what your data *is*; `profiler()` tells you what to *do* about it: every issue ranked worst-first, each with its fix. Instead of dumping statistics for every column, it runs a battery of checks, scores the data's health, and composes the rest of the toolkit (`missing`, `imbalance`, and more) under one entry point.

!!! tip "Similar concept"
    `ydata-profiling` / `pandas.DataFrame.describe`, but diagnostic (it ranks problems and suggests fixes) rather than descriptive.

**Signature**

```python
profiler(data, target=None)
```

**Example**

```python
import pandas as pd
from percentify import profiler

df = pd.DataFrame({
    "user_id": range(100),                                              # identifier
    "plan":    ["free"] * 100,                                          # constant
    "age":     [20 + (i * 37) % 40 for i in range(100)],
    "spend":   [None if i % 2 == 0 else float((i * 13) % 100) for i in range(100)],
    "churn":   ["No"] * 96 + ["Yes"] * 4,                              # imbalanced target
})

report = profiler(df, target="churn")
report.health        # 87
report.to_frame()
```

```text
severity         code   column                          message                           suggestion
 warning     constant     plan          only one distinct value         drop, carries no information
 warning high_missing    spend                      50% missing                       impute or drop
 warning      id_like  user_id identifier-like (100/100 unique)          further diagnostic required
    info    imbalance <target> class 'Yes' is only 4.0% of rows consider resampling or class weights
```

The report renders as a compact, color-coded summary in notebooks and terminals, and exposes everything programmatically:

- `report.errors`, `report.warnings`, `report.infos`: findings filtered by severity.
- `report.to_frame()`: all findings as a tidy DataFrame.
- `report.health`: a 0 to 100 score (100 is clean).
- `assert not report.errors`: drop it straight into a CI data-quality gate.

Pass `target=` to also check for **leakage** (features that predict a target almost perfectly) and class imbalance. It accepts a single column name **or a list of names** for multi-target frames. Each target is set aside from the feature diagnostics and gets its own `TARGET` section showing its **inferred role** (classification or regression) and a **balance / shape** readout (the class balance for classification, the skew for regression), with the checks run against each. A numeric target that arrived as text (scattered values stored as strings) is coerced and read as regression, not mistaken for a giant class list. Accepts pandas or polars input.

```python
profiler(df, target="price")              # one target
profiler(df, target=["price", "sold"])    # two targets, each set aside and checked
```

!!! example "Try it on your own data"
    Point `profiler()` at any messy dataset you have lying around, pandas or Polars, and read the findings top to bottom. It is the fastest way to see what to fix before you model.

---

## `change`

Percentage change: as two numbers, between two columns, or down a whole series.

!!! tip "Similar concept"
    `pandas.DataFrame.pct_change`

**Signature**

```python
change(old, new=None, decimals=2)
```

**Two numbers**

```python
from percentify import change

change(100, 150)
```

```text
50.0
```

**Between two columns** (element-wise), perfect for a new column:

```python
import pandas as pd

df = pd.DataFrame({
    "forecast": [100, 200, 50, 80],
    "actual":   [150, 150, 100, 80],
})

df["change_pct"] = change(df["forecast"], df["actual"])
df
```

```text
   forecast  actual  change_pct
0       100     150        50.0
1       200     150       -25.0
2        50     100       100.0
3        80      80         0.0
```

**Down one column** (period-over-period):

```python
change(pd.Series([100, 150, 90, 135]))
```

```text
0     NaN
1    50.0
2   -40.0
3    50.0
dtype: float64
```

The first value is `NaN` because there is no prior period to compare against. Passing a whole `DataFrame` applies this to every numeric column at once.

!!! note
    Where `old` is `0`, the result is `0.0` (safe division). Two Series are aligned by **index**, not position, which is fine when both columns come from the same DataFrame.

---

## `vif`

Variance Inflation Factor: the classic multicollinearity check, without the six-line loop.

!!! tip "Similar concept"
    `statsmodels.stats.outliers_influence.variance_inflation_factor`

**Signature**

```python
vif(df, decimals=2, flag=None)
```

**Example**

```python
import numpy as np, pandas as pd
from percentify import vif

np.random.seed(7)
base = np.random.randn(200)
df = pd.DataFrame({
    "age":    base + np.random.randn(200) * 0.1,
    "income": base * 2 + np.random.randn(200) * 0.1,   # closely tracks age
    "score":  np.random.randn(200),
})

vif(df)
```

```text
  feature    VIF
0  income  83.54
1     age  83.53
2   score   1.01
```

`age` and `income` have sky-high VIFs (they carry the same signal), while `score` is independent.

**Flag only the problem columns** with `flag`:

```python
vif(df, flag=5.0)
```

```text
  feature    VIF
0  income  83.54
1     age  83.53
```

!!! info "Rules of thumb"
    VIF > 5 suggests moderate multicollinearity; VIF > 10 suggests severe. A perfectly collinear column returns `inf`.

---

## `missing`

The percentage of missing values in each column, sorted highest first.

!!! tip "Similar concept"
    `pandas.DataFrame.isna`

**Signature**

```python
missing(df, decimals=2)
```

**Example**

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
   column  missing_pct
0  salary         50.0
1     age         25.0
2    city          0.0
```

Unlike the numeric-only functions, `missing` reports on **every** column, text included.

---

## `cv`

Coefficient of variation: relative variability (`std ÷ mean`), as a percentage. Handy for comparing spread across columns on different scales.

!!! tip "Similar concept"
    `scipy.stats.variation`

**Signature**

```python
cv(data, decimals=2)
```

**A single column returns a number**

```python
import pandas as pd
from percentify import cv

cv(pd.Series([50000, 52000, 48000, 90000, 51000]))
```

```text
30.65
```

**A DataFrame returns every numeric column, most variable first**

```python
df = pd.DataFrame({
    "salary": [50000, 52000, 48000, 90000, 51000],
    "age":    [25, 30, 35, 40, 45],
})

cv(df)
```

```text
  feature     cv
0  salary  30.65
1     age  22.59
```

---

## `outliers`

The percentage of values that are outliers by the IQR method (below `Q1 − 1.5·IQR` or above `Q3 + 1.5·IQR`).

!!! tip "Similar concept"
    `scipy.stats.iqr`

**Signature**

```python
outliers(data, decimals=2, multiplier=1.5)
```

**A single column returns a number**

```python
import pandas as pd
from percentify import outliers

outliers(pd.Series([10, 11, 12, 13, 14, 15, 200]))
```

```text
14.29
```

One value out of seven (`200`) is an outlier → 14.29%.

**A DataFrame returns every numeric column**

```python
df = pd.DataFrame({
    "salary": [10, 11, 12, 13, 14, 15, 200],
    "age":    [25, 26, 27, 28, 29, 30, 31],
})

outliers(df)
```

```text
  feature  outlier_pct
0  salary        14.29
1     age         0.00
```

Tune the sensitivity with `multiplier` (e.g. `multiplier=3.0` for a looser bound).

---

## `pca_variance`

The percentage of variance explained by each principal component, plus a running cumulative total.

!!! tip "Similar concept"
    `sklearn.decomposition.PCA` (`.explained_variance_ratio_`)

**Signature**

```python
pca_variance(df, decimals=2, n_components=None, standardize=True)
```

**Example**

```python
import numpy as np, pandas as pd
from percentify import pca_variance

np.random.seed(7)
base = np.random.randn(200)
df = pd.DataFrame({
    "height": base + np.random.randn(200) * 0.3,
    "weight": base + np.random.randn(200) * 0.3,   # shares a signal with height
    "noise":  np.random.randn(200),
})

pca_variance(df)
```

```text
  component  variance_explained  cumulative
0       PC1               64.04       64.04
1       PC2               33.34       97.38
2       PC3                2.62      100.00
```

Read the `cumulative` column to decide how many components to keep. Here, PC1 + PC2 already capture 97.4% of the variance.

!!! warning "Standardization matters"
    By default `standardize=True`, so each column is scaled to unit variance first. This stops a column measured in large units (e.g. dollars) from dominating purely because of its scale. Pass `standardize=False` for covariance-based PCA on the raw values.

---

## `pca_loadings`

Principal component loadings: what each component is *made of*. Where `pca_variance` tells you how important each PC is, `pca_loadings` shows how much each original feature contributes to it, as a feature-by-component table.

!!! tip "Similar concept"
    `sklearn.decomposition.PCA` (`.components_`)

**Signature**

```python
pca_loadings(df, decimals=2, n_components=None, standardize=True)
```

**Example**

```python
import numpy as np, pandas as pd
from percentify import pca_loadings

np.random.seed(7)
base = np.random.randn(200)
df = pd.DataFrame({
    "height": base + np.random.randn(200) * 0.3,
    "weight": base + np.random.randn(200) * 0.3,   # shares a signal with height
    "noise":  np.random.randn(200),
})

pca_loadings(df)
```

```text
feature  PC1   PC2   PC3
 height 0.71  0.01 -0.71
 weight 0.71 -0.02  0.71
  noise 0.01  1.00  0.02
```

Read down a column: PC1 is `height` and `weight` together (both `0.71`), while `noise` sits near `0`, so PC1 is the shared height/weight signal. Standardized by default, matching `pca_variance`.

!!! note "Sign is arbitrary"
    A component and its negation are equivalent, so the sign of a loading is not meaningful on its own. Only the relative signs and magnitudes *within* a column matter.

---

## `imbalance`

Summarize class imbalance in a categorical target: the per-class breakdown plus the headline metrics you actually report.

!!! tip "Similar concept"
    `pandas.Series.value_counts`

**Signature**

```python
imbalance(data, decimals=2)
```

**Example**

```python
import pandas as pd
from percentify import imbalance

df = pd.DataFrame({"churn": ["No"] * 850 + ["Yes"] * 150})

result = imbalance(df["churn"])
result
```

```text
  class  count   pct
0    No    850  85.0
1   Yes    150  15.0
```

The headline metrics come attached on `.attrs["summary"]`, so the return stays a clean DataFrame:

```python
result.attrs["summary"]
```

```text
{'n_classes': 2,
 'majority_class': 'No',
 'minority_class': 'Yes',
 'imbalance_ratio': 5.67,
 'entropy_pct': 60.98}
```

- **`imbalance_ratio`**: the majority count divided by the minority count (`5.67` means "No" is 5.7x more common than "Yes").
- **`entropy_pct`**: `100` for a perfectly balanced target, approaching `0` as one class dominates.

Pass a single column (`df["target"]`), not the whole DataFrame. Nulls are dropped before counting.

---

## `correlate`

Correlation with p-values, the piece `df.corr()` leaves out. Pass two Series for a single `(r, p)`, or a whole DataFrame for a tidy table of every numeric pair, strongest first.

!!! tip "Similar concept"
    `scipy.stats.pearsonr` / `scipy.stats.spearmanr`

**Signature**

```python
correlate(a, b=None, method="pearson", decimals=2)
```

**Two columns return `(r, p)`**

```python
import numpy as np, pandas as pd
from percentify import correlate

np.random.seed(7)
base = np.random.randn(200)
df = pd.DataFrame({
    "age":    base,
    "income": base * 2 + np.random.randn(200) * 0.3,   # tracks age
    "score":  np.random.randn(200),
})

correlate(df["age"], df["income"])   # (0.99, 1.7e-166)
```

**A DataFrame returns a ranked table**

```python
correlate(df)
```

```text
feature_1 feature_2    r             p
      age    income 0.99 1.700000e-166
      age     score 0.05  4.900000e-01
   income     score 0.05  4.900000e-01
```

Pass `method="spearman"` for rank (monotonic) correlation.

!!! note "Reading the p-value"
    A p-value is never exactly zero, so `correlate` never reports one. When `p`
    is smaller than the `decimals` resolution it keeps two significant figures
    (`1.7e-166`) instead of collapsing to a misleading `0.00`. One very small
    value puts the whole column into scientific notation, so `4.900000e-01` is
    just `0.49`.

    Read `r` for the strength and `p` only for "is this distinguishable from
    zero". On a large sample almost any correlation is significant, so a tiny
    `p` is not evidence of a strong relationship: `r = 0.05` with
    `p = 1e-20` is still a negligible relationship.

---

## `skew_report`

Distribution shape for every numeric column: skew, kurtosis, outlier percentage, and a suggested transform. Most-skewed first.

!!! tip "Similar concept"
    `pandas.DataFrame.skew` + `pandas.DataFrame.kurt`

**Signature**

```python
skew_report(df, decimals=2)
```

**Example**

```python
import numpy as np, pandas as pd
from percentify import skew_report

np.random.seed(1)
df = pd.DataFrame({
    "income": np.random.exponential(2, 500),      # right-skewed
    "age":    np.random.normal(40, 10, 500),      # roughly symmetric
})

skew_report(df)
```

```text
feature  skew  kurtosis  outlier_pct suggested_transform
 income  1.55      2.77          3.8               log1p
    age -0.01      0.24          0.8                none
```

`suggested_transform` is a starting point: `log1p` for right-skewed non-negative data, `yeo-johnson` for skew with negatives, and `none` when a column is already roughly symmetric. The `outlier_pct` column reuses [`outliers`](#outliers).

---

## `bootstrap_ci`

A confidence interval for any statistic, with no distribution assumed. It resamples the data with replacement and reads the percentiles of the resampled statistic.

!!! tip "Similar concept"
    `scipy.stats.bootstrap`

**Signature**

```python
bootstrap_ci(data, statistic=np.mean, ci=95, n_resamples=1000, decimals=2, random_state=None)
```

**Example**

```python
import numpy as np
from percentify import bootstrap_ci

np.random.seed(1)
income = np.random.exponential(2, 500)

bootstrap_ci(income, random_state=0)   # (1.88, 2.22)
```

Pass any `statistic` (for example `np.median`), change the level with `ci`, and set `random_state` for a reproducible interval.

---

## `permutation_test`

A distribution-free p-value for the difference between two samples. It shuffles the group labels many times and asks how often chance alone produces a difference as large as the one you saw.

!!! tip "Similar concept"
    `scipy.stats.permutation_test`

**Signature**

```python
permutation_test(a, b, statistic=None, n_permutations=1000, decimals=4, random_state=None)
```

**Example**

```python
import numpy as np
from percentify import permutation_test

np.random.seed(1)
a = np.random.normal(100, 15, 120)
b = np.random.normal(106, 15, 120)

permutation_test(a, b, random_state=0)   # 0.001
```

The default statistic is the difference in means; pass your own `statistic(a, b)` for anything else. It returns the number, not a pass or fail verdict, so the judgement stays with you.

!!! note "The smallest p this test can report"
    A permutation test can only resolve down to `1 / (n_permutations + 1)`, so
    with the default 1000 shuffles the floor is `0.001`. That is a limit of the
    shuffling, not proof the effect is that rare. Raise `n_permutations` to
    resolve further. The result is never rounded down to `0.0`.

---

## `effect_size`

How big is the difference, not just whether it is significant. It detects the outcome type and reports the right measure.

!!! tip "Similar concept"
    `pingouin.compute_effsize`

**Signature**

```python
effect_size(df, group, value, decimals=2)
```

**Numeric outcome: Cohen's d, Hedges' g, mean difference**

```python
import numpy as np, pandas as pd
from percentify import effect_size

np.random.seed(1)
df = pd.DataFrame({
    "variant": ["A"] * 500 + ["B"] * 500,
    "revenue": np.concatenate([np.random.normal(100, 20, 500),
                               np.random.normal(112, 20, 500)]),
})

effect_size(df, group="variant", value="revenue")
```

```text
comparison  cohen_d  hedges_g  mean_diff interpretation
    A vs B    -0.58     -0.58     -11.42         medium
```

**Binary outcome (two levels): Cohen's h and lift**

```python
ab = pd.DataFrame({
    "variant":   ["A"] * 100 + ["B"] * 100,
    "converted": [1] * 35 + [0] * 65 + [1] * 20 + [0] * 80,
})

effect_size(ab, group="variant", value="converted")
```

```text
comparison  cohen_h  lift_pct interpretation
    A vs B     0.34      75.0         medium
```

The `interpretation` column labels the magnitude (negligible / small / medium / large), so a raw number is never left without context.

---

## `difference`

Symmetric percentage difference between two values or two columns: how *far apart* they are, regardless of direction. (Reach for `change` when direction matters.)

!!! tip "Similar concept"
    `numpy` / `pandas` element-wise arithmetic

**Signature**

```python
difference(a, b, decimals=2)
```

**Two numbers**

```python
from percentify import difference

difference(10, 20)
```

```text
66.67
```

**Two columns** (element-wise), using the average of the two values as the denominator, so `difference(a, b) == difference(b, a)`:

```python
import pandas as pd

sensors = pd.DataFrame({
    "sensor_a": [10.0, 50.0, 100.0],
    "sensor_b": [12.0, 50.0, 130.0],
})

sensors["pct_gap"] = difference(sensors["sensor_a"], sensors["sensor_b"])
sensors
```

```text
   sensor_a  sensor_b  pct_gap
0      10.0      12.0    18.18
1      50.0      50.0     0.00
2     100.0     130.0    26.09
```

---

## `split`

Distribute a total across weights, proportionally: allocation for budgets, quotas, or apportioning a sum by group.

!!! tip "Similar concept"
    `numpy` / `pandas` weighted arithmetic

**Signature**

```python
split(total, weights, decimals=2)
```

**A list of weights returns a list**

```python
from percentify import split

split(10000, [2, 3, 5])
```

```text
[2000.0, 3000.0, 5000.0]
```

**A column of weights returns an aligned Series**, allocating a budget by population:

```python
import pandas as pd

budget = pd.DataFrame({
    "region": ["North", "South", "East"],
    "population": [200, 300, 500],
})

budget["budget"] = split(10000, budget["population"])
budget
```

```text
  region  population  budget
0  North         200  2000.0
1  South         300  3000.0
2   East         500  5000.0
```

Raises `ValueError` if `weights` is empty or sums to zero.

---

## `display`

Format a number, or a whole column, as clean percentage strings. The "last mile" for reports, dashboards, and exports.

!!! tip "Similar concept"
    `pandas.Series.map` + Python string formatting

**Signature**

```python
display(value, decimals=2, suffix="%", multiply=False)
```

**A single number returns a string**

```python
from percentify import display

display(0.45, multiply=True)
```

```text
'45.0%'
```

**A column returns a Series of strings**, turning ratios into report-ready text:

```python
import pandas as pd

rates = pd.DataFrame({
    "campaign": ["A", "B", "C"],
    "conv_rate": [0.045, 0.12, 0.083],
})

rates["display"] = display(rates["conv_rate"], multiply=True)
rates
```

```text
  campaign  conv_rate display
0        A      0.045    4.5%
1        B      0.120   12.0%
2        C      0.083    8.3%
```

Use `multiply=True` when your values are ratios (`0.45` → `"45.0%"`); leave it off when they're already percentages (`45` → `"45.0%"`). Customize the trailing text with `suffix`.

---

## Warnings and non-numeric data

Percentify never hands you a raw traceback for a predictable mistake. Give a function data it can't use and it raises a `PercentifyWarning` and returns an empty result:

```python
import pandas as pd
from percentify import vif

vif(pd.DataFrame({"city": ["NY", "LA"], "team": ["A", "B"]}))
```

```text
PercentifyWarning: Numeric columns required: no numeric columns found. VIF
measures multicollinearity between numeric features - encode any
categorical/text columns first.
```

!!! tip "Leave them on while you learn; switch them off when you ship"
    Instead of a cryptic traceback, you get a friendly nudge about what percentify did and why, so one messy column never dents your momentum. In a trusted production pipeline, silence `PercentifyWarning` to keep your logs quiet.

The warning is a normal Python warning, so you can catch or silence it:

```python
import warnings
from percentify import PercentifyWarning

warnings.filterwarnings("ignore", category=PercentifyWarning)
```
