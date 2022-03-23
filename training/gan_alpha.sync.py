# %%
"""Library"""
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from common import (data_path, date_col, missing_code, price_col, projectpath,
                    symbol_col, train_percent, valid_percent)
from sklearn.linear_model import LinearRegression

# %%
select = "factor"
with_macro = False

"""Load data"""
data = pd.read_csv(data_path[select])
data[date_col] = pd.to_datetime(data[date_col])
data = data.set_index([symbol_col, date_col])\
    .replace(missing_code, float("NaN"))\
    .dropna(axis=0)\
    .astype("float")

data.head()

# %%
sdf = pd.read_csv(
    f"{projectpath}/data/results/sdf_uk_{select}_macro_{with_macro}.csv"
)
sdf["date"] = pd.to_datetime(sdf["date"])
sdf = sdf.set_index("date")

sdf.head()

# %%
data = data.combine_first(sdf)

data.head()

# %%
"""One asset"""
subset_data = data.loc["ABC", [price_col, "sdf"]]
train_index = round(subset_data.shape[0] * train_percent)
valid_index = round(subset_data.shape[0] * valid_percent)
train_data = subset_data.iloc[:train_index, :]
valid_data = subset_data.iloc[train_index:valid_index, :]
test_data = subset_data.iloc[valid_index:, :]
mod = sm.OLS(train_data.loc[:, price_col],
             train_data.drop(columns=[price_col]).assign(const=1),
             missing=missing_code,
             hasconst=True)

res = mod.fit()

res.summary()

# %%
"""Sklearn Linear Regression"""
coef = {}

for symbol in data.index.get_level_values("symbol").unique():
    subset_data = data.loc[symbol, [price_col, "sdf"]]

    train_index = round(subset_data.shape[0] * train_percent)
    valid_index = round(subset_data.shape[0] * valid_percent)
    train_data = subset_data.iloc[:train_index, :]
    valid_data = subset_data.iloc[train_index:valid_index, :]
    test_data = subset_data.iloc[valid_index:, :]

    train_X = train_data.drop(columns=[price_col])
    train_y = train_data.loc[:, price_col]
    valid_X = valid_data.drop(columns=[price_col])
    valid_y = valid_data.loc[:, price_col]
    test_X = test_data.drop(columns=[price_col])
    test_y = test_data.loc[:, price_col]

    lr = LinearRegression()\
        .fit(train_X, train_y)

    valid_pred = lr.predict(valid_X)
    valid_intercept = (valid_y - valid_pred).mean()

    test_pred = lr.predict(test_X)
    test_intercept = (test_y - test_pred).mean()

    coef[symbol] = dict(zip(lr.feature_names_in_, lr.coef_))
    coef[symbol]["train_intercept"] = lr.intercept_
    coef[symbol]["valid_intercept"] = valid_intercept
    coef[symbol]["test_intercept"] = test_intercept

# %%
coef = pd.DataFrame(coef).T

coef

# %%
coef["train_intercept"].abs().describe()

# %%
coef["valid_intercept"].abs().describe()

# %%
coef["test_intercept"].abs().describe()

# %%
coef["train_intercept"].plot.hist(density=True, alpha=0.7)
coef["valid_intercept"].plot.hist(density=True, alpha=0.7)
coef["test_intercept"].plot.hist(density=True, alpha=0.7)
plt.legend()

# %%
# export results
coef.to_csv(f"{projectpath}/data/alpha/gan_{select}_macro_{with_macro}.csv")
