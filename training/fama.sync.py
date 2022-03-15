# %%
"""Library"""
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns

missing_index = -99.99
col_name = "adj_close_excess"

# %%
"""Load price data"""
price = pd.read_csv("../data/yprice_1mo.csv")\
    .set_index(["symbol", "date"])
price.head()

# %%
"""Load factors"""
factors = pd.read_csv("../data/factors.csv")\
    .set_index("date")
factors.head()

# %%
"""Join the two data"""
data = price.combine_first(factors)\
    .replace(-99.99, None)\
    .dropna(axis=0)\
    .astype("float")

data.head()

# %%
"""One asset"""
mod = sm.OLS(data.loc["3IN", col_name],
             data.loc["3IN"].drop(columns=[col_name]).assign(const=1),
             missing=missing_index,
             hasconst=True
             )

res = mod.fit()

res.summary()

# %%
"""Sklearn"""
coef = {}

for symbol in data.index.get_level_values("symbol").unique():
    train = data.loc[symbol]
    lr = LinearRegression()\
        .fit(
        train.drop(columns=[col_name]),
        train.loc[:, col_name]
    )

    coef[symbol] = lr.intercept_

# %%
pd.DataFrame(coef)
