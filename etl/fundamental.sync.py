# %%
"""Data transformation for UK LSE data.

This script transforms the following data
1. Fundamental data

It is designed to be not following functional form or
objective orientated form to experiment different data
manipulations in notebooks easily.

All final data will be stored in a dictionary called `final`
"""
# library
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# file path
fundamental_path = "~/db/asset_pricing/Finage_LSE_data/fundamental_quarter.csv"

# date range
min_date = "2009-01-01"
max_date = "2022-01-01"

# %%
"""Fundamental data"""
# path
raw = pd.read_csv(fundamental_path)
raw["acceptedDate"] = pd.to_datetime(raw["acceptedDate"])
raw["symbol"] = raw["symbol"].str.replace(".L", "", regex=False)

raw[raw["symbol"] == "3IN"]

# %%

missing_code = "-99.99"
drop_labels = [
    "date", "fillingDate", "period", "link", "finalLink"
]
rename_cols = {"acceptedDate": "date"}
time_index = pd.date_range(
    start=min_date,
    end=max_date,
    freq="M"
)
symbols = raw["symbol"].unique()
index = pd.MultiIndex.from_product(
    [symbols, time_index], names=["symbol", "date"]
)
replace_int = [0, float("inf")]

# transform quarter data to monthly data
data = raw.drop(labels=drop_labels, axis=1)\
    .rename(mapper=rename_cols, axis=1)\
    .set_index(["symbol", "date"])\
    .sort_values(by=["symbol", "date"])

data.loc["3IN"]

# %%
# transform data
# replace missing data with median
data = data\
    .replace(replace_int, np.nan)\
    .groupby("date")\
    .apply(lambda x: np.log(x) - np.log(x.shift(1)))\
    .groupby("date")\
    .apply(lambda x: x.fillna(x.median()))\

data.loc["3IN"]

# %%
data = data\
    .reset_index()\
    .set_index("date")\
    .groupby("symbol")\
    .resample("1M")\
    .ffill()\
    .drop(labels=["symbol"], axis=1)\
    .reindex(index)\
    .astype("float")

data.loc["3IN"]
