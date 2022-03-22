# %%
"""Process fundamental data"""
# library
import numpy as np
import pandas as pd
from common import date_col, processed_path, raw_path, symbol_col

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# %%
# import data
raw = pd.read_csv(raw_path["fundamental"])
raw["acceptedDate"] = pd.to_datetime(raw["acceptedDate"])
raw[symbol_col] = raw[symbol_col].str.replace(".L", "", regex=False)

raw[raw[symbol_col] == "ABC"]

# %%
"""Preprocessing

1. Drop unused columns
2. Standardised column naming
3. Replace values coded as 0
"""
drop_labels = ["date", "fillingDate", "period", "link", "finalLink"]
rename_cols = {"acceptedDate": date_col}
replace_int = [0, float("inf")]

data = raw.drop(labels=drop_labels, axis=1)\
    .rename(mapper=rename_cols, axis=1)\
    .set_index([symbol_col, date_col])\
    .replace(replace_int, float("NaN"))\
    .sort_values(by=[symbol_col, date_col])

data.loc["ABC"]

# %%
# we note that fundamental data does not have balanced panel
# we drop those cols with 50th quantile > threshold %
threshold_percent = 0.1
stat = data.isna()\
    .groupby(symbol_col).mean()\
    .describe().T.sort_values(by="50%")
drop_col = stat[stat["50%"] > 0.1].index.values
print(drop_col)
stat

# %%
"""Transform data
1. Compute log differences
2. Resample quarterly data to monthly data
"""
# compute log and replace missing values
data = data\
    .drop(columns=drop_col)\
    .groupby(date_col)\
    .apply(lambda x: np.log(x) - np.log(x.shift(1)))\
    .dropna()

data.loc["ABC"]

# %%
# resample
data = data\
    .reset_index()\
    .set_index(date_col)\
    .groupby(symbol_col)\
    .resample("1M")\
    .ffill()\
    .drop(labels=[symbol_col], axis=1)\
    .astype("float")

data.loc["ABC"]

# %%
# export
data.to_csv(processed_path["fundamental"])
