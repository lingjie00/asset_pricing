# %%
"""Data transformation for UK LSE data with Yahoo Finance.

This script transforms the following data
- Price data
- Risk-free bond data (using Monthly average SONIA rate)

It is designed to be not following functional form or
objective orientated form to experiment different data
manipulations in notebooks easily.

All final data will be stored in a dictionary called `final`
"""
# library
from datetime import timedelta

import numpy as np
import pandas as pd
from common_etl import missing_code

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# storing transformed data
# file path
price_path = "~/db/asset_pricing/yfinance/1mo.csv"
risk_path = "~/db/asset_pricing/IUMASOIA.csv"

# date range
min_date = "2000-01-01"
max_date = "2022-03-01"


# %%
"""Risk free rate"""
sonia = pd.read_csv(risk_path)

sonia.columns = ["date", "SONIA"]
sonia["date"] = pd.to_datetime(sonia["date"])

sonia.head()

# %%

# compute risk free returns
sonia = sonia.set_index("date")\
    .sort_values(by="date")\
    .astype("float")
sonia["SONIA_return"] = np.log(sonia["SONIA"])\
    - np.log(sonia["SONIA"].shift(1))

# large spikes correspond to financial crisis
sonia.plot()

sonia.head()

# %%
"""Price data"""
headers = ["date", "open", "high",
           "low", "close", "adj_close", "volume", "symbol"]
raw = pd.read_csv(price_path, header=None)\
        .dropna(axis=0)
raw.columns = headers
raw["date"] = pd.to_datetime(raw["date"]) - timedelta(days=1)
raw["symbol"] = raw.loc[:, "symbol"].str.replace(".L", "", regex=False)

raw[raw.loc[:, "symbol"] == "3IN"]

# %%
symbols = raw["symbol"].unique()
time_index = pd.date_range(
    start=min_date,
    end=max_date,
    freq="M"
)
index = pd.MultiIndex.from_product(
    [symbols, time_index], names=["symbol", "date"]
)
data = raw.loc[raw["date"] != "2022-03-15"]\
    .set_index(["symbol", "date"])\
    .sort_values(by=["symbol", "date"])\
    .groupby("symbol")\
    .apply(lambda x: np.log(x) - np.log(x.shift(1)))\
    .combine_first(sonia)\
    .reindex(index)

data.loc["3IN"]

# %%
# compute excess returns
for col in ["close", "high", "low", "open", "adj_close"]:
    data.loc[:, col + "_excess"] = data.loc[:, col] - \
        data.loc[:, "SONIA_return"]

data.loc["3IN"]

# %%
# export data
final = data.loc[:, "adj_close_excess"]\
        .fillna(missing_code)\
        .astype("float")
final.to_csv("../data/yprice_1mo.csv")

final.replace(missing_code, None).plot()
