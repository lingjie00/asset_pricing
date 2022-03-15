# %%
"""Data transformation for UK LSE data.

This script transforms the following data
- Price data
- Risk-free bond data (using Monthly average SONIA rate)

It is designed to be not following functional form or
objective orientated form to experiment different data
manipulations in notebooks easily.

All final data will be stored in a dictionary called `final`
"""
# library
import pandas as pd
from common_etl import missing_code

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# storing transformed data
# file path
price_path = "~/db/asset_pricing/Finage_LSE_data/price_1month.csv"
risk_path = "~/db/asset_pricing/IUMASOIA.csv"

# date range
min_date = "2007-01-01"
max_date = "2022-01-01"

# %%
"""Risk free rate"""
sonia = pd.read_csv(risk_path)

sonia.columns = ["date", "SONIA"]
sonia["date"] = pd.to_datetime(sonia["date"])

sonia.head()

# %%

# compute risk free returns
rename_cols = {"SONIA": "SONIA_return"}
sonia = sonia.set_index("date")\
    .sort_values(by="date")\
    .astype("float")\
    .pct_change(periods=1)\
    .rename(columns=rename_cols)

# large spikes correspond to financial crisis
sonia.plot()

sonia.head()

# %%
"""Price data"""
raw = pd.read_csv(price_path)
raw["t"] = pd.to_datetime(
    raw["t"].str.replace("T12:00:00", "")
).dt.to_period("M").dt.to_timestamp("M")


raw[raw["symbol"] == "3IN"]

# %%
rename_cols = {
    "t": "date",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume"
}
symbols = raw["symbol"].unique()
time_index = pd.date_range(
    start=min_date,
    end=max_date,
    freq="M"
)
index = pd.MultiIndex.from_product(
    [symbols, time_index], names=["symbol", "date"]
)
# Finage database has some error, duplicated 2020-06-01
data = raw.rename(mapper=rename_cols, axis=1)\
    .drop_duplicates(subset=["symbol", "date"])\
    .set_index(["symbol", "date"])\
    .sort_values(by=["symbol", "date"])\
    .reindex(index)

# compute returns
data = data.groupby("symbol")\
    .pct_change(periods=1)\
    .combine_first(sonia)

data.loc["3IN"]

# %%
# compute excess returns
for col in ["close", "high", "low", "open"]:
    data.loc[:, col + "_excess"] = data.loc[:, col] - data.loc[:, "SONIA_return"]

date_filter = data.index.get_level_values("date") >= "2008-01-01"
data = data.loc[date_filter]

data.loc["3IN"]

# %%
# export data
final = data.loc[:, "close_excess"].fillna(missing_code)
final.to_csv("../data/price.csv")
