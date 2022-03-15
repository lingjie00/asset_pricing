# %%
"""Data transformation for UK LSE data.

This script transforms the following data
1. Fundamental data
2. Price data
3. Risk-free bond data (using Monthly average SONIA rate)
4. UK macroeconomic data

It is designed to be not following functional form or
objective orientated form to experiment different data
manipulations in notebooks easily.

All final data will be stored in a dictionary called `final`
"""
# library
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# storing transformed data
final = {}

# file path
fundamental_path = "~/db/asset_pricing/Finage_LSE_data/fundamental_quarter.csv"
price_path = "~/db/asset_pricing/Finage_LSE_data/price_1month.csv"
macro_path = "~/db/asset_pricing/UKMD_February_2022/balanced_uk_md.csv"
risk_path = "~/db/asset_pricing/IUMASOIA.csv"

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
    .set_index("date")\
    .sort_values(by=["symbol", "date"])\
    .groupby("symbol")\
    .resample("1M")\
    .ffill()\
    .drop(labels=["symbol"], axis=1)\
    .astype("float")

# transform data
data = data\
    .reindex(index)\
    .replace(replace_int, missing_code)\
    .fillna(missing_code)

final["fundamental"] = data

data.loc["3IN"]

# %%
"""Risk free rate"""
raw = pd.read_csv(risk_path)

raw.head()

# %%
sonia = raw.copy()
sonia.columns = ["date", "SONIA"]
sonia["date"] = pd.to_datetime(sonia["date"])
sonia = sonia.set_index("date")\
    .sort_values(by="date")\
    .astype("float")\
    .diff(periods=1)

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
vol = data["volume"].pct_change(periods=1)
data = data.groupby("symbol")\
    .diff(periods=1)\
    .combine_first(sonia)\
    .assign(volume=vol)

# compute excess returns
for col in ["close", "high", "low", "open"]:
    data.loc[:, col + "_excess"] = data.loc[:, col] - data.loc[:, "SONIA"]

final["price"] = data

data.loc["3IN"]

# %%
"""UK macroeconomic data"""
raw = pd.read_csv(macro_path)
raw["Date"] = pd.to_datetime(raw["Date"])

raw.head()

# %%
rename_cols = {"Date": "date"}
data = raw.drop(labels="Unnamed: 0", axis=1)\
    .rename(mapper=rename_cols, axis=1)\
    .set_index("date")\
    .loc[min_date:max_date]

final["macro"] = data

data.head()

# %%
final.keys()
