# %%
"""Transform stock prices and returns"""
# library
from datetime import timedelta

import numpy as np
import pandas as pd
from common import (date_col, max_date, missing_code, price_col,
                    processed_path, raw_path, raw_price_col, reindex,
                    remove_missingChar, risk_col, symbol_col)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# %%
"""Risk free rate"""
# monthly
risk_monthly_raw = pd.read_csv(raw_path["risk_monthly"])
risk_monthly_raw.columns = [date_col, risk_col + "_raw"]
risk_monthly_raw[date_col] = pd.to_datetime(risk_monthly_raw[date_col])
# daily
risk_daily_raw = pd.read_csv(raw_path["risk_daily"])
risk_daily_raw.columns = [date_col, risk_col + "_raw"]
risk_daily_raw[date_col] = pd.to_datetime(risk_daily_raw[date_col])


# %%
risk_monthly_raw.head()

# %%
risk_daily_raw.head()

# %%

# compute risk free returns
risk_monthly = risk_monthly_raw.set_index(date_col)\
    .sort_values(by=date_col)\
    .astype("float")
risk_monthly[risk_col] = np.log(risk_monthly[risk_col + "_raw"])\
    - np.log(risk_monthly[risk_col + "_raw"].shift(1))

risk_monthly.plot()

risk_monthly.head()

# %%

risk_daily = risk_daily_raw.set_index(date_col)\
    .sort_values(by=date_col)\
    .astype("float")
risk_daily[risk_col] = np.log(risk_daily[risk_col + "_raw"])\
    - np.log(risk_daily[risk_col + "_raw"].shift(1))

risk_daily.plot()

risk_daily.head()


# %%
"""Price data"""
# preprocessing monthly price data
headers = ["date", "open", "high",
           "low", "close", "adj_close", "volume", "symbol"]
price_monthly_raw = pd.read_csv(raw_path["price_monthly"], header=None)\
    .dropna(axis=0)
price_monthly_raw.columns = headers
price_monthly_raw[date_col] = pd.to_datetime(
    price_monthly_raw[date_col]) - timedelta(days=1)
price_monthly_raw[symbol_col] = price_monthly_raw.loc[:, symbol_col]\
    .str.replace(".L", "", regex=False)

# %%
# preprocess daily price data
price_daily_raw = pd.read_csv(raw_path["price_daily"], header=None)
price_daily_raw.columns = headers
price_daily_raw[date_col] = pd.to_datetime(price_daily_raw[date_col])
price_daily_raw[symbol_col] = price_daily_raw.loc[:, symbol_col]\
    .str.replace(".L", "", regex=False)

# %%
price_monthly_raw[price_monthly_raw.loc[:, symbol_col] == "3IN"]

# %%
price_daily_raw[price_daily_raw.loc[:, symbol_col] == "3IN"]

# %%
"""Process data

1. Filter March 2022 data (incomplete data)
2. Compute returns with log differences
3. Combine with risk-free data
4. Compute excess return
5. Keep only excess returns
"""
# process monthly data
exclude_date = "2022-03-15"
monthly_price = price_monthly_raw.loc[
    price_monthly_raw[date_col] != exclude_date]\
    .set_index([symbol_col, date_col])\
    .sort_values(by=[symbol_col, date_col])\
    .groupby(symbol_col)\
    .apply(lambda x: np.log(x) - np.log(x.shift(1)))\
    .combine_first(risk_monthly)

monthly_price.loc["3IN"]

# %%
# compute excess returns and keep only excess return
monthly_price[price_col] = monthly_price[raw_price_col]\
    - monthly_price[risk_col]
monthly_price = monthly_price[[price_col]].astype("float")

monthly_price.plot()
monthly_price.loc["3IN"]

# %%
# process daily data
daily_price = price_daily_raw\
        .set_index([symbol_col, date_col])\
        .sort_values(by=[symbol_col, date_col])\
        .groupby(symbol_col)\
        .apply(lambda x: np.log(x) - np.log(x.shift(1)))\
        .combine_first(risk_daily)

daily_price.loc["3IN"]

# %%
daily_price[price_col] = daily_price[raw_price_col]\
        - daily_price[risk_col]
daily_price = daily_price[[price_col]].astype("float")

daily_price.plot()
daily_price.loc["3IN"]

# %%
"""Add in firm-specific characteristics data"""


def get_r2_1(df):
    """Return lagged one-month return"""
    price = df.reset_index()\
        .set_index(date_col)\
        .loc[:, [price_col]]\
        .shift(1)
    price.columns = ["r2_1"]
    return price


def get_r12_7(df):
    """Return cum returns from 12 months ago to 7 months ago"""
    price = df.reset_index()\
        .set_index(date_col)\
        .loc[:, [price_col]]\
        .replace(missing_code, float("NaN"))\
        .cumsum()
    result = price.shift(7) - price.shift(13)
    result.columns = ["r12_7"]
    return result


def drop_year(df):
    """Drop df if the entry is not full year."""
    start = df.index.get_level_values("date").min().year
    start = pd.to_datetime(f"{start}-01-01") + pd.tseries.offsets.MonthEnd()
    end = min(df.index.get_level_values("date").max(),
              pd.to_datetime(max_date))
    index = pd.date_range(start=start, end=end, freq="M", name=date_col)
    df_new = df.reindex(index)
    first_na = df_new[index == start].isna().sum().values[0]
    if first_na >= 1:
        # first entry of the year is missing
        start += pd.DateOffset(years=1)
        df_new = df_new[index >= start]
    return df_new


def get_rel2high(df):
    """Get Closeness to past year high.

    (not used to maximise data available)
    Ratio of stock price at previous month and highest daily
    price in past year."""
    df = df.reset_index()
    df[date_col] = df[date_col] + pd.DateOffset(years=1)
    groupby = df.set_index(date_col)\
        .groupby(symbol_col)
    month_price = groupby.apply(lambda x: x.resample("M").last())\
        .drop(columns=[symbol_col])
    year_price = groupby.apply(lambda x: x.resample("Y").max())\
        .drop(columns=[symbol_col])\
        .reindex(month_price.index, method="bfill")
    df = month_price / year_price
    df.columns = ["rel2high"]

    # need to drop first year data if we do not have full
    # year for the variance
    df = df.reset_index()\
        .set_index(date_col)\
        .groupby(symbol_col)\
        .apply(drop_year)\
        .drop(columns=[symbol_col])

    return df


monthly_groupby = monthly_price.groupby(symbol_col)

# add in past return firm characteristics data
monthly_final = monthly_price.copy()\
    .combine_first(monthly_groupby.apply(get_r2_1))\
    .combine_first(monthly_groupby.apply(get_r12_7))\
    .dropna()

monthly_final.loc["3IN"]

# %%
# export
monthly_final.to_csv(processed_path["returns"])
