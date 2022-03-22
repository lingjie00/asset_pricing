# %%
"""ETL for UK stocks

Please run this script after extracting the following data
1. Fundamental data
2. Price data
3. Fama French factor data

"""
# library
import pandas as pd
from common import (data_path, date_col, max_date, min_date, missing_code,
                    price_col, processed_path, reindex, remove_missingChar,
                    remove_symbols, symbol_col, train_index, valid_index)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# %%
"""Load data"""
# load return data
price = pd.read_csv(processed_path["returns"])
price[date_col] = pd.to_datetime(price[date_col])
price = price.set_index([symbol_col, date_col])
price.info()
price.head()

# %%
# load factor data
factors = pd.read_csv(processed_path["factor"])
factors[date_col] = pd.to_datetime(factors[date_col])
factors = factors.set_index(date_col)
factors.info()
factors.head()

# %%
# load fundamental data
fundamental = pd.read_csv(processed_path["fundamental"])
fundamental[date_col] = pd.to_datetime(fundamental[date_col])
fundamental = fundamental.set_index([symbol_col, date_col])
fundamental.info()
fundamental.head()


# %%
"""Join the data"""
# Fama factor data
fama_data = price.combine_first(factors)\
    .dropna()\
    .astype("float")
fama_data = remove_missingChar(fama_data)
fama_data = remove_symbols(fama_data)
fama_data = reindex(fama_data)
fama_data = fama_data[[price_col]]\
    .merge(fama_data.drop(columns=[price_col]),
           left_index=True, right_index=True)
# filter data to the range specified by min_date and max_date
filter_index = fama_data.index.get_level_values(date_col)
filter_index = (filter_index >= min_date) & (filter_index <= max_date)
fama_data = fama_data[filter_index]

print(
    fama_data.index.get_level_values(date_col).min(),
    fama_data.index.get_level_values(date_col).max(),
    fama_data.index.get_level_values(symbol_col).unique().shape[0]
)
fama_data.info()
fama_data.loc["ABC"]

# %%
# Firm characteristic data
firm_char = fama_data.combine_first(fundamental)\
    .dropna()\
    .astype("float")
firm_char = remove_missingChar(firm_char)
firm_char = remove_symbols(firm_char)
firm_char = reindex(firm_char)
firm_char = firm_char[[price_col]]\
    .merge(firm_char.drop(columns=[price_col]),
           left_index=True, right_index=True)
# filter data to the range specified by min_date and max_date
filter_index = firm_char.index.get_level_values(date_col)
filter_index = (filter_index >= min_date) & (filter_index <= max_date)
firm_char = firm_char[filter_index]

print(
    firm_char.index.get_level_values(date_col).min(),
    firm_char.index.get_level_values(date_col).max(),
    firm_char.index.get_level_values(symbol_col).unique().shape[0]
)
firm_char.info()
firm_char.loc["ABC"]

# %%
"""Date check"""
# percentage of non missing observations per month
check = pd.concat([
    fama_data.replace(missing_code, float("NaN"))
    .notnull()
    .reset_index()
    .groupby(date_col)
    .mean()[[price_col]]
    .rename(columns={price_col: "fama"})
    .T,
    fama_data.replace(missing_code, float("NaN"))
    .notnull()
    .reset_index()
    .groupby(date_col)
    .sum()[[price_col]]
    .rename(columns={price_col: "fama_count"})
    .T,
    firm_char.replace(missing_code, float("NaN"))
    .notnull()
    .reset_index()
    .groupby(date_col)
    .mean()[[price_col]]
    .rename(columns={price_col: "firm_char"})
    .T,
    firm_char.replace(missing_code, float("NaN"))
    .notnull()
    .reset_index()
    .groupby(date_col)
    .sum()[[price_col]]
    .rename(columns={price_col: "firm_char_count"})
    .T
])
check

# %%
# check train valid splitting
view = fama_data.loc["3IN"]
# training data
view.iloc[:train_index]

# %%
# validation
view.iloc[train_index:valid_index]

# %%
# test data
view.iloc[valid_index:]

# %%
# check symbol counts
price.index.get_level_values(symbol_col).unique()

# %%
fama_data.index.get_level_values(symbol_col).unique()

# %%
firm_char.index.get_level_values(symbol_col).unique()

# %%
"""Export data"""
firm_char.to_csv(data_path["fundamental"])
fama_data.to_csv(data_path["factor"])
