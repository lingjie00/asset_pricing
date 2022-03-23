"""Storing items shared across all scripts/notebooks."""
import pandas as pd

"""Hyper-parameters"""
# column names
missing_code = -99.99
raw_price_col = "adj_close"
price_col = "excess_returns"
symbol_col = "symbol"
date_col = "date"
risk_col = "SONIA"

# express split in incremental percentage
train_percent = 0.6
valid_percent = 0.8

# date range
min_date = "1997-01-01"
max_date = "2021-12-01"


"""Paths"""
projectpath = "~/projects/thesis/asset_pricing"
dbpath = "~/db/asset_pricing"
# configpath cannot be relative
configpath = "UK_config.json"
# storing the filepath for the final data used for training
data_path = {
    "factor": f"{projectpath}/data/factor_data.csv",
    "fundamental": f"{projectpath}/data/factor_fundamental.csv",
    "macro": f"{dbpath}/UKMD_February_2022/balanced_uk_md.csv"
}
# storing the filepath for original data files
raw_path = {
    "factor": f"{dbpath}/uk_monthlyfactors/monthlyfactors.csv",
    "fundamental": f"{dbpath}/Finage_LSE_data/fundamental_quarter.csv",
    "price_monthly": f"{dbpath}/yfinance/1mo.csv",
    "price_daily": f"{dbpath}/yfinance/1d.csv",
    "risk_monthly": f"{dbpath}/IUMASOIA.csv",
    "risk_daily": f"{dbpath}/SONIA_daily.csv"
}
# storing the filepath for intermediate data files
processed_path = {
    "factor": f"{projectpath}/data/processed/factors.csv",
    "fundamental": f"{projectpath}/data/processed/fundamental.csv",
    "returns": f"{projectpath}/data/processed/returns.csv"
}


"""Functions"""


def remove_symbols(df, percent=0.9):
    """Remove data with > percent missing data"""
    missing_percent = df.groupby(symbol_col)[price_col]\
        .apply(
        lambda x: x.replace(missing_code, float("NaN")).isna().mean()
    )
    nonmissing_symbol = (missing_percent[missing_percent <= percent]).index
    df = df.loc[nonmissing_symbol]
    return df


def reindex(df):
    """Reindex data to ensure balanced index."""

    # reindex
    symbols = df.index.get_level_values(symbol_col).unique()
    start = df.replace(missing_code, float("NaN")).dropna()\
        .index.get_level_values(date_col).min()
    end = df.replace(missing_code, float("NaN")).dropna()\
        .index.get_level_values(date_col).max()
    time_index = pd.date_range(
        start=start,
        end=end,
        freq="M"
    )
    index = pd.MultiIndex.from_product(
        [symbols, time_index], names=[symbol_col, date_col]
    )
    df = df.reindex(index).fillna(missing_code)
    return df


def remove_missingChar(df):
    # replace returns with any missing characteristics as missing
    missing_index = df.replace(missing_code, float("NaN"))\
        .isna().sum(axis=1) > 0
    missing_index = df[missing_index].index
    df.loc[missing_index, price_col] = missing_code
    return df


def load_data(name, load_macro: bool):
    """Load the selected data

    Returns:
        1. Price data
        2. Macro data
    """
    # load price data
    firm = pd.read_csv(data_path[name])
    firm[date_col] = pd.to_datetime(firm[date_col])
    firm = firm.set_index([symbol_col, date_col])
    if not load_macro:
        # return price data only
        return {"firm": firm}
    # load macro data
    macro = pd.read_csv(data_path["macro"], index_col=0)\
        .rename(columns={"Date": date_col})
    macro[date_col] = pd.to_datetime(macro[date_col])\
        + pd.DateOffset(months=1) + pd.tseries.offsets.MonthEnd()
    macro = macro.set_index(date_col)
    macro = macro.loc[firm.index.get_level_values(date_col).unique()]
    return {"firm": firm, "macro": macro}


def split_data(firm, macro=None):
    """Split data"""
    # data features
    num_firms = firm.index.get_level_values(symbol_col).unique().shape[0]
    num_cols = firm.columns.shape[0]
    firm_reshape = firm.values.reshape(-1, num_firms, num_cols)
    # splitting index
    train_index = round(firm_reshape.shape[0] * train_percent)
    valid_index = round(firm_reshape.shape[0] * valid_percent)
    # ensure the first column contains the return
    assert(firm.columns[0] == price_col)
    # split the data
    train_data = {
        "returns": firm_reshape[:train_index, :, 0],
        "firm": firm_reshape[:train_index, :, 1:]
    }
    valid_data = {
        "returns": firm_reshape[train_index:valid_index, :, 0],
        "firm": firm_reshape[train_index:valid_index, :, 1:]
    }
    test_data = {
        "returns": firm_reshape[valid_index:, :, 0],
        "firm": firm_reshape[valid_index:, :, 1:]
    }
    if macro is not None:
        # add in macro data if available
        num_macro = macro.columns.shape[0]
        macro_reshape = macro.values.reshape(-1, num_macro, 1)
        train_data["macro"] = macro_reshape[:train_index, :]
        valid_data["macro"] = macro_reshape[train_index:valid_index, :]
        test_data["macro"] = macro_reshape[valid_index:, :]
    return {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }
