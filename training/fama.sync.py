# %%
"""Library"""
import matplotlib.pyplot as plt
import pandas as pd
from common import (alpha_beta, data_path, date_col, fama_sharpe, missing_code,
                    price_col, processed_path, projectpath, symbol_col,
                    train_index, valid_index)

# load data
select = "factor"
data = pd.read_csv(data_path[select])
data[date_col] = pd.to_datetime(data[date_col])
data = data.set_index([symbol_col, date_col])\
    .replace(missing_code, float("NaN"))\
    .astype("float")

# keep only fama french factors
factor_cols = ["hml", "rmrf", "smb", "umd"]
data = data[[price_col] + factor_cols]

data.dropna().head()

# %%
"""Alpha Beta"""
coef = alpha_beta(data)

coef

# %%
coef["train_intercept"].abs().describe()

# %%
coef["valid_intercept"].abs().describe()

# %%
coef["test_intercept"].abs().describe()

# %%
coef.drop(columns=factor_cols).boxplot()

# %%
coef["train_intercept"].plot.kde(alpha=0.7)
coef["valid_intercept"].plot.kde(alpha=0.7)
coef["test_intercept"].plot.kde(alpha=0.7)
plt.legend()

# %%
# import facotr data
factor_data = pd.read_csv(processed_path["factor"])
factor_data[date_col] = pd.to_datetime(factor_data[date_col])
factor_data = factor_data.set_index(date_col)\
        .reindex(data.index.get_level_values(date_col).unique())\
        .sort_values(date_col)
factor_data.head()

# %%
# sharpe ratio
train_data = factor_data.iloc[:train_index, :]
valid_data = factor_data.iloc[train_index:valid_index, :]
test_data = factor_data.iloc[valid_index:, :]
train_sharpe = fama_sharpe(train_data)
valid_sharpe = fama_sharpe(valid_data)
test_sharpe = fama_sharpe(test_data)

train_sharpe, valid_sharpe, test_sharpe

# %%
# export results
coef.to_csv(f"{projectpath}/data/alpha/fama_{select}.csv")
