# %%
"""Library"""
import matplotlib.pyplot as plt
import pandas as pd
<<<<<<< HEAD
import statsmodels.api as sm
from common import (data_path, date_col, missing_code, price_col, projectpath,
                    symbol_col, train_percent, valid_percent)
from sklearn.linear_model import LinearRegression
=======
from common import (alpha_beta, data_path, date_col, missing_code, price_col,
                    projectpath, symbol_col)
>>>>>>> origin/uk_gan

# %%
select = "factor"
with_macro = False

"""Load data"""
data = pd.read_csv(data_path[select])
data[date_col] = pd.to_datetime(data[date_col])
data = data.set_index([symbol_col, date_col])\
    .replace(missing_code, float("NaN"))\
    .astype("float")

data.dropna().head()

# %%
sdf = pd.read_csv(
    f"{projectpath}/data/stable_results/sdf_uk_{select}_macro_{with_macro}.csv"
)
sdf["date"] = pd.to_datetime(sdf["date"])
sdf = sdf.set_index("date")

sdf.dropna().head()

# %%
data = data.combine_first(sdf)
data = data[[price_col, "sdf"]]

data.dropna().head()

# %%
"""Alpha beta"""
coef = alpha_beta(data)

coef

# %%
coef["train_intercept"].abs().describe()

# %%
coef["valid_intercept"].abs().describe()

# %%
coef["test_intercept"].abs().describe()

# %%
coef.drop(columns=["sdf"]).boxplot()

# %%
coef["train_intercept"].plot.kde(alpha=0.7)
coef["valid_intercept"].plot.kde(alpha=0.7)
coef["test_intercept"].plot.kde(alpha=0.7)
plt.legend()

# %%
# export results
coef.to_csv(f"{projectpath}/data/alpha/gan_{select}_macro_{with_macro}.csv")
