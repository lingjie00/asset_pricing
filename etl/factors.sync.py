# %%
"""Process Fama French factor data"""
# library
import pandas as pd
from common import raw_path, processed_path, date_col

# %%
# import data
raw = pd.read_csv(raw_path["factor"])
# process data
rename_cols = {"month": date_col}
raw["month"] = pd.to_datetime(raw["month"], format="%Ym%m")\
    .dt.to_period("M").dt.to_timestamp("M")
raw = raw.rename(columns=rename_cols)

raw.head()

# %%
# keep only the factors
factor_names = [date_col, "smb", "hml", "umd", "rmrf"]
factors = raw[factor_names]

factors.info()
factors.head()

# %%
# export data
factors.to_csv(processed_path["factor"], index=False)
