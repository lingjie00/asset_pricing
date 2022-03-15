# %%
"""Library"""
import pandas as pd

factor_filepath = "~/db/asset_pricing/uk_monthlyfactors/monthlyfactors.csv"

# %%
"""Load factor data"""
raw = pd.read_csv(factor_filepath)

rename_cols = {"month": "date"}
raw["month"] = pd.to_datetime(raw["month"], format="%Ym%m")\
        .dt.to_period("M").dt.to_timestamp("M")
raw = raw.rename(columns=rename_cols)

raw.head()

# %%
raw.to_csv("../data/factors.csv", index=False)
