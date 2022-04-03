# %%
"""Explains the GAN model"""
# library
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from common import (configpath, load_data, missing_code, price_col,
                    projectpath, split_data)

from ap import create_gan

# logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    filename="logs/training_explain.log", level=logging.DEBUG)

# %%
"""Load data"""
select = "factor"
with_macro = True

logging.info(f"Run {select} and macro {with_macro}")

data = load_data(select, load_macro=with_macro)
firm = data["firm"]
if with_macro:
    macro = data["macro"]
else:
    macro = None

firm.info()
firm.replace(missing_code, float("NaN")).dropna().head()

# %%
if with_macro:
    print(macro.info())
    print(macro.head())

# %%
"""Split data"""
processed_data = split_data(firm, macro)
train_data = processed_data["train"]
if with_macro:
    train_data_list = [train_data["macro"], train_data["firm"]]
else:
    train_data_list = [train_data["firm"]]

# %%
"""Create networks."""
train_networks = create_gan(configpath=configpath, data=train_data)

# %%
"""Load network weights"""
network = train_networks["discriminant_network"]
network.load_weights(
    f"logs/UK_model/{select}_macro_{with_macro}/discriminant.h5")
weight_network = tf.keras.Model(
    inputs=network.inputs,
    outputs=network.get_layer("sdf_w").output
)
old_weight = weight_network(train_data_list)

# %%
"""Compute sensitivity for firm characteristics"""
eps = 1e-6
sensi = {}
columns = list(firm.columns)
columns.remove(price_col)

for index, col in enumerate(columns):
    new_firm = np.copy(train_data["firm"])
    new_firm[:, :, index] = train_data["firm"][:, :, index] - eps
    if with_macro:
        new_data = [train_data["macro"], new_firm]
    else:
        new_data = [new_firm]
    new_weight = weight_network(new_data)
    gradient = (new_weight - old_weight) / eps
    vi = np.sum(np.abs(gradient))
    sensi[col] = [vi]

sensi = pd.DataFrame(sensi).T
sensi.columns = ["VI"]
sensi = sensi.assign(source="firm")

"""Compute sensitivity for macroeconomic data"""
if with_macro:
    eps = 1e-6
    macro_sensi = {}
    columns = list(macro.columns)

    for index, col in enumerate(columns):
        new_macro = np.copy(train_data["macro"])
        new_macro[:, index, :] = train_data["macro"][:, index, :] - eps
        new_data = [new_macro, train_data["firm"]]
        new_weight = weight_network(new_data)
        gradient = (new_weight - old_weight) / eps
        vi = np.sum(np.abs(gradient))
        macro_sensi[col] = [vi]

    macro_sensi = pd.DataFrame(macro_sensi).T
    macro_sensi.columns = ["VI"]
    macro_sensi = macro_sensi.assign(source="macro")
    sensi = pd.concat([sensi, macro_sensi])

# normalise and save sensitivity
sensi["VI"] = sensi["VI"] / sensi["VI"].sum()
sensi = sensi.reset_index()

# regroup macro data
if with_macro:
    macro_threshold = 10
    bottom_index = macro_sensi.sort_values(by="VI", ascending=False)\
        .iloc[macro_threshold:].index.values
    sensi["variable"] = sensi["index"].replace(bottom_index, "rest of macro")
sensi = sensi.groupby("variable")["VI"].sum().sort_values()

sensi.to_csv(
    f"{projectpath}/data/results/sensi_uk_{select}_macro_{with_macro}.csv"
)

# %%
sensi.plot.barh(x="variable", y="VI", xlabel="")
plt.tight_layout()
plt.savefig(
    "../data/results/vi_uk_{select}_macro_{with_macro}.pdf",
    dpi=120, format="pdf")

# %%
"""SDF weight structure"""
weight_structure = {}
col_index = {
    "hml": 0,
    "rmrf": 3,
    "smb": 4,
    "umd": 5
}

new_firm = np.copy(train_data["firm"])
for fix_col, fix_index in col_index.items():
    weight_structure[fix_col] = {}
    for col, index in col_index.items():
        results = {}
        for i in range(new_firm.shape[2]):
            # reset values to median
            new_firm[:, :, i] = np.quantile(
                new_firm[train_data["returns"] != -99.99, i], 0.5)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            name = f"{col} = {q}"
            results[name] = {}
            # change the col to specific quantile result
            new_firm[:, :, index] = q
            for value in np.linspace(-1, 1, num=30):
                # change the fixed col to the value
                new_firm[:, :, fix_index] = value
                new_data = [new_firm]
                if with_macro:
                    new_data = [train_data["macro"], new_data]
                weight = weight_network(new_data)[0].numpy()[0]
                results[name][value] = weight

        results = pd.DataFrame(results)
        weight_structure[fix_col][col] = results


# %%
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
for fix_index, fix_col in enumerate(col_index.keys()):
    for index, col in enumerate(col_index.keys()):
        weight_structure[fix_col][col].plot(ax=axes[fix_index][index])
axes[0][0].set_title("hml")
axes[0][1].set_title("rmrf")
axes[0][2].set_title("smb")
axes[0][3].set_title("umd")
axes[0][0].set_ylabel("hml")
axes[1][0].set_ylabel("rmrf")
axes[2][0].set_ylabel("smb")
axes[3][0].set_ylabel("umd")
plt.savefig("../data/results/interaction_plot.pdf", dpi=120, format="pdf")
