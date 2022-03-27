# %% [md]
"""Runs GAN model for UK data"""
# %%
# Library
import logging

import pandas as pd
import tensorflow as tf
from common import configpath, load_data, missing_code, projectpath, split_data

from ap import create_gan, sharpe_loss, train_gan

# logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    filename="logs/training_UK.log", level=logging.DEBUG)

# set seed for TensorFlow
tf.random.set_seed(20220102)

# %%
"""Load data"""
select = "factor"
with_macro = False

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
valid_data = processed_data["valid"]
test_data = processed_data["test"]
if with_macro:
    train_data_list = [train_data["macro"], train_data["firm"]]
    valid_data_list = [valid_data["macro"], valid_data["firm"]]
    test_data_list = [test_data["macro"], test_data["firm"]]
else:
    train_data_list = [train_data["firm"]]
    valid_data_list = [valid_data["firm"]]
    test_data_list = [test_data["firm"]]

# %%
"""Create networks."""
train_networks = create_gan(configpath=configpath, data=train_data)
valid_networks = create_gan(configpath=configpath, data=valid_data)
test_networks = create_gan(configpath=configpath, data=test_data)

# %%
"""Train GAN models."""
train_gan(
    configpath=configpath,
    train_data=train_data,
    train_networks=train_networks,
    valid_data=valid_data,
    valid_networks=valid_networks
)

# %%
"""Compute final pricing loss and Sharpe loss for all data"""
logging.info(f"Run {select} and macro {with_macro}")
# Loss for train
sdf_train = train_networks["discriminant_network"](train_data_list)
sharpe_loss_train = sharpe_loss(sdf_train)
logging.info(f"GAN Trained train SHARPE loss: {sharpe_loss_train}")

# Loss for valid
sdf_valid = valid_networks["discriminant_network"](valid_data_list)
sharpe_loss_valid = sharpe_loss(sdf_valid)
logging.info(f"GAN Trained valid SHARPE loss: {sharpe_loss_valid}")

# Loss for test
test_networks["discriminant_network"].set_weights(
    train_networks["discriminant_network"].get_weights()
)
sdf_test = test_networks["discriminant_network"](test_data_list)
sharpe_loss_test = sharpe_loss(sdf_test)
logging.info(f"GAN Trained test SHARPE loss: {sharpe_loss_test}")

sharpe_results = {
    "train": sharpe_loss_train.numpy(),
    "valid": sharpe_loss_valid.numpy(),
    "test": sharpe_loss_test.numpy()
}
sharpe_results = pd.DataFrame(
    sharpe_results, index=[f"{select} with macro {with_macro}"]
)

sharpe_results.to_csv(
    f"{projectpath}/data/results/sharpe_uk_{select}_macro_{with_macro}.csv"
)

sharpe_results

# %%
# export sdf
sdf_train_df = pd.DataFrame(sdf_train, columns=["sdf"])
sdf_valid_df = pd.DataFrame(sdf_valid, columns=["sdf"])
sdf_test_df = pd.DataFrame(sdf_test, columns=["sdf"])
sdf_df = pd.concat([sdf_train_df, sdf_valid_df, sdf_test_df], axis=0)
sdf_df.index = firm.index.get_level_values("date").unique()

sdf_df.to_csv(
    f"{projectpath}/data/results/sdf_uk_{select}_macro_{with_macro}.csv"
)

sdf_df.head()
