# %%
"""# Library and configurations"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from IPython.display import Image, display
from tensorflow.keras.utils import model_to_dot

from models.gan import create_gan
from models.loss import PricingLoss, sharpe_loss
from models.training import train_gan

# logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    filename="logs/training.log", level=logging.DEBUG)

# set seed for TensorFlow
tf.random.set_seed(20220102)


def view_pydot(pdot):
    """View model in notebook without exporting"""
    plt = Image(pdot.create_png())
    display(plt)


# %%
"""Load Chen's data"""
# load data
path = "../datasets"
configpath = "config.json"

# training data
macro_train_zip = np.load(f"{path}/macro/macro_train.npz")
firm_train_zip = np.load(f"{path}/char/Char_train.npz")
train_macro = macro_train_zip["data"]
train_firm = firm_train_zip["data"]
train_data = {
    "returns": train_firm[:, :, 0],
    "macro": train_macro,
    "firm": train_firm[:, :, 1:]
}

# validation data
macro_valid_zip = np.load(f"{path}/macro/macro_valid.npz")
firm_valid_zip = np.load(f"{path}/char/Char_valid.npz")
valid_macro = macro_valid_zip["data"]
valid_firm = firm_valid_zip["data"]
valid_data = {
    "returns": valid_firm[:, :, 0],
    "macro": valid_macro,
    "firm": valid_firm[:, :, 1:]
}

# test data
macro_test_zip = np.load(f"{path}/macro/macro_test.npz")
firm_test_zip = np.load(f"{path}/char/Char_test.npz")
test_macro = macro_test_zip["data"]
test_firm = firm_test_zip["data"]
test_data = {
    "returns": test_firm[:, :, 0],
    "macro": test_macro,
    "firm": test_firm[:, :, 1:]
}

# remove returns from the firm data
train_firm = train_firm[:, :, 1:]
valid_firm = valid_firm[:, :, 1:]
test_firm = test_firm[:, :, 1:]

logging.info(f"macro train shape: {train_macro.shape}")
logging.info(f"firm train shape: {train_firm.shape}")
logging.info(f"macro valid shape: {valid_macro.shape}")
logging.info(f"firm valid shape: {valid_firm.shape}")
logging.info(f"macro test shape: {test_macro.shape}")
logging.info(f"firm test shape: {test_firm.shape}")

# remove zip files
del macro_train_zip, firm_train_zip
del macro_valid_zip, firm_valid_zip
del macro_test_zip, firm_test_zip

# %%
"""Create networks"""
train_networks = create_gan(
    configpath=configpath,
    data=train_data
)
valid_networks = create_gan(
    configpath=configpath,
    data=valid_data
)
test_networks = create_gan(
    configpath=configpath,
    data=test_data
)

view_pydot(model_to_dot(train_networks["discriminant_network"]))
view_pydot(model_to_dot(train_networks["generative_network"]))


# %%
"""Extract some features from the network for viewing"""
# compute the loss before training
sdf = train_networks["discriminant_network"]([train_macro, train_firm])
moment = train_networks["generative_network"]([train_macro, train_firm])
loss = PricingLoss(
    sdf=sdf,
    moment=moment,
    returns=train_data["returns"],
    mask_key=-99.99
)
shape_loss = sharpe_loss(sdf)
logging.debug(f"Initial Pricing loss: {loss}")
logging.debug(f"Initial SHARPE loss: {shape_loss}")

# view the SDF weights
weight_model = keras.Model(
    inputs=train_networks["discriminant_network"].inputs,
    outputs=train_networks["discriminant_network"].get_layer("sdf_w").output
)
logging.info(f"Initial weights: {weight_model([train_macro, train_firm])}")

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
# Loss for train
sdf_train = train_networks["discriminant_network"]([train_macro, train_firm])
sharpe_loss_train = sharpe_loss(sdf_train)
logging.info(f"GAN Trained train SHARPE loss: {sharpe_loss_train}")

# Loss for valid
sdf_valid = valid_networks["discriminant_network"]([valid_macro, valid_firm])
sharpe_loss_valid = sharpe_loss(sdf_valid)
logging.info(f"GAN Trained valid SHARPE loss: {sharpe_loss_valid}")

# Loss for test
test_networks["discriminant_network"].set_weights(
    train_networks["discriminant_network"].get_weights()
)
sdf_test = test_networks["discriminant_network"]([test_macro, test_firm])
sharpe_loss_test = sharpe_loss(sdf_test)
logging.info(f"GAN Trained test SHARPE loss: {sharpe_loss_test}")
