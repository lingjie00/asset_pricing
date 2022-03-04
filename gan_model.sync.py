# %%
# import library
from IPython.display import Image, display
import logging
import datetime
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import model_to_dot

from models.gan import create_gan
from models.loss import PricingLoss, sharpe_loss
from models.training import train_gan

# logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    filename="training.log", level=logging.DEBUG)
# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
# config files
with open("config.json", "r") as file:
    config = json.load(file)
    config_train = config["training"]
    discriminant_epochs = config_train["discriminant_epochs"]
    generative_epochs = config_train["generative_epochs"]
    gan_epochs = config_train["gan_epochs"]
    min_epochs = config_train["min_epochs"]
    patience = config_train["patience"]
    load_weights = config_train["load_weights"]
    save_weights = config_train["save_weights"]
    mask_key = config["hyperparameters"]["mask_key"]

# set seed for TensorFlow
tf.random.set_seed(20220102)

############################
# Test if GPU is available #
############################
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    logging.info(gpus)
else:
    raise Exception("No GPU")


# %%
# view pydot
def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)


# %%
# load data
path = "../datasets"
configpath = "config.json"

# training data
macro_train_zip = np.load(f"{path}/macro/macro_train.npz")
firm_train_zip = np.load(f"{path}/char/Char_train.npz")
train_macro = macro_train_zip["data"]
train_firm = firm_train_zip["data"]
train_return = train_firm[:, :, 0]
train_firm = train_firm[:, :, 1:]

train_data = {
    "returns": train_return,
    "macro": train_macro,
    "firm": train_firm
}

# validation data
macro_valid_zip = np.load(f"{path}/macro/macro_valid.npz")
firm_valid_zip = np.load(f"{path}/char/Char_valid.npz")
valid_macro = macro_valid_zip["data"]
valid_firm = firm_valid_zip["data"]
valid_return = valid_firm[:, :, 0]
valid_firm = valid_firm[:, :, 1:]

valid_data = {
    "returns": valid_return,
    "macro": valid_macro,
    "firm": valid_firm
}

# test data
macro_test_zip = np.load(f"{path}/macro/macro_test.npz")
firm_test_zip = np.load(f"{path}/char/Char_test.npz")
test_macro = macro_test_zip["data"]
test_firm = firm_test_zip["data"]
test_return = test_firm[:, :, 0]
test_firm = test_firm[:, :, 1:]

test_data = {
    "returns": test_return,
    "macro": test_macro,
    "firm": test_firm
}

logging.info(f"macro train shape: {train_macro.shape}")
logging.info(f"firm train shape: {train_firm.shape}")
logging.info(f"macro valid shape: {valid_macro.shape}")
logging.info(f"firm valid shape: {valid_firm.shape}")
logging.info(f"macro test shape: {test_macro.shape}")
logging.info(f"firm test shape: {test_firm.shape}")

# %%
# remove zip files
del macro_train_zip, firm_train_zip
del macro_valid_zip, firm_valid_zip
del macro_test_zip, firm_test_zip

# %%
train_networks = create_gan(
    configpath=configpath,
    data=train_data
)

# %%
view_pydot(model_to_dot(train_networks["discriminant_network"]))
view_pydot(model_to_dot(train_networks["generative_network"]))


# %%
# validation network
valid_networks = create_gan(
    configpath=configpath,
    data=valid_data
)

# %%
# test network
test_networks = create_gan(
    configpath=configpath,
    data=test_data
)

# %%
weight_model = keras.Model(
    inputs=train_networks["discriminant_network"].inputs,
    outputs=train_networks["discriminant_network"].get_layer("sdf_w").output
)

weight_model([train_macro, train_firm])

# %%
sdf = train_networks["discriminant_network"]([train_macro, train_firm])
moment = train_networks["generative_network"]([train_macro, train_firm])

loss = PricingLoss(
    sdf=sdf,
    moment=moment,
    returns=train_return,
    mask_key=-99.99
)

shape_loss = sharpe_loss(sdf)

logging.debug(f"Initial Pricing loss: {loss}")
logging.debug(f"Initial SHARPE loss: {shape_loss}")

# %%
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
train_gan(
    train_data=train_data,
    train_networks=train_networks,
    optimizer=optimizer,
    valid_data=valid_data,
    valid_networks=valid_networks,
    train_summary_writer=train_summary_writer,
    valid_summary_writer=valid_summary_writer,
    discriminant_epochs=discriminant_epochs,
    generative_epochs=generative_epochs,
    gan_epochs=gan_epochs,
    min_epochs=min_epochs,
    patience=patience,
    mask_key=mask_key,
    load_weights=load_weights,
    save_weights=save_weights
)

# %%
# Loss for train
sdf_train = train_networks["discriminant_network"]([train_macro, train_firm])

sharpe_loss_train = sharpe_loss(sdf_train)

logging.debug(f"GAN Trained train SHARPE loss: {sharpe_loss_train}")

# %%
# Loss for valid
sdf_valid = valid_networks["discriminant_network"]([valid_macro, valid_firm])

sharpe_loss_valid = sharpe_loss(sdf_valid)

logging.debug(f"GAN Trained valid SHARPE loss: {sharpe_loss_valid}")

# %%
# Loss for test
test_networks["discriminant_network"].set_weights(
    train_networks["discriminant_network"].get_weights()
)

sdf_test = test_networks["discriminant_network"]([test_macro, test_firm])

sharpe_loss_test = sharpe_loss(sdf_test)

logging.debug(f"GAN Trained test SHARPE loss: {sharpe_loss_test}")
