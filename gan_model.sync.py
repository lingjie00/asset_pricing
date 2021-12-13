# %%
# import library
from IPython.display import Image, display
import logging
import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import model_to_dot

import models.gan as gan
importlib.reload(gan)

logging.basicConfig(level=logging.DEBUG)

# set seed for TensorFlow
tf.random.set_seed(20220102)

############################
# Test if GPU is available #
############################
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(gpus)
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

# training data
macro_train_zip = np.load(f"{path}/macro/macro_train.npz")
firm_train_zip = np.load(f"{path}/char/Char_train.npz")
macro_train = macro_train_zip["data"]
firm_train = firm_train_zip["data"]
return_train = firm_train[:, :, 0]
firm_train = firm_train[:, :, 1:]

# validation data
macro_valid_zip = np.load(f"{path}/macro/macro_valid.npz")
firm_valid_zip = np.load(f"{path}/char/Char_valid.npz")
macro_valid = macro_valid_zip["data"]
firm_valid = firm_valid_zip["data"]
return_valid = firm_valid[:, :, 0]
firm_valid = firm_valid[:, :, 1:]

# test data
macro_test_zip = np.load(f"{path}/macro/macro_test.npz")
firm_test_zip = np.load(f"{path}/char/Char_test.npz")
macro_test = macro_test_zip["data"]
firm_test = firm_test_zip["data"]
return_test = firm_test[:, :, 0]
firm_test = firm_test[:, :, 1:]

print(f"macro train shape: {macro_train.shape}")
print(f"firm train shape: {firm_train.shape}")
print(f"macro valid shape: {macro_valid.shape}")
print(f"firm valid shape: {firm_valid.shape}")
print(f"macro test shape: {macro_test.shape}")
print(f"firm test shape: {firm_test.shape}")

# %%
# remove zip files
del macro_train_zip, firm_train_zip
del macro_valid_zip, firm_valid_zip
del macro_test_zip, firm_test_zip

# %%
# Hyper-parameters (train)
_macro_feature = macro_train.shape[1]
_num_firms = firm_train.shape[1]
_num_chars = firm_train.shape[2]
macro_shape = (_macro_feature, 1, )
firm_shape = (_num_firms, _num_chars, )

print(f"macro shape: {macro_shape}")
print(f"firm shape: {firm_shape}")

macro_network = gan.create_macro_network(
    macro_shape=macro_shape,
    num_firms=_num_firms,
    name="discriminant_macro",
    LSTM_units=4
)

discriminant_network = gan.create_discriminant_network(
    firm_shape=firm_shape,
    macro_network=macro_network,
    returns=return_train
)

print(f"""network output shape: {discriminant_network(
    [macro_train, firm_train]).shape}""")
discriminant_network.summary()

generative_macro = gan.create_macro_network(
    macro_shape=macro_shape,
    num_firms=_num_firms,
    name="generative_macro",
    LSTM_units=32
)

generative_network = gan.create_generative_network(
    firm_shape=firm_shape,
    macro_network=generative_macro
)

print(f"""network output shape: {generative_network(
    [macro_train, firm_train]).shape}""")
generative_network.summary()

# %%
view_pydot(model_to_dot(discriminant_network))
view_pydot(model_to_dot(generative_network))


# %%
# Hyper-parameters (valid)
_macro_feature = macro_valid.shape[1]
_num_firms = firm_valid.shape[1]
_num_chars = firm_valid.shape[2]
macro_shape = (_macro_feature, 1, )
firm_shape = (_num_firms, _num_chars, )

print(f"macro shape: {macro_shape}")
print(f"firm shape: {firm_shape}")

macro_network_valid = gan.create_macro_network(
    macro_shape=macro_shape,
    num_firms=_num_firms,
    name="discriminant_macro_valid",
    LSTM_units=4
)

discriminant_network_valid = gan.create_discriminant_network(
    firm_shape=firm_shape,
    macro_network=macro_network_valid,
    returns=return_valid
)

print(f"""network output shape: {discriminant_network_valid(
    [macro_valid, firm_valid]).shape}""")
discriminant_network.summary()

# %%
# Hyper-parameters (test)
_macro_feature = macro_test.shape[1]
_num_firms = firm_test.shape[1]
_num_chars = firm_test.shape[2]
macro_shape = (_macro_feature, 1, )
firm_shape = (_num_firms, _num_chars, )

print(f"macro shape: {macro_shape}")
print(f"firm shape: {firm_shape}")

macro_network_test = gan.create_macro_network(
    macro_shape=macro_shape,
    num_firms=_num_firms,
    name="discriminant_macro_test",
    LSTM_units=4
)

discriminant_network_test = gan.create_discriminant_network(
    firm_shape=firm_shape,
    macro_network=macro_network_test,
    returns=return_test
)

print(f"""network output shape: {discriminant_network_test(
    [macro_test, firm_test]).shape}""")
discriminant_network.summary()

# %%
weight_model = keras.Model(
    inputs=discriminant_network.inputs,
    outputs=discriminant_network.get_layer("sdf_w").output
)

weight_model([macro_train, firm_train])

# %%
sdf = discriminant_network([macro_train, firm_train])
moment = generative_network([macro_train, firm_train])

loss = gan.PricingLoss(
    sdf=sdf,
    moment=moment,
    returns=return_train
)

shape_loss = gan.sharpe_loss(sdf)

logging.debug(f"Initial Pricing loss: {loss}")
logging.debug(f"Initial SHARPE loss: {shape_loss}")

# %%
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
gan.train(
    inputs=[macro_train, firm_train],
    returns=return_train,
    optimizer=optimizer,
    discriminant_network=discriminant_network,
    generative_network=generative_network,
    valid_inputs=[macro_valid, firm_valid],
    valid_returns=return_valid,
    valid_discriminant=discriminant_network_valid,
    discriminant_epochs=100,
    generative_epochs=100,
    gan_epochs=100,
    verbose_interval=10,
    min_epochs=10,
    patience=5000,
    load_weights=[True, True, True],
    save_weights=[False, False, False]
)

# %%
# Loss for train
sdf_train = discriminant_network([macro_train, firm_train])

sharpe_loss_train = gan.sharpe_loss(sdf_train)

logging.debug(f"GAN Trained train SHARPE loss: {sharpe_loss_train}")

# %%
# Loss for valid
sdf_valid = discriminant_network_valid([macro_valid, firm_valid])

sharpe_loss_valid = gan.sharpe_loss(sdf_valid)

logging.debug(f"GAN Trained valid SHARPE loss: {sharpe_loss_valid}")

# %%
# Loss for test
discriminant_network_test.set_weights(
    discriminant_network.get_weights()
)

sdf_test = discriminant_network_test([macro_test, firm_test])

sharpe_loss_test = gan.sharpe_loss(sdf_test)

logging.debug(f"GAN Trained test SHARPE loss: {sharpe_loss_test}")

# %%
# compute beta
y = firm_valid[:, 53, 0]
mask = y != -99.99
X = discriminant_network_valid([macro_valid, firm_valid]).numpy()
y = y[mask]
X = np.squeeze(X[mask])
X = sm.add_constant(X)

y.shape, X.shape

# %%
results = sm.OLS(y, X).fit()
results.summary()

# %%
X = discriminant_network([macro_train, firm_train]).numpy()
X = np.squeeze(X)
X = sm.add_constant(X)
all_results = []
for index in range(firm_train.shape[1]):
    y = firm_train[:, index, 0]
    mask = y != -99.99
    result = sm.OLS(y[mask], sm.add_constant(X[mask])).fit()
    all_results.append(result.params)
np.mean(list(map(lambda x: x[0], all_results)))

# %%
# check if the weights are within (0, 1)
sdf_weights = keras.Model(
    discriminant_network.inputs,
    discriminant_network.get_layer("sdf_w").output
)
weights = sdf_weights([macro_train, firm_train]).numpy()

pd.DataFrame(weights).describe()
