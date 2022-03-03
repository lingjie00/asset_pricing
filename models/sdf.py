"""Implement Stochastic Discount Factor (SDF) Network."""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from .data import masking


def compute_sdf(
        sdf_w: tf.Tensor,
        returns: tf.Tensor,
        mask_key: float,
) -> tf.Tensor:
    """Computes the SDF M_t+1 with given weight and returns.

    sdf should be time variant, firm invariant."""

    # pre-processing data:
    # 1. reduce dimension of sdf weights
    # 2. create mask based on returns and mask key
    # 3. convert returns data types if not already float32
    sdf_w = tf.reshape(sdf_w, [-1])
    mask = returns != mask_key
    if returns.dtype != "float32":
        # change data type to float32 for all tensors
        returns = tf.cast(returns, "float32")

    # construct portfolio (aka weighted return)
    returns = tf.boolean_mask(returns, mask)
    weighted_return = tf.multiply(returns, sdf_w)

    # masked portfolio (due to some firms having missing value)
    num_obs = tf.reduce_sum(tf.cast(mask, "int32"), axis=1)
    masked_weighted_return = tf.split(weighted_return, num_obs)
    # each item here is tangency portfolio at time t
    items = []
    for item in masked_weighted_return:
        item = tf.reduce_sum(item, keepdims=True)
        items.append(item)
    portfolio = tf.concat(items, axis=0)

    # construct sdf
    # author use sdf = 1 + portfolio in code
    # however, in paper it was written as sdf = 1 - portfolio
    # we kept the paper implementation
    # there is no difference in the end result
    sdf = 1 - portfolio
    sdf = tf.expand_dims(sdf, axis=1)
    return sdf


def create_discriminant_network(
    firm_shape: tuple,
    macro_network: keras.Model,
    returns: tf.Tensor,
    dense_units: int,
    dropout_rate: float,
    mask_key: float,
    name: str = "discriminant_network"
):
    """Creates discriminant network.

    params:
        firm_shape: tuple of (num of firms, num of chars, )
        macro_network: macro LSTM network
        name: name of the model
        dense_units: num of hidden units in dense layer
    """
    macro_input = macro_network.get_layer(
        name="macro_input"
    ).input
    macro_output = macro_network.get_layer(
        name="macro_output").output
    firm_input = layers.Input(
        firm_shape,
        name="firm_input"
    )
    combined_input = layers.Concatenate(
        axis=2,
        name="combined_input")([
            firm_input, macro_output
        ])
    combined_input = layers.Lambda(
        lambda x: masking(x, returns, mask_key),
        name="masked_input"
    )(combined_input)
    dense1 = layers.Dense(
        dense_units,
        name="dense1",
        activation="relu"
    )(combined_input)
    dense1 = layers.Dropout(
        rate=dropout_rate,
        name="dropout1"
    )(dense1)
    dense2 = layers.Dense(
        dense_units,
        name="dense2",
        activation="relu"
    )(dense1)
    dense2 = layers.Dropout(
        rate=dropout_rate,
        name="dropout2"
    )(dense2)
    sdf_w = layers.Dense(
        1,
        name="sdf_w",
        activation="linear"
    )(dense2)
    sdf = layers.Lambda(
        lambda w: compute_sdf(w, returns, mask_key),
        name="sdf"
    )(sdf_w)
    network = keras.Model(
        inputs=[macro_input, firm_input],
        outputs=sdf,
        name=name
    )

    return network
