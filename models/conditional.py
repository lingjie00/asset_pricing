"""Create a Conditional Network."""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def create_generative_network(
        firm_shape: tuple,
        macro_network: keras.Model,
        num_moments: int,
        dropout_rate: float,
        name: str = "generative_network"
):
    """Create generative network.

    params:
        firm_shape: tuple of (num of firms, num of chars, )
        macro_network: macro LSTM network
        name: name of model
        num_moments: num of latent factors to generate
    """
    macro_input = macro_network.get_layer(
        name="macro_input"
    ).input
    macro_output = macro_network.get_layer(
        name="macro_output"
    ).output
    inputs = layers.Input(
        firm_shape,
        name="firm_input"
    )
    combined_input = layers.Concatenate(
        axis=2,
        name="combined_input"
    )([inputs, macro_output])
    combined_input = layers.Dropout(
        rate=dropout_rate,
        name="moments_dropout"
    )(combined_input)
    moments = layers.Dense(
        num_moments,
        name="moments_g",
        activation="linear"  # author: tanh
    )(combined_input)
    moments = layers.Lambda(
        lambda x: tf.transpose(x, perm=[2, 0, 1]),
        name="transpose"
    )(moments)
    model = keras.Model(
        inputs=[macro_input, inputs],
        outputs=moments,
        name=name
    )
    return model
