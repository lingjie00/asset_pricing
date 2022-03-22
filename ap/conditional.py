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
    # create input layers
    if macro_network is not None:
        # macro network available
        macro_input = macro_network.get_layer(
            name="macro_input"
        ).input
        macro_output = macro_network.get_layer(
            name="macro_output"
        ).output
        firm_input = layers.Input(
            firm_shape,
            name="firm_input"
        )
        combined_input = layers.Concatenate(
            axis=2,
            name="combined_input"
        )([firm_input, macro_output])
        inputs = layers.Dropout(
            rate=dropout_rate,
            name="moments_dropout"
        )(combined_input)
    else:
        # if no macro network
        firm_input = layers.Input(
            firm_shape,
            name="firm_input"
        )
        inputs = layers.Dropout(
            rate=dropout_rate,
            name="moments_dropout"
        )(firm_input)

    # model
    moments = layers.Dense(
        num_moments,
        name="moments_g",
        activation="linear"  # author: tanh
    )(inputs)
    print(moments.shape)
    moments = layers.Lambda(
        lambda x: tf.transpose(x, perm=[2, 0, 1]),
        name="transpose"
    )(moments)

    # create output
    if macro_network is not None:
        model = keras.Model(
            inputs=[macro_input, firm_input],
            outputs=moments,
            name=name
        )
    else:
        # if no macro network
        model = keras.Model(
            inputs=firm_input,
            outputs=moments,
            name=name
        )
    return model
