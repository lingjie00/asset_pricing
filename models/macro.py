"""Creates Macroeconomic network."""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def create_macro_network(
        macro_shape: tuple,
        num_firms: int,
        name: str,
        LSTM_units: int,
        dropout_rate: float
):
    """Creates Macroeconomic Network.

    params:
        macro_shape: tuple of (num of macro features, 1, )
        num_firms: number of firms,
            used to expand the LSTM output
        name: name of the network
        LSTM_units: number of hidden units in LSTM layer
        Dropout_rate: dropout rate for LSTM output

    return:
        macro network
    """
    macro_input = layers.Input(
        macro_shape,
        name="macro_input")
    macro_lstm = layers.Dropout(
        rate=dropout_rate,
        name="macro_dropout"
    )(macro_input)
    macro_lstm = layers.LSTM(
        LSTM_units,
        name="macro_lstm"
    )(macro_lstm)

    # expand output
    # to be merged with firm data later
    macro_output = layers.Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="expand_dim"
    )(macro_lstm)

    macro_output = layers.Lambda(
        lambda x: tf.tile(x, [1, num_firms, 1]),
        name="macro_output"
    )(macro_output)

    # define model
    macro_network = keras.Model(
        inputs=macro_input,
        outputs=macro_output,
        name=name
    )

    return macro_network
