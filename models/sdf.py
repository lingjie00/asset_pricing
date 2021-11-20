"""Implement Stochastic Discount Factor (SDF) Network."""
import tensorflow as tf
import tensorflow.keras as keras

"""
Model structure:
=macro data= -> |RNN| + =firm data= -> |FFN| -> |SDF|

Input data:
    - macro data (time series)
    - firm data (time series, entity level)

Output:
    - SDF
"""


class SDFModel(keras.Model):
    """Builds a SDF model.

    Procedure:
        Pre-process data
        -> Macro data RNN layer
        -> (Macro + firm data) Dense layer
        -> compute SDF
    """

    def __init__(self,
                 LSTM_units: int = 4,
                 Dense_units: int = 32,
                 Dropout_rate: float = 0.50):
        """Init model."""
        super().__init__(name="SDF")
        self.lstm = keras.layers.LSTM(units=LSTM_units, name="State_RNN")
        self.dense1 = keras.layers.Dense(units=Dense_units, activation="relu",
                                         name="SDF_dense1")
        self.dense2 = keras.layers.Dense(units=Dense_units, activation="relu",
                                         name="SDF_dense2")
        self.dense_output = keras.layers.Dense(units=1, activation="linear",
                                               name="SDF_w")
        self.dropout = keras.layers.Dropout(Dropout_rate)

    def call(self,
             inputs: list,
             training: bool,
             verbose: bool = False
             ):
        """Defines the network architecture.

        :param inputs: Input data = [macroeconomic data, firm data]
        macroeconomic data: Time * macro feature dimension * 1
        firm data: Time * No firms * firm feature dimension + 1
                    + 1 for the Return data
        :param training: only during training will we use dropout
        """
        ###################
        # Data processing #
        ###################
        macro_data, firm_data, return_data, mask = inputs

        ########################
        # Macro data RNN layer #
        ########################
        # Macro data -> Dropout -> LSTM
        if verbose:
            if training:
                print("Training SDF")
            else:
                print("Get SDF weight")

        h = self.lstm(macro_data)
        h = self.dropout(h, training=training)

        ###################################
        # (Macro + firm data) Dense layer #
        ###################################
        # Merge macro data with firm data
        # We need to repeat macro data for each firm
        # Therefore, the final macro data dimension:
        #   Time * No firms * macro feature
        num_firms = firm_data.shape[1]
        h = tf.expand_dims(h, axis=1)  # insert Tensor of 1s in axis
        h = tf.tile(h, [1, num_firms, 1])  # repeat macro data for each firm

        if verbose:
            print(f"num firms = {num_firms}")
            print(f"macro LSTM shape = {h.shape}")

        # The input data will be dim:
        #  total time * (macro feature + firm feature)
        firm_data = tf.cast(firm_data, "float")
        h = tf.cast(h, "float")
        concat = tf.concat([firm_data, h], axis=2)

        if verbose:
            print(f"firm data shape = {firm_data.shape}")
            print(f"concat shape = {concat.shape}")

        # mask input data
        if verbose:
            print(f"mask shape = {mask.shape}")
        mask = tf.cast(mask, "float")
        concat_mask = tf.expand_dims(mask, axis=2)
        concat_mask = tf.repeat(concat_mask, concat.shape[2], axis=2)

        if verbose:
            print(f"concat mask shape = {concat_mask.shape}")
        masked_concat = tf.multiply(concat, concat_mask)

        if verbose:
            print(f"masked concat shape = {masked_concat.shape}")

        # Training macro + firm data
        w = self.dense1(masked_concat)
        w = self.dropout(w, training=training)
        w = self.dense2(w)
        w = self.dropout(w, training=training)
        w = self.dense_output(w)  # dim = (time entry, num firms)
        w = tf.squeeze(w)

        if verbose:
            print(f"SDF weight shape = {w.shape}")

        weighted_return = w * return_data * mask
        SDF = 1 - tf.reduce_sum(weighted_return, axis=1)
        SDF = tf.expand_dims(SDF, axis=1)

        return SDF
