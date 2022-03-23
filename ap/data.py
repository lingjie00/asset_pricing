"""Wrapper to iterate through data."""
import numpy as np
import tensorflow as tf


def compute_macro_shape(macro_data: tf.Tensor):
    """With given data, return the macro shape."""
    # macro_data dim: years x features
    macro_feature = macro_data.shape[1]
    # increase one dim as LSTM require 3 dim shape
    return (macro_feature, 1, )


def compute_firm_shape(firm_data: tf.Tensor):
    """With given data, return the firm shape."""
    # firm data dim: years x firms x chars
    n = firm_data.shape[1]
    chars = firm_data.shape[2]
    return (n, chars, )


def masking(
        inputs: tf.Tensor,
        returns: tf.Tensor,
        mask_key: float,
) -> tf.Tensor:
    """Mask the inputs.

    Cannot use Masking Layer as our masked value rely on the
    Returns, and not the characteristic values."""
    mask = returns != mask_key
    masked_input = tf.boolean_mask(inputs, mask)
    return masked_input


class ChenData(object):
    """Pre-process Chen's data for training."""

    def __init__(self):
        """Init."""
        self.name = "Pre-process data."

    def clean(self,
              macro_data: np.array,
              firm_data: np.array):
        """Pre-process the data.

        (macro, firm, return, mask)
        """
        # Macro data should be in 3 dim
        # for LSTM to accept it
        if len(macro_data.shape) != 3:
            num_row, num_features = macro_data.shape
            macro_data = macro_data.reshape(num_row, 1, num_features)

        # Firm data should be in 3 dim
        # for model to process it
        if len(firm_data.shape) != 3:
            num_firms, num_features = firm_data.shape
            firm_data = firm_data.reshape(-1, num_firms, num_features)

        return_data, firm_data = firm_data[:, :, 0], firm_data[:, :, 1:]
        mask = return_data != -99.99
        return macro_data, firm_data, return_data, mask
