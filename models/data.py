"""Wrapper to iterate through data."""
import numpy as np


class Data(object):
    """Pre-process data for training."""

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

        # Firm dta should be in 3 dim
        # for model to process it
        if len(firm_data.shape) != 3:
            num_firms, num_features = firm_data.shape
            firm_data = firm_data.reshape(-1, num_firms, num_features)

        return_data, firm_data = firm_data[:, :, 0], firm_data[:, :, 1:]
        mask = return_data != -99.99
        return macro_data, firm_data, return_data, mask
