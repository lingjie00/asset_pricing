"""Defines the loss function for GAN model."""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class PricingLoss(keras.losses.Loss):
    """Defines the Pricing Loss for GAN model."""
    def __init__(self):
        """Init."""
        super().__init__()

    def __call__(self,
                 sdf: tf.Tensor,
                 moment: tf.Tensor,
                 return_data: tf.Tensor,
                 mask: tf.Tensor,
                 verbose: bool = False
                 ):
        """Compute GAN training loss."""
        ###################
        # Preprocess data #
        ###################
        mask = tf.cast(mask, "float")
        Ti = tf.reduce_sum(mask, axis=0)

        ################
        # Compute Loss #
        ################
        if verbose:
            print(f"mask shape = {mask.shape}")
            print(f"return data shape = {return_data.shape}")
            print(f"SDF shape: {sdf.shape}")
            print(f"moment shape: {moment.shape}")

        masked_return = return_data * mask
        if verbose:
            print(f"masked return shape: {masked_return.shape}")

        empirical_sum = sdf * masked_return * moment
        empirical_sum = tf.reduce_sum(empirical_sum, axis=1)
        if verbose:
            print(f"empirical mean shape: {empirical_sum.shape}")

        empirical_mean = empirical_sum / Ti
        if verbose:
            print(f"empirical mean shape: {empirical_sum.shape}")

        loss = tf.square(empirical_mean)
        if verbose:
            print(f"loss shape: {loss.shape}")

        # loss is weighted with the number of valid periods
        Tmax = tf.reduce_max(Ti)  # max time period
        loss_weight = Ti / Tmax
        loss *= loss_weight

        loss = tf.reduce_mean(loss)
        if verbose:
            print(f"normalized loss shape: {loss.shape}")
            print(loss)
        return loss

    def sharpe(self, portfolio: tf.Tensor):
        """Calculate the SHARPE based on a portfolio of returns."""
        if not (type(portfolio) == np.ndarray):
            portfolio = portfolio.numpy()
        return np.mean(portfolio / portfolio.std())

    def sharpe_loss(self,
                    sdf: tf.Tensor):
        """Calculate the SHARPE as a loss based on sdf."""
        portfolio = 1 - sdf
        return self.sharpe(portfolio)
