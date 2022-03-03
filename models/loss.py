"""Loss functions used in training."""
import tensorflow as tf


def PricingLoss(
    sdf: tf.Tensor,
    moment: tf.Tensor,
    returns: tf.Tensor,
    mask_key: float
) -> tf.Tensor:
    """Compute Pricing Loss for GAN.

    Loss is based on No Arbitrage equation and pricing error.
    No Arbitrage equation: E(M_t+1 x R^e) = 0
    Pricing error: E(M_t+1 x R^e) != 0

    Params:
        sdf: stochastic discount factor
        moment: factors
        returns: excess returns
        mask_key: mask value
    """
    # compute mask
    mask = returns != mask_key

    # convert dtypes, ensure all data types are float32
    moment = tf.cast(moment, "float32")
    sdf = tf.cast(sdf, "float32")
    returns = tf.cast(returns, "float32")
    float_mask = tf.cast(mask, "float32")

    # compute the actual observation per period for each firm
    total_obs = tf.reduce_sum(float_mask, axis=0)

    # compute loss
    masked_return = tf.multiply(returns, float_mask)
    no_arbitrage = tf.multiply(masked_return, sdf)
    pricing_loss = tf.multiply(no_arbitrage, moment)
    empirical_sum = tf.reduce_sum(
        pricing_loss,
        axis=1
    )
    empirical_mean = tf.divide(empirical_sum, total_obs)
    loss = tf.square(empirical_mean)

    # weighted loss
    max_obs = tf.reduce_max(total_obs)
    loss_weight = tf.divide(total_obs, max_obs)
    weighted_loss = tf.multiply(loss, loss_weight)

    # average loss
    reduce_loss = tf.reduce_mean(weighted_loss)

    return reduce_loss


def sharpe(portfolio: tf.Tensor) -> tf.Tensor:
    """Calculate the SHARPE based on a portfolio of returns.

    params:
        portfolio: excess returns over time

    returns:
        SHARPE loss (tf.Tensor)
    """
    assert len(tf.squeeze(portfolio).shape) == 1

    mean = tf.math.reduce_mean(portfolio)
    std = tf.math.reduce_std(portfolio)
    portfolio_sharpe = tf.divide(mean, std)
    return portfolio_sharpe


def sharpe_loss(sdf: tf.Tensor) -> tf.Tensor:
    """Calculate the SHARPE as a loss based on sdf.

    params:
        sdf: stochastic discount factor

    returns:
        SHARPE loss (tf.Tensor)
    """
    # since sdf = 1 - efficient portfolio
    # efficient portfolio = 1 - sdf by construction
    portfolio = 1 - sdf
    return sharpe(portfolio)
