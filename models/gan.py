"""Defines GAN model."""
import logging

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

#########################
# Model hyper-parameter #
#########################
_mask_key = -99.99  # default: -99.99
_dropout_rate = 0.05  # higher = more dropout
_discriminant_dense_unts = 64  # default: 64
_generative_moments = 8  # default: 8


#######
# SDF #
#######
def compute_sdf(sdf_w: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the SDF M_t+1 with given weight and returns.

    sdf should be time variant, firm invariant."""

    # pre-processing data:
    # 1. reduce dimension of sdf weights
    # 2. create mask based on returns and mask key
    # 3. convert returns data types if not already float32
    sdf_w = tf.reshape(sdf_w, [-1])
    mask = returns != _mask_key
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


def masking(inputs: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Mask the inputs.

    Cannot use Masking Layer as our masked value rely on the
    Returns, and not the characteristic values."""
    mask = returns != _mask_key
    masked_input = tf.boolean_mask(inputs, mask)
    return masked_input


########
# Loss #
########
def PricingLoss(
    sdf: tf.Tensor,
    moment: tf.Tensor,
    returns: tf.Tensor,
    mask_key: float = _mask_key
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


##########
# Models #
##########


def create_macro_network(
        macro_shape: tuple,
        num_firms: int,
        name: str,
        LSTM_units: int,
        dropout_rate: float = _dropout_rate
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


def create_discriminant_network(
    firm_shape: tuple,
    macro_network: keras.Model,
    returns: tf.Tensor,
    name: str = "discriminant_network",
    dense_units: int = _discriminant_dense_unts,
    dropout_rate: float = _dropout_rate
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
        lambda x: masking(x, returns),
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
        lambda w: compute_sdf(w, returns),
        name="sdf"
    )(sdf_w)
    network = keras.Model(
        inputs=[macro_input, firm_input],
        outputs=sdf,
        name=name
    )

    return network


def create_generative_network(
        firm_shape: tuple,
        macro_network: keras.Model,
        name: str = "generative_network",
        num_moments: int = _generative_moments
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
        rate=_dropout_rate,
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


def train_discriminant(
        discriminant_network: keras.Model,
        optimizer: keras.optimizers,
        inputs: list,
        returns: tf.Tensor,
        moment: tf.Tensor
):
    """Trains discriminant network."""
    with tf.GradientTape() as tape:
        sdf = discriminant_network(
            inputs,
            training=True
        )
        loss = PricingLoss(
            sdf=sdf,
            moment=moment,
            returns=returns
        )
    gradients = tape.gradient(
        loss, discriminant_network.trainable_variables
    )
    optimizer.apply_gradients(
        zip(gradients, discriminant_network.trainable_variables)
    )
    return loss


def train_generative(
        generative_network: keras.Model,
        optimizer: keras.optimizers,
        inputs: list,
        returns: tf.Tensor,
        sdf: tf.Tensor
):
    """Trains generative network."""
    macro_data, firm_data = inputs

    with tf.GradientTape() as tape:
        moment = generative_network(
            inputs,
            training=True
        )
        loss = PricingLoss(
            sdf=sdf,
            moment=moment,
            returns=returns
        )
        # maximising error
        loss = - loss
    gradients = tape.gradient(
        loss, generative_network.trainable_variables
    )
    optimizer.apply_gradients(
        zip(gradients, generative_network.trainable_variables)
    )
    return -loss


def train(
    inputs: list,
    returns: tf.Tensor,
    optimizer: keras.optimizers,
    discriminant_network: keras.Model,
    generative_network: keras.Model,
    discriminant_epochs: int,
    generative_epochs: int,
    gan_epochs: int,
    patience: int,
    min_epochs: int,
    verbose_interval: int,
    valid_inputs: list = None,
    valid_returns: tf.Tensor = None,
    valid_discriminant: keras.Model = None,
    save_weights: list[bool, bool, bool] = [False, False, False],
    load_weights: list[bool, bool, bool] = [False, False, False]
):
    """Trains GAN."""
    # train discriminant
    best_epoch = 0
    best_loss = float("inf")
    wait = 0
    dummy_moment = tf.ones(
        shape=(1,
               inputs[1].shape[0],
               inputs[1].shape[1])
    )

    if load_weights[0]:
        discriminant_network.load_weights("./saved_weights/discriminant")
        logging.info("Loaded Discriminant weights")

    for epoch in range(discriminant_epochs+1):
        loss = train_discriminant(
            discriminant_network=discriminant_network,
            optimizer=optimizer,
            inputs=inputs,
            returns=returns,
            moment=dummy_moment)
        sdf = discriminant_network(inputs, training=False)
        sloss = sharpe_loss(sdf)
        valid_discriminant.set_weights(
            discriminant_network.get_weights())
        sdf_valid = valid_discriminant(valid_inputs)
        valid_sloss = sharpe_loss(sdf_valid)
        if epoch % verbose_interval == 0:
            message = ""
            message += f"Discriminant epoch {epoch}: "
            message += f"Mis-Pricing (train): {loss:.6f}; "
            message += f"SHARPE (train): {sloss:.6f}; "
            # print validation results
            message += f"SHARPE (valid): {valid_sloss:.6f}"
            print(message)
        # early stopping and restore best weight
        wait += 1
        if loss < best_loss:
            best_epoch = epoch
            best_loss = loss
            wait = 0
            weights = discriminant_network.get_weights()
        if wait >= patience and epoch >= min_epochs:
            message = ""
            message += f"Restore weights at epoch {best_epoch}: "
            message += f"Mis-Pricing (train): {best_loss}; "
            message += f"SHARPE (train): {sloss}; "
            print(message)
            break
    print(f"Total Discriminant training epochs {best_epoch}")
    discriminant_network.set_weights(weights)
    if save_weights[0]:
        discriminant_network.save_weights("./saved_weights/discriminant")
        logging.info("Saved Discriminant weights")

    # train generative
    best_epoch = 0
    best_loss = float("-inf")
    wait = 0
    sdf = discriminant_network(inputs, training=False)

    if load_weights[1]:
        generative_network.load_weights("./saved_weights/generative")
        logging.info("Loaded Generative weights")

    for epoch in range(generative_epochs+1):
        loss = train_generative(
            generative_network=generative_network,
            optimizer=optimizer,
            inputs=inputs,
            returns=returns,
            sdf=sdf
        )
        if epoch % verbose_interval == 0:
            message = ""
            message += f"Generative epoch {epoch}: "
            message += f"Mis-Pricing (train): {loss:.6f}; "
            print(message)
        # early stopping and restore best weight
        wait += 1
        if loss > best_loss:
            best_epoch = epoch
            best_loss = loss
            wait = 0
            gweights = generative_network.get_weights()
        if wait >= patience and epoch >= min_epochs:
            message = ""
            message += f"Restore weights at epoch {best_epoch}: "
            message += f"Mis-Pricing (train): {best_loss}; "
            print(message)
            break
    print(f"Total Generative training epochs {best_epoch}")
    generative_network.set_weights(gweights)
    if save_weights[1]:
        generative_network.save_weights("./saved_weights/generative")
        logging.info("Saved Generative weights")

    # train GAN
    best_epoch = 0
    best_loss = float("inf")
    wait = 0

    if load_weights[2]:
        discriminant_network.load_weights("./saved_weights/gan_dis")
        logging.info("Loaded Discriminant GAN weights")
        generative_network.load_weights("./saved_weights/gan_gen")
        logging.info("Loaded Generative GAN weights")

    for epoch in range(gan_epochs+1):
        sdf = discriminant_network(inputs, training=False)
        train_generative(
            generative_network=generative_network,
            optimizer=optimizer,
            inputs=inputs,
            returns=returns,
            sdf=sdf
        )
        moment = generative_network(inputs, training=False)
        loss = train_discriminant(
            discriminant_network=discriminant_network,
            optimizer=optimizer,
            inputs=inputs,
            returns=returns,
            moment=moment
        )
        sdf = discriminant_network(inputs, training=False)
        sloss = sharpe_loss(sdf)
        valid_discriminant.set_weights(
            discriminant_network.get_weights()
        )
        sdf_valid = valid_discriminant(valid_inputs)
        valid_sloss = sharpe_loss(sdf_valid)
        if epoch % verbose_interval == 0:
            message = ""
            message += f"GAN epoch {epoch}: "
            message += f"Mis-Pricing (train): {loss:.6f}; "
            message += f"SHARPE (train): {sloss:.6f}; "
            # print validation results
            message += f"SHARPE (valid): {valid_sloss:.6f}"
            print(message)
        # early stopping and restore best weight
        # TODO: GAN needs a different early stopping
        # wait += 1
        # if loss < best_loss:
        #     best_epoch = epoch
        #     best_loss = loss
        #     wait = 0
        #     gweights = generative_network.get_weights()
        #     weights = discriminant_network.get_weights()
        # if wait >= patience and epoch >= min_epochs:
        #     message = ""
        #     message += f"Restore weights at epoch {best_epoch}: "
        #     message += f"Mis-Pricing (train): {best_loss}; "
        #     message += f"SHARPE (train): {sloss}; "
        #     print(message)
        #     break
    # print(f"Total GAN training epochs {best_epoch}")
    # generative_network.set_weights(gweights)
    # discriminant_network.set_weights(weights)
    print(f"Total GAN training epochs {epoch}")
    if save_weights[2]:
        discriminant_network.save_weights("./saved_weights/gan_dis")
        logging.info("Saved Discriminant GAN weights")
        generative_network.save_weights("./saved_weights/gan_gen")
        logging.info("Saved Generative GAN weights")
