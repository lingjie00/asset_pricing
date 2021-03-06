"""Contains training procedure for the networks.

train_discriminant: trains sdf network
train_generative: trains generative network
train_gan: trains GAN network
"""
import datetime
import json
import logging
import os
from typing import Union

import tensorflow as tf
import tensorflow.keras as keras

from loss import PricingLoss, sharpe_loss


def train_discriminant(
        discriminant_network: keras.Model,
        optimizer: keras.optimizers,
        inputs: list,
        returns: tf.Tensor,
        moment: tf.Tensor,
        mask_key: float
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
            returns=returns,
            mask_key=mask_key
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
        sdf: tf.Tensor,
        mask_key: float
):
    """Trains generative network."""
    with tf.GradientTape() as tape:
        moment = generative_network(
            inputs,
            training=True
        )
        loss = PricingLoss(
            sdf=sdf,
            moment=moment,
            returns=returns,
            mask_key=mask_key
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


def train_n_discriminant(
    train_data: dict,
    train_networks: dict,
    optimizer: keras.optimizers,
    discriminant_epochs: int,
    patience: int,
    min_epochs: int,
    valid_networks: dict,
    valid_data: dict,
    mask_key: float,
    gan_training: bool,
    save_weight: Union[bool, str] = False,
    load_weight: Union[bool, str] = False,
    train_summary_writer=None,
    valid_summary_writer=None
):
    """Setup"""
    # retrieve networks
    train_discriminant_network = train_networks["discriminant_network"]
    valid_discriminant_network = valid_networks["discriminant_network"]

    # retrieve data
    if "macro" in train_data:
        # if macro data is available
        train_inputs = [train_data["macro"], train_data["firm"]]
        valid_inputs = [valid_data["macro"], valid_data["firm"]]
    else:
        train_inputs = train_data["firm"]
        valid_inputs = valid_data["firm"]
    train_returns = train_data["returns"]
    valid_returns = valid_data["returns"]

    # set names
    if gan_training:
        price_loss_name = "Pricing_loss_GAN"
        sharpe_loss_name = "Sharpe_loss_GAN"
    else:
        price_loss_name = "Pricing_loss"
        sharpe_loss_name = "Sharpe_loss"

    # set path
    if isinstance(load_weight, str):
        if gan_training:
            load_weight += "/weights/gan_discriminant.h5"
        else:
            load_weight += "/weights/discriminant.h5"
    if isinstance(save_weight, str):
        if gan_training:
            save_weight += "/weights/gan_discriminant.h5"
        else:
            save_weight += "/weights/discriminant.h5"

    """Train discriminant"""
    logging.info("Training discriminant network")
    best_epoch = 0
    best_loss = float("inf")
    wait = 0
    if gan_training:
        train_moment = train_networks["generative_network"](
            train_inputs, training=False)
        valid_moment = valid_networks["generative_network"](
            valid_inputs, training=False)
    else:
        train_shape = train_data["firm"].shape
        valid_shape = valid_data["firm"].shape
        train_moment = tf.ones(
            shape=(1,
                   train_shape[0],
                   train_shape[1])
        )
        valid_moment = tf.ones(
            shape=(1,
                   valid_shape[0],
                   valid_shape[1])
        )

    if load_weight is not False:
        # load discriminant weights
        train_discriminant_network.load_weights(load_weight)
        logging.info("Loaded Discriminant weights")

    for epoch in range(discriminant_epochs+1):
        # train discriminant network for n epochs
        train_price_loss = train_discriminant(
            discriminant_network=train_discriminant_network,
            optimizer=optimizer,
            inputs=train_inputs,
            returns=train_returns,
            moment=train_moment,
            mask_key=mask_key
        )
        train_sharpe_loss = sharpe_loss(
            train_discriminant_network(train_inputs, training=False)
        )
        # copy the weights from training to validation networks
        valid_discriminant_network.set_weights(
            train_discriminant_network.get_weights())
        # get loss from validation networks
        valid_sdf = valid_discriminant_network(
            valid_inputs, training=False)
        valid_price_loss = PricingLoss(
            sdf=valid_sdf,
            moment=valid_moment,
            returns=valid_returns,
            mask_key=mask_key
        )
        valid_sharpe_loss = sharpe_loss(valid_sdf)
        # record loss in tensorboard
        if train_summary_writer is not None:
            # record loss in tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    price_loss_name, train_price_loss, step=epoch)
                tf.summary.scalar(
                    sharpe_loss_name, train_sharpe_loss, step=epoch)
            with valid_summary_writer.as_default():
                tf.summary.scalar(
                    price_loss_name, valid_price_loss, step=epoch)
                tf.summary.scalar(
                    sharpe_loss_name, valid_sharpe_loss, step=epoch)
        # early stopping and restore best weight
        # early stopping is based on validation loss to prevent overfitting
        wait += 1
        improvement = (valid_price_loss - best_loss) / best_loss
        if improvement <= -0.001 or best_loss == float("inf"):
            # record new best weights if improvement at least 0.1%
            best_epoch = epoch
            best_loss = valid_price_loss
            wait = 0
            weights = train_discriminant_network.get_weights()
        if wait >= patience and epoch >= min_epochs:
            # stop training
            message = ""
            message += f"Restore weights at epoch {best_epoch}: "
            # log training loss
            message += f"Mis-Pricing loss (train): {train_price_loss:.6f}; "
            message += f"SHARPE (train): {train_sharpe_loss:.6f}; "
            # log validation loss
            message += f"Mis-Pricing loss (valid): {valid_price_loss:.6f}; "
            message += f"SHARPE (valid): {valid_sharpe_loss:.6f}"
            logging.info(message)
            break
    # at the end of training, report the total training
    # epochs and reload the best weights
    logging.info(f"Total Discriminant training epochs {epoch}")
    train_discriminant_network.set_weights(weights)
    if save_weight is not False:
        train_discriminant_network.save_weights(save_weight)
        logging.info("Saved Discriminant weights")

    # update the networks
    train_networks["discriminant_network"] = train_discriminant_network
    valid_networks["discriminant_network"] = valid_discriminant_network

    return train_networks, valid_networks


def train_n_generative(
    train_data: dict,
    train_networks: dict,
    optimizer: keras.optimizers,
    generative_epochs: int,
    patience: int,
    min_epochs: int,
    valid_networks: dict,
    valid_data: dict,
    mask_key: float,
    gan_training: bool,
    save_weight: Union[bool, str] = False,
    load_weight: Union[bool, str] = False,
    train_summary_writer=None,
    valid_summary_writer=None
):
    """Setup"""
    # retrieve networks
    train_discriminant_network = train_networks["discriminant_network"]
    train_generative_network = train_networks["generative_network"]
    valid_discriminant_network = valid_networks["discriminant_network"]
    valid_generative_network = valid_networks["generative_network"]

    # retrieve data
    if "macro" in train_data:
        # if macro data is available
        train_inputs = [train_data["macro"], train_data["firm"]]
        valid_inputs = [valid_data["macro"], valid_data["firm"]]
    else:
        train_inputs = train_data["firm"]
        valid_inputs = valid_data["firm"]
    train_returns = train_data["returns"]
    valid_returns = valid_data["returns"]

    # set names
    if gan_training:
        price_loss_name = "Pricing_loss_generative_GAN"
    else:
        price_loss_name = "Pricing_loss_generative"

    # set path
    if isinstance(load_weight, str):
        if gan_training:
            load_weight += "/weights/gan_generative.h5"
        else:
            load_weight += "/weights/generative.h5"
    if isinstance(save_weight, str):
        if gan_training:
            save_weight += "/weights/gan_generative.h5"
        else:
            save_weight += "/weights/generative.h5"

    """Train generative"""
    logging.info("Training generative network")
    best_epoch = 0
    best_loss = float("-inf")
    wait = 0
    # fix the SDF from train discriminant to use for train
    # generative network training
    train_sdf = train_discriminant_network(train_inputs, training=False)
    valid_sdf = valid_discriminant_network(valid_inputs, training=False)

    if load_weight is not False:
        train_generative_network.load_weights(load_weight)
        logging.info("Loaded Generative weights")

    for epoch in range(generative_epochs+1):
        # train generative network for n epochs
        train_price_loss = train_generative(
            generative_network=train_generative_network,
            optimizer=optimizer,
            inputs=train_inputs,
            returns=train_returns,
            sdf=train_sdf,
            mask_key=mask_key
        )
        # copy the weights from training to validation networks
        valid_generative_network.set_weights(
            train_generative_network.get_weights())
        # get loss from validation networks
        valid_price_loss = PricingLoss(
            sdf=valid_sdf,
            moment=valid_generative_network(
                valid_inputs, training=False),
            returns=valid_returns,
            mask_key=mask_key
        )
        if train_summary_writer is not None:
            # record loss in tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    price_loss_name, train_price_loss, step=epoch)
            with valid_summary_writer.as_default():
                tf.summary.scalar(
                    price_loss_name, valid_price_loss, step=epoch)
        # early stopping and restore best weight
        wait += 1
        improvement = (valid_price_loss - best_loss) / best_loss
        if improvement >= 0.001 or best_loss == float("-inf"):
            # record new best weights if improvement at least 0.1%
            best_epoch = epoch
            best_loss = valid_price_loss
            wait = 0
            gweights = train_generative_network.get_weights()
        if wait >= patience and epoch >= min_epochs:
            # stop training
            message = ""
            message += f"Restore weights at epoch {best_epoch}: "
            # log training loss
            message += f"Mis-Pricing loss (train): {train_price_loss:.6f}; "
            # log validation loss
            message += f"Mis-Pricing loss (valid): {valid_price_loss:.6f}; "
            logging.info(message)
            break
    # at the end of training, report the total training
    # epochs and reload the best weights
    logging.info(f"Total Generative training epochs {epoch}")
    train_generative_network.set_weights(gweights)
    if save_weight is not False:
        train_generative_network.save_weights(save_weight)
        logging.info("Saved Generative weights")

    # update the networks
    train_networks["discriminant_network"] = train_discriminant_network
    train_networks["generative_network"] = train_generative_network
    valid_networks["discriminant_network"] = valid_discriminant_network
    valid_networks["generative_network"] = valid_generative_network

    return train_networks, valid_networks


def train_gan(
    configpath: str,
    train_data: dict,
    train_networks: dict,
    valid_networks: dict,
    valid_data: dict,
    use_cpu: bool = False
):
    """Test GPU availability"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logging.info(f"Found GPU with info: {gpus}")
    elif use_cpu is False:
        logging.info("No GPU found")
        raise Exception("No GPU")
    else:
        logging.info("No GPU found, using CPU")

    """Create TensorFlow summary writer"""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = 'logs/gradient_tape/' + current_time
    train_log_dir = dir + '/train'
    valid_log_dir = dir + '/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    """Load config"""
    with open(configpath, "r") as file:
        # load config
        config = json.load(file)
        # export config to the log
        with open(dir + "/config.json", "w") as writer:
            json.dump(config, writer)
        logging.info(f"config: {config}")
        config_training = config["training"]
        mask_key = config["hyperparameters"]["mask_key"]
        discriminant_unc_epochs = config_training["discriminant_unc_epochs"]
        generative_unc_epochs = config_training["generative_unc_epochs"]
        discriminant_epochs = config_training["discriminant_epochs"]
        generative_epochs = config_training["generative_epochs"]
        gan_epochs = config_training["gan_epochs"]
        min_epochs = config_training["min_epochs"]
        patience = config_training["patience"]
        load_weight = config_training["load_weight"]
        if config_training["save_weight"] is True:
            save_weight = dir
            os.mkdir(f"{dir}/weights")
        else:
            save_weight = False

    """Create optimizer"""
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    """Train discriminant"""
    train_networks, valid_networks = train_n_discriminant(
        train_data=train_data,
        train_networks=train_networks,
        optimizer=optimizer,
        discriminant_epochs=discriminant_unc_epochs,
        patience=patience,
        min_epochs=min_epochs,
        valid_networks=valid_networks,
        valid_data=valid_data,
        mask_key=mask_key,
        save_weight=save_weight,
        load_weight=load_weight,
        gan_training=False,
        train_summary_writer=train_summary_writer,
        valid_summary_writer=valid_summary_writer
    )

    """Train generative"""
    train_networks, valid_networks = train_n_generative(
        train_data=train_data,
        train_networks=train_networks,
        optimizer=optimizer,
        generative_epochs=generative_unc_epochs,
        patience=patience,
        min_epochs=min_epochs,
        valid_networks=valid_networks,
        valid_data=valid_data,
        mask_key=mask_key,
        save_weight=save_weight,
        load_weight=load_weight,
        gan_training=False,
        train_summary_writer=train_summary_writer,
        valid_summary_writer=valid_summary_writer
    )

    """Train GAN"""
    if "macro" in valid_data:
        valid_inputs = [valid_data["macro"], valid_data["firm"]]
    else:
        valid_inputs = [valid_data["firm"]]
    best_gan_sharpe = 0
    wait = 0
    for _ in range(gan_epochs):
        try:
            logging.info(f"GAN epoch: {_}")
            train_networks, valid_networks = train_n_discriminant(
                train_data=train_data,
                train_networks=train_networks,
                optimizer=optimizer,
                discriminant_epochs=discriminant_epochs,
                patience=patience,
                min_epochs=min_epochs,
                valid_networks=valid_networks,
                valid_data=valid_data,
                mask_key=mask_key,
                save_weight=save_weight,
                load_weight=load_weight,
                gan_training=True,
                train_summary_writer=train_summary_writer,
                valid_summary_writer=valid_summary_writer
            )
            train_networks, valid_networks = train_n_generative(
                train_data=train_data,
                train_networks=train_networks,
                optimizer=optimizer,
                generative_epochs=generative_epochs,
                patience=patience,
                min_epochs=min_epochs,
                valid_networks=valid_networks,
                valid_data=valid_data,
                mask_key=mask_key,
                save_weight=save_weight,
                load_weight=load_weight,
                gan_training=True,
                train_summary_writer=train_summary_writer,
                valid_summary_writer=valid_summary_writer
            )
            # check if Sharpe loss has increase
            wait += 1
            new_gan_sharpe = sharpe_loss(
                valid_networks["discriminant_network"](
                    valid_inputs, training=False)
            )
            improvement = (new_gan_sharpe - best_gan_sharpe) / best_gan_sharpe
            if improvement >= 0.001:
                # if positive improvement of at least 0.1%%
                # continue training
                best_gan_sharpe = new_gan_sharpe
                wait = 0
            elif wait > min_epochs:
                # else break training
                break
        except KeyboardInterrupt:
            # quit training mid way
            logging.info("Keyboard Interrupt training")
            break
