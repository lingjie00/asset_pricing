"""Defines GAN model."""
import json
import logging

from conditional import create_generative_network
from data import compute_firm_shape, compute_macro_shape
from macro import create_macro_network
from sdf import create_discriminant_network


def create_gan(
        configpath: str,
        data: dict
):
    """Creates networks used in GAN.

    params:
        configpath: file path to config json file
        data: dictionary containing the following data
            1. returns: excess returns
            2. macro: macroeconomic features
            3. firm: firm characteristics
    """

    ############
    # Get data #
    ############
    returns = data["returns"]
    firm = data["firm"]

    logging.info(f"firm shape: {compute_firm_shape(firm)}")

    if "macro" in data:
        # if macro data is available
        macro = data["macro"]
        logging.info(f"macro shape: {compute_macro_shape(macro)}")

    #########################
    # Model hyper-parameter #
    #########################

    with open(configpath, "r") as configfile:
        config = json.load(configfile)
        param = config["hyperparameters"]
        logging.debug(f"config: {config}")

    _dropout_rate = param["dropout"]  # higher = more dropout
    _discriminant_dense_unts = param["sdf_dense"]  # default: 64
    _generative_moments = param["factors"]  # default: 8
    _mask_key = param["mask_key"]

    ##############
    # train: SDF #
    ##############
    if "macro" in data:
        # if macro data is available
        discriminant_macro = create_macro_network(
            macro_shape=compute_macro_shape(macro),
            num_firms=compute_firm_shape(firm)[0],
            name="discriminant_macro",
            LSTM_units=4,
            dropout_rate=_dropout_rate
        )
    else:
        discriminant_macro = None

    discriminant_network = create_discriminant_network(
        firm_shape=compute_firm_shape(firm),
        macro_network=discriminant_macro,
        returns=returns,
        dense_units=_discriminant_dense_unts,
        dropout_rate=_dropout_rate,
        mask_key=_mask_key
    )

    if "macro" in data:
        logging.info(f"""sdf network output shape:
                {discriminant_network( [macro, firm]).shape}""")
    else:
        logging.info(f"""sdf network output shape:
                {discriminant_network(firm).shape}""")
    logging.info(f"Discriminant network: {discriminant_network.summary()}")

    #############################
    # Train: Generative network #
    #############################
    if "macro" in data:
        # if macro data available
        generative_macro = create_macro_network(
            macro_shape=compute_macro_shape(macro),
            num_firms=compute_firm_shape(firm)[0],
            name="generative_macro",
            LSTM_units=32,
            dropout_rate=_dropout_rate
        )
    else:
        # if no macro data
        generative_macro = None

    generative_network = create_generative_network(
        firm_shape=compute_firm_shape(firm),
        macro_network=generative_macro,
        num_moments=_generative_moments,
        dropout_rate=_dropout_rate
    )

    if "macro" in data:
        logging.info(f"""generative network output shape:
                {generative_network( [macro, firm]).shape}""")
    else:
        logging.info(f"""generative network output shape:
                {generative_network(firm).shape}""")
    logging.info(f"Generative network: {generative_network.summary()}")

    return {
        "discriminant_macro": discriminant_macro,
        "discriminant_network": discriminant_network,
        "generative_macro": generative_macro,
        "generative_network": generative_network
    }
