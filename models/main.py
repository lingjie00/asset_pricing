"""Conduct end to end training."""
import numpy as np
import tensorflow as tf

from data import Data
from sdf import SDFModel
from conditional import ConditionalModel
from loss import PricingLoss
from training import Trainer


def train_GAN():
    """Trains the GAN model."""
    ###############
    # Import data #
    ###############
    data_path = "../../../datasets"

    macro_path = f"{data_path}/macro/macro_train.npz"
    macro_zip = np.load(macro_path)

    firm_path = f"{data_path}/char/Char_train.npz"
    firm_zip = np.load(firm_path)

    valid_macro_path = f"{data_path}/macro/macro_valid.npz"
    valid_macro_zip = np.load(valid_macro_path)

    valid_firm_path = f"{data_path}/char/Char_valid.npz"
    valid_firm_zip = np.load(valid_firm_path)

    test_macro_path = f"{data_path}/macro/macro_test.npz"
    test_macro_zip = np.load(test_macro_path)

    test_firm_path = f"{data_path}/char/Char_test.npz"
    test_firm_zip = np.load(test_firm_path)

    ###################
    # Data processing #
    ###################
    # input in format: macro, firm, return, mask
    cleaner = Data()
    inputs = cleaner.clean(macro_zip["data"],
                           firm_zip["data"])
    valid_inputs = cleaner.clean(valid_macro_zip["data"],
                                 valid_firm_zip["data"])
    test_inputs = cleaner.clean(test_macro_zip["data"],
                                test_firm_zip["data"])

    #################
    # Create models #
    #################
    sdf_model = SDFModel()
    sdf_untrainined = sdf_model(inputs, training=True, verbose=True)

    conditional_model = ConditionalModel()
    conditional_model(inputs, training=True, verbose=True)

    ######################
    # Loss and Optimizer #
    ######################
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = PricingLoss()

    # compute starting sharpe
    starting_sharpe = loss.sharpe_loss(
        sdf_untrainined
    )
    print(f"Staring train SHARPE: {starting_sharpe}")

    ############
    # Training #
    ############
    trainer = Trainer(loss)
    trainer.train(
        sdf_model=sdf_model,
        conditional_model=conditional_model,
        optimizer=optimizer,
        inputs=inputs,
        valid_inputs=valid_inputs,
        sdf_epoches_unc=1000,  # author: 256, mine: 500
        moment_epoches=1000,  # author: 64, mine: 500
        epoches=5000,  # author: 1024, mine: 5000
        verbose_freq=100
    )

    ############
    # Test set #
    ############
    test_sdf = sdf_model(test_inputs)
    test_sharpe = loss.sharpe_loss(
        test_sdf
    )
    # print(f"Final test sharpe: {test_sharpe}")

    return sdf_model, conditional_model


if __name__ == "__main__":
    sdf_model, conditional_model = train_GAN()
    ##############
    # Save model #
    ##############
    sdf_model.save_weights("saved_models/sdf/")
    conditional_model.save_weights("saved_models/conditional/")
    print("Model weights saved")
