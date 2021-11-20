"""Model training for GAN model."""
import tensorflow as tf
import tensorflow.keras as keras


class Trainer(object):
    """Trains the model.

    Training procedure:
        1. Train unconditional SDF loss
        2. Train moment condition
        3. Train conditional SDF loss
    """

    def __init__(self, loss: keras.losses.Loss):
        """Init trainer."""
        self.Loss = loss

    def train_sdf(self,
                  optimizer: keras.optimizers.Optimizer,
                  sdf_model: keras.models.Model,
                  inputs: list,
                  conditional_model: keras.models.Model = None,
                  train_conditional: bool = False
                  ):
        """Take in model and data and conduct one training training."""
        macro_data, firm_data, return_data, mask = inputs

        with tf.GradientTape() as tape:
            sdf = sdf_model(
                inputs=inputs,
                training=True
            )
            if train_conditional:
                moment = conditional_model(inputs, training=False)
            else:
                moment = tf.ones(
                    shape=(1, firm_data.shape[0],
                           firm_data.shape[1])
                )
            loss = self.Loss(sdf=sdf,
                             moment=moment,
                             return_data=inputs[2],
                             mask=inputs[3],
                             verbose=False
                             )
        gradients = tape.gradient(loss, sdf_model.trainable_weights)
        optimizer.apply_gradients(zip(
            gradients, sdf_model.trainable_variables))

        return sdf, loss, gradients

    def train_moment(self,
                     optimizer: keras.optimizers.Optimizer,
                     conditional_model: keras.models.Model,
                     inputs: list,
                     sdf_model: keras.models.Model
                     ):
        """Conduct a single training for the moment."""
        macro_data, firm_data, return_data, mask = inputs

        with tf.GradientTape() as tape:
            moment = conditional_model(
                inputs,
                training=True
            )
            sdf = sdf_model(
                inputs=inputs,
                training=False
            )
            loss = self.Loss(sdf=sdf,
                             moment=moment,
                             return_data=inputs[2],
                             mask=inputs[3])
            # we want to maximise the loss with conditional moments
            loss = -loss
        gradients = tape.gradient(loss,
                                  conditional_model.trainable_weights)
        optimizer.apply_gradients(zip(
            gradients, conditional_model.trainable_variables))

        return loss, gradients

    def train(self,
              sdf_model: keras.models.Model,
              conditional_model: keras.models.Model,
              optimizer: keras.optimizers.Optimizer,
              inputs: list,
              valid_inputs: list,
              sdf_epoches_unc: int,
              moment_epoches: int,
              epoches: int,
              verbose_freq: int
              ):
        """Conduct training for the GAN model."""
        max_sharpe, max_valid_sharpe = 0, 0
        print("==============Start unconditional training===========""")
        # 1. Train unconditional SDF
        for jj in range(sdf_epoches_unc):
            # one epoch
            epoch_sdf, loss, gradient = self.train_sdf(
                optimizer=optimizer,
                sdf_model=sdf_model,
                inputs=inputs
            )
            sharpe_loss = self.Loss.sharpe_loss(
                epoch_sdf
            )
            valid_sharpe = self.Loss.sharpe_loss(
                sdf_model(valid_inputs)
            )
            if sharpe_loss > max_sharpe:
                max_sharpe = sharpe_loss
            if valid_sharpe > max_valid_sharpe:
                max_valid_sharpe = valid_sharpe
            if not jj % verbose_freq:
                print(f"""
                      unconditional SDF epochs {jj}
                      epoch loss = {loss}
                      epoch Sharpe = {sharpe_loss, valid_sharpe}
                      max Sharpe = {max_sharpe, max_valid_sharpe}
                      """)

        print("==============Start moments training===========""")
        # 2. Train conditional moments
        for jj in range(moment_epoches):
            # one epoch
            loss, gradient = self.train_moment(
                optimizer=optimizer,
                conditional_model=conditional_model,
                sdf_model=sdf_model,
                inputs=inputs
            )
            if not jj % verbose_freq:
                print(f"""
                      moments epochs {jj}
                      epoch loss = {loss}
                      """)

        print("==============Start GAN training===========""")
        for jj in range(epoches):
            # one epoch
            # Train moments
            moment_loss, gradient = self.train_moment(
                optimizer=optimizer,
                conditional_model=conditional_model,
                inputs=inputs,
                sdf_model=sdf_model
            )
            # Train SDF
            epoch_sdf, sdf_loss, gradient = self.train_sdf(
                optimizer=optimizer,
                sdf_model=sdf_model,
                inputs=inputs,
                conditional_model=conditional_model
            )
            # generate Sharpe loss
            sharpe_loss = self.Loss.sharpe_loss(
                epoch_sdf
            )
            valid_sharpe = self.Loss.sharpe_loss(
                sdf_model(valid_inputs)
            )
            if sharpe_loss > max_sharpe:
                max_sharpe = sharpe_loss
            if valid_sharpe > max_valid_sharpe:
                max_valid_sharpe = valid_sharpe
            if not jj % verbose_freq:
                print(f"""
                      epochs {jj}
                      epoch SDF loss = {sdf_loss}
                      epoch moment loss = {moment_loss}
                      epoch Sharpe = {sharpe_loss, valid_sharpe}
                      max Sharpe = {max_sharpe, max_valid_sharpe}
                      """)
