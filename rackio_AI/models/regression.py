import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class RackioRegressionLSTMCell(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioRegressionLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.rackio_regression_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        r"""
        Documentation here
        """
        outputs = self.rackio_regression_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)

        return norm_outputs


class RegressionScaler:
    r"""
    Documentation here
    """
    
    def __init__(self, scaler):
        r"""
        Documentation here
        """
        self.input_scaler = scaler['inputs']
        self.output_scaler = scaler['outputs']
    
    def apply(self, inputs, **kwargs):
        r"""
        Documentation here
        """
        # INPUT SCALING
        samples, timesteps, features = inputs.shape
        scaled_inputs = np.concatenate([
            self.input_scaler[feature](inputs[:, :, feature].reshape(-1, 1)).reshape((samples, timesteps, 1)) for feature in range(features)
        ], axis=2)
        
        # OUTPUT SCALING
        if 'outputs' in kwargs:
            outputs = kwargs['outputs']
            samples, timesteps, features = outputs.shape
            scaled_outputs = np.concatenate([
                self.output_scaler[feature](outputs[:, :, feature].reshape(-1, 1)).reshape((samples, timesteps, 1)) for feature in range(features)
            ], axis=2)

            return scaled_inputs, scaled_outputs
        
        return scaled_inputs

    def inverse(self, *outputs):
        r"""
        Documentation here
        """
        result = list()
        
        for output in outputs:
            
            features = output.shape[-1]
            samples = output.shape[0]
            # INVERSE APPLY
            scaled_output = np.concatenate([
                self.output_scaler[feature].inverse(output[:, feature].reshape(-1, 1)).reshape((samples, features, 1)) for feature in range(features)
            ], axis=2)

            result.append(scaled_output)
       
        return tuple(result)


class RackioRegression(tf.keras.Model):
    r"""
    Documentation here
    """

    def __init__(
        self,
        units, 
        activations,
        scaler=None, 
        **kwargs
        ):

        super(RackioRegression, self).__init__(**kwargs)
        self.units = units
        
        # INITIALIZATION
        self.scaler = RegressionScaler(scaler)
        layers_names = self.__create_layer_names(**kwargs)
        if not self.__check_arg_length(units, activations, layers_names):
            raise ValueError('units, activations and layer_names must be of the same length')

        self.activations = activations
        self.layers_names = layers_names

        # HIDDEN/OUTPUT STRUCTURE DEFINITION
        self.__hidden_output_structure_definition()

        # LAYERS DEFINITION
        self.__hidden_layers_definition()
        self.__output_layer_definition()

    def call(self, inputs):
        r"""
        Documentation here
        """
        x = inputs

        # HIDDEN LAYER CALL
        for layer_num, units in enumerate(self.hidden_layers_units):
           
            acunet_layer = getattr(self, self.hidden_layers_names[layer_num])
            x = acunet_layer(x)

        # OUTPUT LAYER CALL
        acunet_output_layer = getattr(self, self.output_layer_name)
        
        return acunet_output_layer(x)

    def compile(
        self,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, 
            amsgrad=True
            ),
        loss='mse',
        metrics=tf.keras.metrics.MeanAbsoluteError(),
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
        ):
        r"""
        Configures the model for training.

        **Parameters**

        * **:param optimizer:** String (name of optimizer) or optimizer instance.
            * **[tf.keras.optimizers.Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)**: 
            Optimizer that implements the Adam algorithm.
            * **[tf.keras.optimizers.Adadelta](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adadelta)**: 
            Optimizer that implements the Adadelta algorithm.
            * **[tf.keras.optimizers.Adagrad](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad)**: 
            Optimizer that implements the Adagrad algorithm.
            * **[tf.keras.optimizers.Adamax](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax)**: 
            Optimizer that implements the Adamax algorithm.
            * **[tf.keras.optimizers.Ftrl](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl)**: 
            Optimizer that implements the FTRL algorithm.
            * **[tf.keras.optimizers.Nadam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam)**: 
            Optimizer that implements the Nadam algorithm.
            * **[tf.keras.optimizers.RMSprop](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)**: 
            Optimizer that implements the RMSprop algorithm.
            * **[tf.keras.optimizers.SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)**: 
            Optimizer that implements the SGD algorithm.
        * **:param loss:** String (name of objective function), objective function or tf.keras.losses.Loss 
        instance. See [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses). 
        An objective function is any callable with the signature loss = fn(y_true, y_pred), 
        where y_true = ground truth values with shape = [batch_size, d0, .. dN], except sparse loss 
        functions such as sparse categorical crossentropy where shape = [batch_size, d0, .. dN-1]. 
        y_pred = predicted values with shape = [batch_size, d0, .. dN]. It returns a weighted loss float tensor. 
        If a custom Loss instance is used and reduction is set to NONE, return value has the shape [batch_size, d0, .. dN-1] 
        ie. per-sample or per-timestep loss values; otherwise, it is a scalar. If the model has multiple outputs, 
        you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that 
        will be minimized by the model will then be the sum of all individual losses.

            ## Classes

            * **[tf.keras.losses.BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)** 
            Computes the cross-entropy loss between true labels and predicted labels.
            * **[tf.keras.losses.CategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)**
            Computes the crossentropy loss between the labels and predictions.
            * **[tf.keras.losses.CategoricalHinge](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalHinge)** 
            Computes the categorical hinge loss between y_true and y_pred.
            * **[tf.keras.losses.CosineSimilarity](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity)** 
            Computes the cosine similarity between labels and predictions.
            * **[tf.keras.losses.Hinge](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Hinge)** 
            Computes the hinge loss between y_true and y_pred.
            * **[tf.keras.losses.Huber](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber)** 
            Computes the Huber loss between y_true and y_pred.
            * **[tf.keras.losses.KLDivergence](https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLDivergence)** 
            Computes Kullback-Leibler divergence loss between y_true and y_pred.
            * **[tf.keras.losses.LogCosh](https://www.tensorflow.org/api_docs/python/tf/keras/losses/LogCosh)** 
            Computes the logarithm of the hyperbolic cosine of the prediction error.
            * **[tf.keras.losses.MeanAbsoluteError](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError)** 
            Computes the mean of absolute difference between labels and predictions.
            * **[tf.keras.losses.MeanAbsolutePercentageError](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsolutePercentageError)** 
            Computes the mean absolute percentage error between y_true and y_pred.
            * **[tf.keras.losses.MeanSquaredError](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError)**
             Computes the mean of squares of errors between labels and predictions.
            * **[tf.keras.losses.MeanSquaredLogarithmicError](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError)** 
            Computes the mean squared logarithmic error between y_true and y_pred.
            * **[tf.keras.losses.Poisson](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson)**
            Computes the Poisson loss between y_true and y_pred.
            * **[tf.keras.losses.Reduction](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction)** 
            Types of loss reduction.
            * **[tf.keras.losses.SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)** 
            Computes the crossentropy loss between the labels and predictions.
            * **[tf.keras.losses.SquaredHinge](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SquaredHinge)** 
            Computes the squared hinge loss between y_true and y_pred.

            ## Functions

            * **[KLD(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/KLD):** 
            Computes Kullback-Leibler divergence loss between y_true and y_pred.
            * **[MAE(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MAE):** 
            Computes the mean absolute error between labels and predictions.
            * **[MAPE(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MAPE):** 
            Computes the mean absolute percentage error between y_true and y_pred.
            * **[MSE(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MSE):** 
            Computes the mean squared error between labels and predictions.
            * **[MSLE(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MSLE):** 
            Computes the mean squared logarithmic error between y_true and y_pred.
            * **[binary_crossentropy(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/binary_crossentropy):** 
            Computes the binary crossentropy loss.
            * **[categorical_crossentropy(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy):** 
            Computes the categorical crossentropy loss.
            * **[categorical_hinge(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_hinge):** 
            Computes the categorical hinge loss between y_true and y_pred.
            * **[cosine_similarity(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/cosine_similarity):** 
            Computes the cosine similarity between labels and predictions.
            * **[deserialize(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/deserialize):** 
            Deserializes a serialized loss class/function instance.
            * **[get(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/get):** 
            Retrieves a Keras loss as a function/Loss class instance.
            * **[hinge(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/hinge):** 
            Computes the hinge loss between y_true and y_pred.
            * **[huber(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/huber):** 
            Computes Huber loss value.
            * **[kl_divergence(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/kl_divergence):** 
            Computes Kullback-Leibler divergence loss between y_true and y_pred.
            * **[kld(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/kld):** 
            Computes Kullback-Leibler divergence loss between y_true and y_pred.
            * **[kullback_leibler_divergence(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/kullback_leibler_divergence):** 
            Computes Kullback-Leibler divergence loss between y_true and y_pred.
            * **[log_cosh(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/log_cosh):** 
            Logarithm of the hyperbolic cosine of the prediction error.
            * **[logcosh(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/logcosh):** 
            Logarithm of the hyperbolic cosine of the prediction error.
            * **[mae(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mae):** 
            Computes the mean absolute error between labels and predictions.
            * **[mape(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mape):** 
            Computes the mean absolute percentage error between y_true and y_pred.
            * **[mean_absolute_error(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mean_absolute_error):** 
            Computes the mean absolute error between labels and predictions.
            * **[mean_absolute_percentage_error(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mean_absolute_percentage_error):** 
            Computes the mean absolute percentage error between y_true and y_pred.
            * **[mean_squared_error(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mean_squared_error):** 
            Computes the mean squared error between labels and predictions.
            * **[mean_squared_logarithmic_error(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mean_squared_logarithmic_error):** 
            Computes the mean squared logarithmic error between y_true and y_pred.
            * **[mse(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/mse):** 
            Computes the mean squared error between labels and predictions.
            * **[msle(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/msle):** 
            Computes the mean squared logarithmic error between y_true and y_pred.
            * **[poisson(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/poisson):** 
            Computes the Poisson loss between y_true and y_pred.
            * **[serialize(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/serialize):** 
            Serializes loss function or Loss instance.
            * **[sparse_categorical_crossentropy(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy):** 
            Computes the sparse categorical crossentropy loss.
            * **[squared_hinge(...)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/squared_hinge):** 
            Computes the squared hinge loss between y_true and y_pred.
        """
        super(RackioRegression, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs
        )

    def fit(
        self,
        x=None,
        y=None,
        validation_data=None,
        epochs=3,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=1e-6,
                mode='min')
            ],
        plot=False,
        data_section='validation',
        **kwargs
        ):
        r"""
        Documentation here
        """
        self._train_data = (x, y)
        self._validation_data = validation_data

        if self.scaler:
            x_test, y_test = validation_data
            x, y = self.scaler.apply(x, outputs=y)
            validation_data = self.scaler.apply(x_test, outputs=y_test)

        history = super(RackioRegression, self).fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )

        if plot:

            if data_section.lower()=='validation':
                
                x, y = validation_data
            
            self.evaluate(x, y, plot_prediction=True)

        return history

    def predict(
        self,
        x,
        **kwargs
        ):
        r"""
        Documentation here
        """
        if self.scaler:
            
            x = self.scaler.apply(x)
        
        y = super(RackioRegression, self).predict(x, **kwargs)

        if self.scaler:

            y = self.scaler.inverse(y)[0]

        return y

    def evaluate(
        self,
        x=None,
        y=None,
        plot_prediction=False,
        **kwargs
        ):
        r"""
        Documentation here
        """        
        evaluation = super(RackioRegression, self).evaluate(x, y, **kwargs)

        if plot_prediction:
            
            y_predict = super(RackioRegression, self).predict(x, **kwargs)

            if self.scaler:

                y_predict = self.scaler.inverse(y_predict)[0]
                y = self.scaler.inverse(y)[0]
            
            # PLOT RESULT
            y = y.reshape(y.shape[0], y.shape[-1])
            y_predict = y_predict.reshape(y_predict.shape[0], y_predict.shape[-1])
            _result = np.concatenate((y_predict, y), axis=1)
            result = pd.DataFrame(_result, columns=['Prediction', 'Original'])
            result.plot(kind='line')
            plt.show()

        return evaluation
    
    def __check_arg_length(self, *args):
        r"""
        Documentation here
        """
        flag_len = len(args[0])

        for arg in args:
            
            if len(arg) != flag_len:
                
                return False
        
        return True

    def __create_layer_names(self, **kwargs):
        r"""
        Documentation here
        """
        layers_names = list()

        if 'layers_names' in kwargs:
            
            layers_names = kwargs['layers_names']
        
        else:
            
            for layer_num in range(len(self.units)):
                
                layers_names.append('AcuNet_Layer_{}'.format(layer_num))

        self.layers_names = layers_names

        return layers_names

    def __hidden_output_structure_definition(self):
        r"""
        Documentation here
        """
        self.output_layer_units = self.units.pop()
        self.output_layer_activation = self.activations.pop()
        self.output_layer_name = self.layers_names.pop()
        self.hidden_layers_units = self.units
        self.hidden_layers_activations = self.activations
        self.hidden_layers_names = self.layers_names

    def __hidden_layers_definition(self):
        r"""
        Documentation here
        """
        for layer_num, units in enumerate(self.hidden_layers_units):
            if layer_num==len(self.hidden_layers_units) - 1:
                setattr(
                    self, 
                    self.hidden_layers_names[layer_num], 
                    RackioRegressionLSTMCell(units, self.hidden_layers_activations[layer_num], return_sequences=False)
                    )

            else:
                setattr(
                    self, 
                    self.hidden_layers_names[layer_num], 
                    RackioRegressionLSTMCell(units, self.hidden_layers_activations[layer_num], return_sequences=True)
                    )

    def __output_layer_definition(self):
        r"""
        Documentation here
        """
        setattr(
            self, 
            self.output_layer_name, 
            tf.keras.layers.Dense(self.output_layer_units, self.output_layer_activation)
            )