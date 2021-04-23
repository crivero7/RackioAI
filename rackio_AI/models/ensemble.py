import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class RackioEnsembleLSTMCell(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioEnsembleLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.rackio_ensemble_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        r"""
        Documentation here
        """
        outputs = self.rackio_ensemble_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)

        return norm_outputs


class EnsembleScaler:
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


class RackioEnsemble(tf.keras.Model):
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

        super(RackioEnsemble, self).__init__(**kwargs)
        self.units = units
        
        # INITIALIZATION
        self.scaler = EnsembleScaler(scaler)
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
        Documentation here
        """
        super(RackioEnsemble, self).compile(
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

        history = super(RackioEnsemble, self).fit(
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
        
        y = super(RackioEnsemble, self).predict(x, **kwargs)

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
        evaluation = super(RackioEnsemble, self).evaluate(x, y, **kwargs)

        if plot_prediction:
            
            y_predict = super(RackioEnsemble, self).predict(x, **kwargs)

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