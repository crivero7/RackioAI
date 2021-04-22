import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class AcuNetLSTMCell(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(AcuNetLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.acunet_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        r"""
        Documentation here
        """
        outputs = self.acunet_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)

        return norm_outputs


class AcuNetScaler:
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


class AcuNet(tf.keras.Model):
    r"""
    Documentation here
    """

    def __init__(
        self,
        units, 
        activations,
        compile_options,
        scaler=None, 
        **kwargs
        ):

        super(AcuNet, self).__init__(**kwargs)
        self.units = units
        
        # INITIALIZATION
        self.scaler = AcuNetScaler(scaler)
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
        
        self.compile(**compile_options)

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

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=3,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=1e-6,
                mode='min')
            ],
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        plot=False,
        data_section='validation'
        ):
        r"""
        Documentation here
        """
        self._validation_data = validation_data
        self._train_data = (x, y)

        if self.scaler:
            x_test, y_test = validation_data
            x, y = self.scaler.apply(x, outputs=y)
            validation_data = self.scaler.apply(x_test, outputs=y_test)

        history = super(AcuNet, self).fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )

        if plot:

            self.__plot_prediction(data_section=data_section)

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
        
        y = super(AcuNet, self).predict(x, **kwargs)

        if self.scaler:

            y = self.scaler.inverse(y)[0]

        return y

    def __plot_prediction(self, data_section:str='validation'):
        r"""
        Documentation here
        """
        x, y = self._validation_data
        # MODEL PREDICTION
        if data_section.lower() == 'train':
            x, y = self._train_data
        
        y_predict = self.predict(x)

        # PLOT RESULT
        y = y.reshape(y.shape[0], y.shape[-1])
        y_predict = y_predict.reshape(y_predict.shape[0], y_predict.shape[-1])
        _result = np.concatenate((y_predict, y), axis=1)
        result = pd.DataFrame(_result, columns=['Prediction', '{}'.format(data_section).capitalize()])
        result.plot(kind='line')
        plt.show()
        
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
        initializer = tf.keras.initializers.GlorotUniform()
        for layer_num, units in enumerate(self.hidden_layers_units):
            if layer_num==len(self.hidden_layers_units) - 1:
                setattr(
                    self, 
                    self.hidden_layers_names[layer_num], 
                    AcuNetLSTMCell(units, self.hidden_layers_activations[layer_num], return_sequences=False)
                    )

            else:
                setattr(
                    self, 
                    self.hidden_layers_names[layer_num], 
                    AcuNetLSTMCell(units, self.hidden_layers_activations[layer_num], return_sequences=True)
                    )

    def __output_layer_definition(self):
        r"""
        Documentation here
        """
        initializer = tf.keras.initializers.GlorotUniform()
        setattr(
            self, 
            self.output_layer_name, 
            tf.keras.layers.Dense(self.output_layer_units, self.output_layer_activation)
            )
