import tensorflow as tf
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

    def apply(self, inputs, outputs:list=[]):
        r"""
        Documentation here
        """
        # INPUT SCALING
        samples, timesteps, features = inputs.shape
        scaled_inputs = np.concatenate([
            self.input_scaler[feature](inputs[:, :, feature].reshape(-1, 1)).reshape((samples, timesteps, 1)) for feature in range(features)
        ], axis=2)
        
        # OUTPUT SCALING
        if outputs.any():
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
        
        # INITIALIZATION
        self.scaler = AcuNetScaler(scaler)
        layers_names = self.__create_layer_names(**kwargs)
        self.__check_arg_length(units, activations, layers_names)

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

    def __check_arg_length(self, *args):
        r"""
        Documentation here
        """
        flag_len = len(args[0])

        for arg in args:
            
            if len(arg) != flag_len:
                
                raise ValueError('Arguments must be the same length')

    def __create_layer_names(self, **kwargs):
        r"""
        Documentation here
        """
        layers_names = list()

        if 'layers_names' in kwargs:
            
            layers_names = kwargs['layers_names']
        
        else:
            
            for layer_num in range(len(units)):
                
                layers_names.append('AcuNet_Layer_{}'.format(layer_num))

        self.layers_names = layers_names

        return layers_names

    def __hidden_output_structure_definition(self):
        r"""
        Documentation here
        """
        self.output_layer_units = units.pop()
        self.output_layer_activation = activations.pop()
        self.output_layer_name = self.layers_names.pop()
        self.hidden_layers_units = units
        self.hidden_layers_activations = activations
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
