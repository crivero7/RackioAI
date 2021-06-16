import tensorflow as tf

class RackioLSTMCell(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.rackio_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        r"""
        Documentation here
        """
        self._input_shape = input_shape
        
        return super().build(input_shape)

    def call(self, inputs):
        r"""
        Documentation here
        """
        outputs = self.rackio_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)

        return norm_outputs

    def compute_output_shape(self, input_shape):
        
        return super().compute_output_shape(input_shape)

    def get_config(self):
        r"""
        Documentation here
        """
        
        base_config = super().get_config()
        
        return {
            **base_config, 
            "units": self.units,
            "input_shape": self._input_shape,
            "activation": self.activation
        }
