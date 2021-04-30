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

    def call(self, inputs):
        r"""
        Documentation here
        """
        outputs = self.rackio_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)

        return norm_outputs
