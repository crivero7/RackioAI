import tensorflow as tf

class RackioGaussianNoise(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, stddev=1.0, **kwargs):
        r"""
        Documentation here
        """
        super(RackioGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        r"""
        Documentation here
        """
        if training:

            noise = tf.random.normal(tf.shape(X), self.stddev)

            return X + noise

        else:

            return X

    def compute_output_shape(self, input_shape):
        r"""
        Documentation here
        """
        return super(RackioGaussianNoise).compute_output_shape(input_shape)