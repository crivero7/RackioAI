import numpy as np
import tensorflow as tf

class RackioDNNScaler:
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
        _inputs_list = list()

        for feature in range(features):

            _inputs = tf.reshape(inputs[:, :, feature], (-1, 1))
            
            _inputs = self.input_scaler[feature](_inputs)
            print(_inputs)
            _inputs = tf.reshape(_inputs, (samples, timesteps, 1))
            
            _inputs_list.append(_inputs)

        scaled_inputs = tf.concat(_inputs_list, axis=2)

        # scaled_inputs = np.concatenate([
        #     self.input_scaler[feature](inputs[:, :, feature].reshape(-1, 1)).reshape((samples, timesteps, 1)) for feature in range(features)
        # ], axis=2)
        
        # OUTPUT SCALING
        if 'outputs' in kwargs:
            outputs = kwargs['outputs']
            samples, timesteps, features = outputs.shape
            _outputs_list = list()
            for feature in range(features):
                _outputs = tf.reshape(outputs[:, :, feature], (-1, 1))
                _outputs = self.output_scaler[feature](_outputs)
                _outputs = tf.reshape(_outputs, (samples, timesteps, 1))
                _outputs_list.append(_outputs)
            scaled_outputs = tf.concat(_outputs_list, axis=2)
            # scaled_outputs = np.concatenate([
            #     self.output_scaler[feature](outputs[:, :, feature].reshape(-1, 1)).reshape((samples, timesteps, 1)) for feature in range(features)
            # ], axis=2)

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


class RackioDNNLayerScaler(tf.keras.layers.Layer):
    r"""
    Documentation here
    """

    def __init__(self, X_min, X_max, **kwargs):
        r"""
        Documentation here
        """
        super().__init__(**kwargs)
        self.X_min = X_min
        self.X_max = X_max

    def call(self, X):
        r"""
        Documentation here
        """
        X = (X - self.X_min) / (self.X_max - self.X_min)
        return X
    
    def get_config(self):
        r"""
        Documentation here
        """
        base_config = super().get_config()
        return {**base_config, "X_max": self.X_max, "X_min": self.X_min}


class RackioDNNLayerInverseScaler(tf.keras.layers.Layer):
    r"""
    Documentation here
    """

    def __init__(self,  y_min, y_max, **kwargs):
        r"""
        Documentation here
        """
        super().__init__(**kwargs)
        self.y_min = y_min
        self.y_max = y_max 

    def call(self, y):
        r"""
        Documentation here
        """
        y = y * (self.y_max - self.y_min) + self.y_min
        return y

    def get_config(self):
        r"""
        Documentation here
        """
        base_config = super().get_config()
        return {**base_config, "y_max": self.y_max, "y_min": self.y_min}
