import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from rackio_AI.models.scaler import RackioDNNLayerScaler, RackioDNNLayerInverseScaler
from rackio_AI.models.gaussian_noise import RackioGaussianNoise


class RackioObserverDense(tf.keras.layers.Layer):
    r"""
    Documentation here
    """

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):
        r"""
        Documentation here
        """
        self.kernel = self.add_weight(
            name="kernel",
            shape=[batch_input_shape[-1], self.units],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
            trainable=True
        )

        super().build(batch_input_shape)

    def call(self, X):
        r"""
        Documentation here
        """

        return self.activation(X @ self.kernel + self.bias)

    # def compute_output_shape(self, batch_input_shape):
    #     r"""
    #     Documentation here
    #     """

    #     return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        r"""
        Documentation here
        """
        base_config = super().get_config()

        return {
            **base_config, 
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation)
        }


class RackioObserverLSTM_f(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioObserverLSTM_f, self).__init__(**kwargs)
        self.units = units
        self.rackio_observer_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.F = tf.Variable(np.zeros((1, 1)), dtype=tf.dtypes.float32)
        self.rackio_dense_layer = RackioObserverDense(1, activation="tanh")
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        r"""
        Documentation here
        """
        self.batch_size = 32
        
        if input_shape[0]:
            
            self.batch_size = input_shape[0]

        super(RackioObserverLSTM_f, self).build(input_shape)

    def call(self, inputs):
        r"""
        Documentation here
        """
        # Firt LSTM_f layer
        outputs = self.rackio_observer_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)
        # Dense layer
        f_t = self.rackio_dense_layer(norm_outputs)
        f = tf.reshape(f_t, (f_t.shape[1], f_t.shape[2]))
        
        # f Jacobian
        rows_f, cols_f = f.shape

        # Si la salida de la LSTM_f es una sola variable se calcula gradiente y no el jacobiano
        for i in range(cols_f):

            for j in range(cols_f):
                y_t = f[0, i]
                y_t_1 = f[1, i]
                x_k_1 = f[1, j]
                x_k_2 = f[2, j]

                self.F[i, j].assign((y_t - y_t_1) / (x_k_1 - x_k_2))

        return f_t, f, self.F


class RackioObserverLSTM_Q(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioObserverLSTM_Q, self).__init__(**kwargs)
        self.units = units
        self.rackio_observer_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.rackio_dense_layer = RackioObserverDense(1, activation="tanh")
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        r"""
        Documentation here
        """
        self.batch_size = 32
        
        if input_shape[0]:
            
            self.batch_size = input_shape[0]

        super(RackioObserverLSTM_Q, self).build(input_shape)

    def call(self, inputs):
        r"""
        Documentation here
        """
        # Firt LSTM_f layer
        outputs = self.rackio_observer_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)
        # Dense layer
        q_t = self.rackio_dense_layer(norm_outputs)
        q = tf.reshape(q_t, (q_t.shape[1], q_t.shape[2]))
        Q = tfp.stats.covariance(q, event_axis=1)

        return Q


class RackioObserverLSTM_R(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioObserverLSTM_R, self).__init__(**kwargs)
        self.units = units
        self.rackio_observer_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.rackio_dense_layer = RackioObserverDense(2, activation="tanh")
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        r"""
        Documentation here
        """
        self.batch_size = 32
        
        if input_shape[0]:
            
            self.batch_size = input_shape[0]

        super(RackioObserverLSTM_R, self).build(input_shape)

    def call(self, inputs):
        r"""
        Documentation here
        """
        # Firt LSTM_f layer
        outputs = self.rackio_observer_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)
        # Dense layer
        r_t = self.rackio_dense_layer(norm_outputs)
        r = tf.reshape(r_t, (r_t.shape[1], r_t.shape[2]))
        R = tfp.stats.covariance(r, event_axis=1)

        return R


class RackioObserverLSTM_H(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, units, activation='tanh', return_sequences=False, **kwargs):
        r"""
        Documentation here
        """
        super(RackioObserverLSTM_H, self).__init__(**kwargs)
        self.units = units
        self.rackio_observer_lstm_cell = tf.keras.layers.LSTM(units, activation=None, return_sequences=return_sequences, **kwargs)
        self.rackio_dense_layer = RackioObserverDense(2, activation="tanh")
        self.H = tf.Variable(np.zeros((2, 1)), dtype=tf.dtypes.float32)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        r"""
        Documentation here
        """
        self.batch_size = 32
        
        if input_shape[0]:
            
            self.batch_size = input_shape[0]

        super(RackioObserverLSTM_H, self).build(input_shape)

    def call(self, inputs, f):
        r"""
        Documentation here
        """
        # Firt LSTM_f layer
        outputs = self.rackio_observer_lstm_cell(inputs)
        norm_outputs = self.activation(outputs)
        # Dense layer
        h = self.rackio_dense_layer(norm_outputs)

        # LSTM_H computation
        rows_f, cols_f = f.shape
        h_c = tf.reshape(h, (h.shape[1], h.shape[2]))
        x = f
        rows_h, cols_h = h_c.shape

        for i in range(cols_h):

            for j in range(cols_f):
                h_t = h_c[0, i]
                h_t_1 = h_c[1, i]
                x_k_1 = x[1, j]
                x_k_2 = x[2, j]
                self.H[i, j].assign((h_t - h_t_1) / (x_k_1 - x_k_2))

        h = h_c[0, :]
        h = tf.reshape(h, (h.shape[0], 1))

        return h, self.H


class RackioKF(tf.keras.layers.Layer):
    r"""
    Documentation here
    """
    def __init__(self, **kwargs):
        r"""
        Documentation here
        """
        super(RackioKF, self).__init__(**kwargs)
        # y_t_corregido initialization
        y_t_corregido = np.zeros((1, 1))
        self.y_t_corregido = tf.Variable(y_t_corregido, dtype=tf.dtypes.float32)
        # P_t initialization
        p = np.zeros((1, 1))
        self.P_t = tf.Variable(p, dtype=tf.dtypes.float32)
        self.I = tf.eye(1, 1)

    def build(self, input_shape):
        r"""
        Documentation here
        """
        self.batch_size = 32

        if input_shape[0]:
            
            self.batch_size = input_shape[0]

        super(RackioKF, self).build(input_shape)

    def call(self, z, f, F, h, H, Q, R):
        r"""
        Documentation here
        """
        F_t = tf.transpose(F)
        H_t = tf.transpose(H)
        # Prediction Step (Kalman Intro / Coskun)
        P_t = tf.add( tf.matmul( tf.matmul(F, self.P_t), F_t), Q)

        # Correction Step
        K_t = tf.matmul( tf.matmul(P_t, H_t), tf.linalg.inv( tf.add( tf.matmul( tf.matmul(H, P_t), H_t), R)))
        self.y_t_corregido.assign(tf.add( f[0, :], tf.matmul( K_t, tf.subtract(z, h))))
        self.P_t.assign(tf.matmul(tf.subtract(self.I, tf.matmul(K_t, H)), P_t))
        
        return self.y_t_corregido


class RackioObserver(tf.keras.Model):
    r"""
    Documentation here
    """

    def __init__(
        self,
        units, 
        activations,
        min_max_values=None,
        add_gn: bool=True,
        **kwargs
        ):
        # INITIALIZATION
        super(RackioObserver, self).__init__(**kwargs)
        self.lstm_f = RackioObserverLSTM_f(units[0], activation=activations[0], return_sequences=True)
        self.lstm_H = RackioObserverLSTM_H(units[3], activation=activations[3], return_sequences=True)
        self.lstm_Q = RackioObserverLSTM_Q(units[1], activation=activations[1], return_sequences=True)
        self.lstm_R = RackioObserverLSTM_R(units[2], activation=activations[2], return_sequences=True)
        self.KF = RackioKF(**kwargs)
        self.scaler = None
        self.add_gn = add_gn

        if self.add_gn:

            self.gaussian_noise = RackioGaussianNoise()

        if min_max_values:

            self.X_min, self.y_min, self.X_max, self.y_max = min_max_values
            self.scaler = RackioDNNLayerScaler(self.X_min, self.X_max)
            self.inverse_scaler = RackioDNNLayerInverseScaler(self.y_min, self.y_max)
 
    def call(self, inputs):
        r"""
        **Parameters**

        * *:param u:* (Input tensor) Inlet / Outlet Pressure
        * *:param z:* (Input tensor) Inlet / Outlet Flow

        """
        
        if self.add_gn:

            inputs = self.gaussian_noise(inputs)

        if self.scaler:
            
            inputs = self.scaler(inputs)

        u = inputs[:, :, 0:2]
        z = inputs[:, :, 2:]
        f_t, f, F = self.lstm_f(u)
        h, H = self.lstm_H(f_t, f)
        Q = self.lstm_Q(f_t)
        R = self.lstm_R(z)
        z = tf.reshape(z, (z.shape[1], z.shape[2]))[0, :]
        z = tf.reshape(z, (z.shape[0], 1))

        # Observer layer based on Kalman Filter
        y = self.KF(z, f, F, h, H, Q, R)

        if self.inverse_scaler:

            y = self.inverse_scaler(y)

        return y

    def compile(
        self,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01, 
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

        * **:param metrics:** List of metrics to be evaluated by the model during training and testing. 
        Each of this can be a string (name of a built-in function), function or a 
        [tf.keras.metrics.Metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric) instance. 
        See [tf.keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics). 
        Typically you will use *metrics=['accuracy']*. A function is any callable with the signature 
        result = fn(y_true, y_pred). To specify different metrics for different outputs of a multi-output model, 
        you could also pass a dictionary, such as *metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}*.
        You can also pass a list *(len = len(outputs))* of lists of metrics such as 
        *metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]*. 
        When you pass the strings 'accuracy' or 'acc', we convert this to one of 
        [tf.keras.metrics.BinaryAccuracy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy), 
        [tf.keras.metrics.CategoricalAccuracy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy), 
        [tf.keras.metrics.SparseCategoricalAccuracy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy)
        based on the loss function used and the model output shape. We do a similar conversion for the strings 
        'crossentropy' and 'ce' as well.
        * **:param loss_weights:** Optional list or dictionary specifying scalar coefficients (Python floats) 
        to weight the loss contributions of different model outputs. The loss value that will be minimized 
        by the model will then be the weighted sum of all individual losses, weighted by the loss_weights coefficients. 
        If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected 
        to map output names (strings) to scalar coefficients.
        * **:param weighted_metrics:** List of metrics to be evaluated and weighted by sample_weight or class_weight 
        during training and testing.
        * **:param run_eagerly:** Bool. Defaults to *False*. If *True*, this Model's logic will not be wrapped in a
        [tf.function](https://www.tensorflow.org/api_docs/python/tf/function). Recommended to leave this as None 
        unless your Model cannot be run inside a *tf.function*.
        * **:param steps_per_execution:** Int. Defaults to 1. The number of batches to run during each tf.function call. 
        Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small 
        models with a large Python overhead. At most, one full epoch will be run each execution. 
        If a number larger than the size of the epoch is passed, the execution will be truncated to the size of the epoch. 
        Note that if steps_per_execution is set to N, 
        [Callback.on_batch_begin and Callback.on_batch_end](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
        methods will only be called every N batches (i.e. before/after each tf.function execution).
        * **:param kwargs:** Arguments supported for backwards compatibility only.

        **Raise**

        * **ValueError:** In case of invalid arguments for *optimizer*, *loss* or *metrics*.
        """
        super(RackioObserver, self).compile(
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
        *training_data,
        validation_data=None,
        epochs=3,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                min_delta=1e-6,
                mode='min')
            ],
        **kwargs
        ):
        r"""
        Trains the model for a fixed number of epochs (iterations on a dataset).

        **Parameters**

        * **:param x:** Input data. It could be:
            * A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            * A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            * A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
            * A [tf.data](https://www.tensorflow.org/guide/data) dataset. Should return a tuple of either 
            (inputs, targets) or (inputs, targets, sample_weights).
            * A generator or [tf.keras.utils.Sequence](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) 
            returning (inputs, targets) or (inputs, targets, sample_weights). 
            A more detailed description of unpacking behavior for iterator types (Dataset, generator, Sequence) 
            is given below.
        * **:param y:** Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). 
        It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely). 
        If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified 
        (since targets will be obtained from x).
        """
        x, y = training_data

        history = super(RackioObserver, self).fit(
            x,
            y,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )

        return history

    def predict(self, x, **kwargs):
        r"""
        Documentation here
        """
        return super(RackioObserver, self).predict(x, **kwargs)

    def evaluate(
        self,
        x=None,
        y=None,
        **kwargs
        ):
        r"""
        Documentation here
        """        
        evaluation = super(RackioObserver, self).evaluate(x, y, **kwargs)

        return evaluation