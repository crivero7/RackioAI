import tensorflow as tf
from rackio_AI.models.lstm_layer import RackioLSTMCell
from rackio_AI.models.scaler import RackioDNNLayerScaler, RackioDNNLayerInverseScaler
from rackio_AI.models.gaussian_noise import RackioGaussianNoise


class RackioLSTM(tf.keras.Model):
    r"""
    Documentation here
    """

    def __init__(
        self,
        units, 
        activations,
        min_max_values=None,
        layers_names: list=[], 
        add_gn: bool=True,
        **kwargs
        ):
        # INITIALIZATION
        super(RackioLSTM, self).__init__(**kwargs)
        self.units = units
        self.activations = activations
        self.scaler = None
        self.inverse_scaler = None
        self.add_gn = add_gn

        if self.add_gn:

            self.gaussian_noise = RackioGaussianNoise(stddev=10.0)

        if min_max_values:

            self.X_min, self.y_min, self.X_max, self.y_max = min_max_values
            self.scaler = RackioDNNLayerScaler(self.X_min, self.X_max)
            self.inverse_scaler = RackioDNNLayerInverseScaler(self.y_min, self.y_max)

        self.layers_names = self.create_layer_names(units, layers_names=layers_names)
        
        if not self.check_arg_length(units, activations, self.layers_names):

            raise ValueError('units, activations and layer_names must be of the same length')

        # HIDDEN/OUTPUT STRUCTURE DEFINITION
        self.define_structure_hidden_output_layers()

        # LAYERS DEFINITION
        self.__define_hidden_layers()
        self.__define_output_layer()

    def call(self, inputs, training=None):
        r"""
        Documentation here
        """
        x = inputs
        
        if self.add_gn:

            x = self.gaussian_noise(x, training=training)

        if self.scaler:
            
            x = self.scaler(x)

        # HIDDEN LAYER CALL
        for layer_num, _ in enumerate(self.hidden_layers_units):
           
            hidden_layer = getattr(self, self.hidden_layers_names[layer_num])
            x = hidden_layer(x)

        # OUTPUT LAYER CALL
        output_layer = getattr(self, self.output_layer_name)
        
        y = output_layer(x)

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
        super(RackioLSTM, self).compile(
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
        history = super(RackioLSTM, self).fit(
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
        return super(RackioLSTM, self).predict(x, **kwargs)

    def evaluate(
        self,
        x=None,
        y=None,
        **kwargs
        ):
        r"""
        Documentation here
        """        
        evaluation = super(RackioLSTM, self).evaluate(x, y, **kwargs)

        return evaluation

    def __define_hidden_layers(self):
        r"""
        Documentation here
        """
        
        for layer_num, units in enumerate(self.hidden_layers_units):
            
            if layer_num==len(self.hidden_layers_units) - 1:
                
                setattr(
                    self, 
                    self.hidden_layers_names[layer_num], 
                    RackioLSTMCell(units, self.hidden_layers_activations[layer_num], return_sequences=False)
                    )

            else:
                
                setattr(
                    self, 
                    self.hidden_layers_names[layer_num], 
                    RackioLSTMCell(units, self.hidden_layers_activations[layer_num], return_sequences=True)
                    )

    def __define_output_layer(self):
        r"""
        Documentation here
        """
        setattr(
            self, 
            self.output_layer_name, 
            tf.keras.layers.Dense(self.output_layer_units, self.output_layer_activation)
            )

    @staticmethod
    def check_arg_length(*args):
        r"""
        Documentation here
        """
        flag_len = len(args[0])

        for arg in args:
            
            if len(arg) != flag_len:
                
                return False
        
        return True

    @staticmethod
    def create_layer_names(units: list, layers_names: list=[]):
        r"""
        Documentation here
        """
        layers_names = list()

        if layers_names:
            
            layers_names = layers_names
        
        else:
            
            for layer_num in range(len(units)):
                
                layers_names.append('RackioDNN_Layer_{}'.format(layer_num))

        return layers_names

    def define_structure_hidden_output_layers(self):
        r"""
        Documentation here
        """
        self.output_layer_units = self.units.pop()
        self.output_layer_activation = self.activations.pop()
        self.output_layer_name = self.layers_names.pop()
        self.hidden_layers_units = self.units
        self.hidden_layers_activations = self.activations
        self.hidden_layers_names = self.layers_names
