import tensorflow as tf
from rackio_AI.models.lstm_layer import RackioLSTMCell
from rackio_AI.models.scaler import RackioDNNScaler


class RackioClassification(tf.keras.Model):
    r"""
    Documentation here
    """

    def __init__(
        self,
        units, 
        activations,
        scaler=None,
        layers_names: list=[], 
        **kwargs
        ):

        super(RackioClassification, self).__init__(**kwargs)
        self.units = units
        self.activations = activations
        
        # INITIALIZATION
        self.scaler = RackioDNNScaler(scaler)
        self.layers_names = self.create_layer_names(units, layers_names=layers_names)
        
        if not self.check_arg_length(units, activations, self.layers_names):
            
            raise ValueError('units, activations and layer_names must be of the same length')

        # HIDDEN/OUTPUT STRUCTURE DEFINITION
        self.define_structure_hidden_output_layers()

        # LAYERS DEFINITION
        self.__define_hidden_layers()
        self.__define_output_layer()

    def call(self, inputs):
        r"""
        Documentation here
        """
        x = inputs

        # HIDDEN LAYER CALL
        for layer_num, units in enumerate(self.hidden_layers_units):
           
            classification_layer = getattr(self, self.hidden_layers_names[layer_num])
            x = classification_layer(x)

        # OUTPUT LAYER CALL
        classification_output_layer = getattr(self, self.output_layer_name)
        
        return classification_output_layer(x)

    def compile(
        self,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, 
            amsgrad=True
            ),
        loss='binary_crossentropy',
        metrics=tf.keras.metrics.BinaryAccuracy(),
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
        ):
        r"""
        Documentation here
        """
        super(RackioClassification, self).compile(
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
        **kwargs
        ):
        r"""
        Documentation here
        """
        history = super(RackioClassification, self).fit(
            x,
            y,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )

        return history

    def predict(
        self,
        x,
        **kwargs
        ):
        r"""
        Documentation here
        """
        y = super(RackioClassification, self).predict(x, **kwargs)

        return y

    def evaluate(
        self,
        x=None,
        y=None,
        **kwargs
        ):
        r"""
        Documentation here
        """        
        evaluation = super(RackioClassification, self).evaluate(x, y, **kwargs)

        return evaluation

    def plot(self, x, y, **kwargs):
        r"""
        Documentation here
        """
        return super(RackioClassification, self).predict(x, **kwargs)

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
                
                layers_names.append('LeakNet_Layer_{}'.format(layer_num))

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
