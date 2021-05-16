from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

@tf.function
def scaler(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        self = args[0]
        x = args[1]
        
        if self.scaler:
            
            x = self.scaler.apply(x)

        y = f(*args, **kwargs)

        if self.scaler:

            y = self.scaler.inverse(y)[0]

        return y

    return decorated

def fit_scaler(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        self = args[0]
        x = args[1]
        y = args[2]
        validation_data = kwargs['validation_data']
        
        if self.scaler:
            x_test, y_test = validation_data
            x, y = self.scaler.apply(x, outputs=y)
            kwargs['validation_data'] = self.scaler.apply(x_test, outputs=y_test)

        y = f(self, x, y, **kwargs)

        return y

    return decorated

def plot_scaler(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        self = args[0]
        x = args[1]
        y = args[2]
        
        if self.scaler:

            x = self.scaler.apply(x)

        y_predict = f(self, x, y, **kwargs)

        if self.scaler:

            y_predict = self.scaler.inverse(y_predict)[0]

        # PLOT RESULT
        y = y.reshape(y.shape[0], y.shape[-1])
        y_predict = y_predict.reshape(y_predict.shape[0], y_predict.shape[-1])
        _result = np.concatenate((y_predict, y), axis=1)
        result = pd.DataFrame(_result, columns=['Prediction', 'Original'])
        result.plot(kind='line')
        plt.show()

        return y_predict

    return decorated

def plot_confussion_matrix(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        self = args[0]
        x = args[1]
        y = args[2]
        
        if self.scaler:

            x = self.scaler.apply(x)

        y_predict = f(self, x, y, **kwargs)

        if self.scaler:

            y_predict = self.scaler.inverse(y_predict)[0]

        # PLOT RESULT
        y = y.reshape(y.shape[0], y.shape[-1])
        y_predict = y_predict.reshape(y_predict.shape[0], y_predict.shape[-1])
        cm = confusion_matrix(y,y_predict)
        print('Confusion matrix')
        print(cm)


        return y_predict

    return decorated