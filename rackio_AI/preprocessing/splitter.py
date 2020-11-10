from sklearn.model_selection import train_test_split as TTS
import numpy as np
import pandas as pd


class Splitter:
    """
    This is a *RackioAI* preprocessing class to split the data to create a Deep learning model
    """

    def __init__(self):
        """
        Splitter instantiation

        **Parameters**

        * None

        **:return:**

        * **Splitter:** (Splitter object)

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')
        >>> print(preprocess.splitter)
        Splitter Object
        {'train_size': 0.7, 'test_size': 0.3, 'validation_size': 0, 'random_state': None, 'shuffle': True, 'stratify': None}

        ```
        """
        self.default_options = {'train_size': 0.7,
                                'test_size': 0.3,
                                'validation_size': 0,
                                'random_state': None,
                                'shuffle': True,
                                'stratify': None}

    def split(self, *arrays, **options):
        """
        Split arrays or matrices into random train and test subsets

        **Parameters**

        * **:*arrays:** (sequence of indexables with the same length / shape[0]) Allowed inputs are lists, numpy arrays,
        scipy-sparse matrices or pandas DataFrame

        * **:train_size:** (float or int, default=None) If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value
        is automatically set to the complement of the test size.

        * **:test_size:** (float or int, default=None) If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the
        value is set to the complement of the train size. If *train_size* is also None, it will be set to 0.30.

        * **:validation_size:** (float or int, default=None) If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the validation split. If int, represents the absolute number of test samples. If None, the
        value is set to the complement of the train size and test size. If *train_size* is also None, it will be set to 0.0.

        * **:random_state:** (int or RandomState instance, default=None) Controls the suffling applied to the data before
        applying split. Pass an int for reproducible output across multiple function calls.
        See [Glosary](https://scikit-learn.org/stable/glossary.html#term-random-state)

        * **:shuffle:** (bool, default=True) Whether or not to shuffle the data before splitting. If shuffle=False then stratify
         must be None.

        * **:stratify:** (array-like, default=None) If not None, data is split in a stratified fashion, using this as the class labels.

        **:return:**

        * **splitting:** (list, length=3 * len(arrays)) list containing train-test split of inputs.

        ___

        ## Snippet code
        ```python
        >>> from rackio_AI import RackioAI, Preprocessing
        >>> from rackio import Rackio
        >>> import numpy as np
        >>> app = Rackio()
        >>> RackioAI(app)
        >>> preprocess = Preprocessing(name= 'Preprocess model name',description='preprocess for data', problem_type='regression')
        >>> X, y = np.arange(20).reshape((10, 2)), range(10)
        >>> X
        array([[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7],
               [ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15],
               [16, 17],
               [18, 19]])
        >>> list(y)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        ```
        ## Snippet code 2
        ```python
        >>> X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess.splitter.split(X, y, train_size=0.6, test_size=0.2, validation_size=0.2, random_state=0)
        >>> X_train
        array([[ 2,  3],
               [12, 13],
               [14, 15],
               [ 6,  7],
               [ 0,  1],
               [10, 11]])
        >>> X_test
        array([[16, 17],
               [ 4,  5]])
        >>> X_validation
        array([[ 8,  9],
               [18, 19]])
        >>> y_train
        [1, 6, 7, 3, 0, 5]
        >>> y_test
        [8, 2]
        >>> y_validation
        [4, 9]

        ```

        ## Snippet code 3
        ```python
        >>> X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess.splitter.split(X, y, train_size=0.6, test_size=0.2, random_state=0)
        >>> X_train
        array([[ 2,  3],
               [12, 13],
               [14, 15],
               [ 6,  7],
               [ 0,  1],
               [10, 11]])
        >>> X_test
        array([[16, 17],
               [ 4,  5]])
        >>> X_validation
        array([[ 8,  9],
               [18, 19]])
        >>> y_train
        [1, 6, 7, 3, 0, 5]
        >>> y_test
        [8, 2]
        >>> y_validation
        [4, 9]

        ```

        ## Snippet code 4
        ```python
        >>> X_train, X_test, y_train, y_test = preprocess.splitter.split(X, y, train_size=0.7, test_size=0.3, random_state=0)
        >>> X_train
        array([[18, 19],
               [ 2,  3],
               [12, 13],
               [14, 15],
               [ 6,  7],
               [ 0,  1],
               [10, 11]])
        >>> X_test
        array([[ 4,  5],
               [16, 17],
               [ 8,  9]])
        >>> y_train
        [9, 1, 6, 7, 3, 0, 5]
        >>> y_test
        [2, 8, 4]

        ```
        """

        data = [array.values if isinstance(array, pd.DataFrame) else array for array in arrays]

        self.options = {key: options[key] if key in list(options.keys()) else self.default_options[key] for key in list(self.default_options.keys())}

        self._check_split_sizes()

        validation_size = self.options.pop('validation_size')
        self.options['test_size'] += validation_size
        # First split
        X_train, X_test, y_train, y_test = TTS(*data, **self.options)
        if validation_size == 0:

            return X_train, X_test, y_train, y_test
        # Redefining split sizes
        self.options['test_size'] -= validation_size
        test_size = self.options['test_size']

        self.options['train_size'] = test_size / (test_size + validation_size)

        self.options['test_size'] = 1 - self.options['train_size']

        # Second split
        X_test, X_validation, y_test, y_validation= TTS(X_test, y_test, **self.options)

        return X_train, X_test, X_validation, y_train, y_test, y_validation

    def _split_sequence(self):
        """

        """
        pass

    def _check_split_sizes(self):
        """

        """
        train_size = self.options['train_size']
        test_size = self.options['test_size']
        validation_size = self.options['validation_size']

        if validation_size == 0 and train_size + test_size < 1:

            self.options['validation_size'] = 1 - train_size - test_size

            return True

        elif validation_size != 0 and validation_size + train_size + test_size != 1:

            self.options['train_size'] = train_size / (train_size + test_size + validation_size)
            self.options['test_size'] = test_size / (train_size + test_size + validation_size)
            self.options['validation_size'] = validation_size / (train_size + test_size + validation_size)

            return True



    def __str__(self):
        """

        :return:
        """
        return "Splitter Object\n{}".format(self.default_options)

if __name__=="__main__":
    import doctest
    doctest.testmod()