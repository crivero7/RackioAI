from sklearn.model_selection import train_test_split as TTS
from rackio_AI.utils import Utils
import numpy as np
import pandas as pd


ONE_SPLIT = 1
TWO_SPLIT = 2

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
        >>> from rackio_AI import Preprocessing
        >>> preprocess = Preprocessing(name='Preprocess 1',description='preprocess for data', problem_type='regression')
        >>> print(preprocess.splitter)
        Splitter Object
        {'train_size': None, 'test_size': None, 'validation_size': None, 'random_state': None, 'shuffle': True, 'stratify': None}

        ```
        """
        self.default_options = {'train_size': None,
                                'test_size': None,
                                'validation_size': None,
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
        >>> from rackio_AI import  Preprocessing
        >>> import numpy as np
        >>> preprocess = Preprocessing(name='Preprocess 2',description='preprocess for data', problem_type='regression')
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
        """
        data = [array.values if isinstance(array, pd.DataFrame) else array for array in arrays]

        # Check default options
        default_options = {key: options[key] if key in list(options.keys()) else self.default_options[key] for key in list(self.default_options.keys())}

        # remove validation_size key to be used in train_test_split method from scikit-learn
        self.validation_size = default_options.pop('validation_size')

        # check if is necessary to do train-test-validation split
        lst = [default_options['train_size'], default_options['test_size'], self.validation_size]

        #if lst.count(None) <= 1 or (default_options['train_size'] + default_options['test_size'] < 1):
        if default_options['train_size'] and default_options['test_size'] and self.validation_size==None and \
                (default_options['train_size'] + default_options['test_size'] < 1) or \
                (default_options['train_size'] and default_options['test_size'] and self.validation_size):

            #default_options, _ = self._check_split_sizes(**default_options)

            return self._split(TWO_SPLIT, *data, **default_options)


        return self._split(ONE_SPLIT, *data, **default_options)


    def _split(self, flag, *data, **default_options):
        """
        splitter manager to do one_split or two_split
        :param flag:
        :param data:
        :param default_options:
        :return:
        """
        if flag==ONE_SPLIT:

            return self._one_split(*data, **default_options)

        elif flag==TWO_SPLIT:

            return self._two_splits(*data, **default_options)

        return

    def _one_split(self, *data, **default_options):
        """
        Split data in train and test datasets
        :return:
        """
        default_options['test_size'] = None

        return TTS(*data, **default_options)

    def _two_splits(self, *data, **default_options):
        """
        Split data in train-test and validation datasets
        """
        test_size = default_options['test_size']

        validation_size = 1 - default_options['train_size'] - test_size

        default_options['test_size'] = None

        # First split
        X_train, X_test, y_train, y_test = TTS(*data, **default_options)

        default_options['train_size'] = test_size / (test_size + validation_size)

        # Second split
        X_test, X_validation, y_test, y_validation = TTS(X_test, y_test, **default_options)

        return [X_train, X_test, X_validation, y_train, y_test, y_validation]

    def _check_split_sizes(self, **options):
        """
        Normalize proportion if their sum is not 1.0
        """
        train_size = options['train_size']
        if not train_size:

            train_size = 0.0

        test_size = options['test_size']
        if not test_size:

            test_size = 0.0

        validation_size = self.validation_size
        if not validation_size:

            validation_size = 0.0

        if train_size!=None and test_size!=None and validation_size !=None:

            options['train_size'] = train_size / (train_size + test_size + validation_size)
            options['test_size'] = test_size / (train_size + test_size + validation_size)
            validation_size = validation_size / (train_size + test_size + validation_size)

            return options, validation_size

    def __str__(self):
        """

        :return:
        """
        return "Splitter Object\n{}".format(self.default_options)


class LSTMDataPreparation:
    """
    Documentation here
    """

    def __init__(self):
        pass

    def split_sequences(
        self, 
        df: pd.DataFrame, 
        timesteps:int,
        stepsize: int= 1, 
        input_cols: list=None, 
        output_cols: list=None
        ):
        """
        **Parameters**

        * **:param df:** (Pandas.DataFrame)
        * **:param timesteps:** (int)
        * **:param input_cols:** (list)
        * **:param output_cols:** (list)

        **returns**

        **sequences** (tensor)

        ```python
        >>> import numpy as np
        >>> from rackio_AI import Preprocessing
        >>> a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]).reshape(-1,1)
        >>> b = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95]).reshape(-1,1)
        >>> c = np.array([a[i]+b[i] for i in range(len(a))]).reshape(-1,1)
        >>> data = np.hstack((a,b,c))
        >>> data
        array([[ 10,  15,  25],
               [ 20,  25,  45],
               [ 30,  35,  65],
               [ 40,  45,  85],
               [ 50,  55, 105],
               [ 60,  65, 125],
               [ 70,  75, 145],
               [ 80,  85, 165],
               [ 90,  95, 185]])
        >>> df = pd.DataFrame(data, columns=['a', 'b', 'c'])
        >>> preprocess = Preprocessing(name='LSTM Data Preparation', description='LSTM')
        >>> x, y = preprocess.lstm_data_preparation.split_sequences(df, 2)
        >>> x
        array([[[10., 15.],
                [20., 25.]],
        <BLANKLINE>
               [[20., 25.],
                [30., 35.]],
        <BLANKLINE>
               [[30., 35.],
                [40., 45.]],
        <BLANKLINE>
               [[40., 45.],
                [50., 55.]],
        <BLANKLINE>
               [[50., 55.],
                [60., 65.]],
        <BLANKLINE>
               [[60., 65.],
                [70., 75.]],
        <BLANKLINE>
               [[70., 75.],
                [80., 85.]],
        <BLANKLINE>
               [[80., 85.],
                [90., 95.]]])
        >>> y
        array([[[ 25.],
                [ 45.]],
        <BLANKLINE>
               [[ 45.],
                [ 65.]],
        <BLANKLINE>
               [[ 65.],
                [ 85.]],
        <BLANKLINE>
               [[ 85.],
                [105.]],
        <BLANKLINE>
               [[105.],
                [125.]],
        <BLANKLINE>
               [[125.],
                [145.]],
        <BLANKLINE>
               [[145.],
                [165.]],
        <BLANKLINE>
               [[165.],
                [185.]]])

        ```
        """
        if not input_cols:

            input_cols = Utils.get_column_names(df)
            input_cols = input_cols[:-1]
        
        if not output_cols:

            output_cols = Utils.get_column_names(df)
            output_cols = [output_cols[-1]]

        input_data = df.loc[:, input_cols]
        output_data = df.loc[:, output_cols]
        iteration = list(range(0, input_data.shape[0] - timesteps + stepsize, stepsize))

        x_sequences = np.zeros((len(iteration), timesteps, len(input_cols)))
        y_sequences = np.zeros((len(iteration), timesteps, len(output_cols)))
        
        for i in iteration:

            x_sequences[i] = input_data.values[i : i + stepsize + 1, :]
            y_sequences[i] = output_data.values[i : i + stepsize + 1, :]

        return x_sequences, y_sequences



if __name__=="__main__":
    import doctest
    doctest.testmod()
    