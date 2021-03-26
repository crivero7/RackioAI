from sklearn.model_selection import train_test_split as TTS
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad_seq
from rackio_AI.utils.utils_core import Utils
from easy_deco.progress_bar import ProgressBar
import numpy as np
import pandas as pd
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


TRAIN_TEST_SPLIT = 1
TRAIN_TEST_VALIDATION_SPLIT = 2

@set_to_methods(del_temp_attr)
class RackioAISplitter:
    """
    This is a *RackioAI* preprocessing class to split the data to create a Deep learning model
    """
    _instances = list()

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
        >>> from rackio_AI import RackioAI
        >>> preprocess = RackioAI.get("Preprocessing", _type="Preprocessing")
        >>> print(preprocess.splitter)
        Splitter Object
        {'train_size': None, 'test_size': None, 'validation_size': None, 'random_state': None, 'shuffle': True, 'stratify': None}

        ```
        """
        RackioAISplitter._instances.append(self)
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
        >>> from rackio_AI import  RackioAI
        >>> import numpy as np
        >>> preprocess = RackioAI.get("Preprocessing", _type="Preprocessing")
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
        array([[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7],
               [ 8,  9],
               [10, 11]])
        >>> X_test
        array([[12, 13],
               [14, 15]])
        >>> X_validation
        array([[16, 17],
               [18, 19]])
        >>> y_train
        [0, 1, 2, 3, 4, 5]
        >>> y_test
        [6, 7]
        >>> y_validation
        [8, 9]

        ```

        ## Snippet code 3
        ```python
        >>> X_train, X_test, y_train, y_test = preprocess.splitter.split(X, y, train_size=0.6, test_size=0.4, random_state=0)
        >>> X_train
        array([[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7],
               [ 8,  9],
               [10, 11]])
        >>> X_test
        array([[12, 13],
               [14, 15],
               [16, 17],
               [18, 19]])
        >>> y_train
        [0, 1, 2, 3, 4, 5]
        >>> y_test
        [6, 7, 8, 9]


        ```
        """
        default_options = {'train_size': None,
                           'test_size': None,
                           'validation_size': None,
                           'random_state': None,
                           'shuffle': False,
                           'stratify': None}

        data = [array.values if isinstance(array, pd.DataFrame) else array for array in arrays]
        options = Utils.check_default_kwargs(default_options, options)
        train_size = options['train_size']
        test_size = options['test_size']
        self.validation_size = options.pop('validation_size')
        lst = [options['train_size'], options['test_size'], self.validation_size]
        
        if lst.count(None) >= 1 or (options['train_size'] + options['test_size'] == 1):
            
            return self.__split(TRAIN_TEST_SPLIT, *data, **options)
        
        return self.__split(TRAIN_TEST_VALIDATION_SPLIT, *data, **options)

    def __split(self, flag, *data, **options):
        """
        splitter manager to do one_split or two_split
        :param flag:
        :param data:
        :param default_options:
        :return:
        """
        if flag==TRAIN_TEST_SPLIT:

            return self.__one_split(*data, **options)

        elif flag==TRAIN_TEST_VALIDATION_SPLIT:

            return self.__two_splits(*data, **options)

        return

    def __one_split(self, *data, **options):
        """
        Split data in train and test datasets
        :return:
        """
        return TTS(*data, **options)

    def __two_splits(self, *data, **options):
        """
        Split data in train-test and validation datasets
        """
        test_size = options['test_size']

        validation_size = 1 - options['train_size'] - test_size

        options['test_size'] = None

        # First split
        X_train, X_test, y_train, y_test = TTS(*data, **options)

        options['train_size'] = test_size / (test_size + validation_size)

        # Second split
        X_test, X_validation, y_test, y_validation = TTS(X_test, y_test, **options)

        return [X_train, X_test, X_validation, y_train, y_test, y_validation]

    def __check_split_sizes(self, **options):
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


@set_to_methods(del_temp_attr)
class LSTMDataPreparation(RackioAISplitter):
    """
    Documentation here
    """
    _instances = list()

    def __init__(self):
        """Documnetation here"""
        super(LSTMDataPreparation, self).__init__()
        LSTMDataPreparation._instances.append(self)

    def split_sequences(
        self, 
        df: pd.DataFrame, 
        timesteps,
        stepsize: int= 1, 
        input_cols: list=None, 
        output_cols: list=None,
        maxlen=None,
        dtype: str='int32',
        padding: str='pre',
        truncating: str='pre',
        value: float=0.
        ):
        """
        Splits dataframe in a 3D numpy array format supported by LSTM architectures using sliding windows concept.

        **Parameters**

        * **:param df:** (pandas.DataFrame) Contains inputs and outputs data
        * **:param timesteps:** (list or int) Timestep for each input variable.
            * If timestep is an int value, all input columns will be the same timestep
            * If timestep is a list, must be same lenght that input_cols argument
        * **:param stepsize:** (int, default = 1) step size for the sliding window
        * **:param input_cols:** (list, default = None) Column names that represents the input variables to LSTM
            * If input_cols is None the method assumes that inputs are all column except the last one.
        * **:param output_cols:** (list, default = None) Column names that represents the output variables to LSTM
            * If output_cols is None the method assumes that output is the last column.

        The rest of parameters represent the parameters for *pad_sequences* method, see its description.

        **returns**

        **sequences** (3D numpy array) dimensions (df.shape[0] - max(timesteps), max(timesteps), features)

        ```python
        >>> import numpy as np
        >>> from rackio_AI import RackioAI
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
        >>> preprocess = RackioAI.get("Preprocessing", _type="Preprocessing")
        >>> x, y = preprocess.lstm_data_preparation.split_sequences(df, 2)
        >>> x.shape
        (8, 2, 2)
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
        >>> y.shape
        (8, 1, 1)
        >>> y
        array([[[ 45.]],
        <BLANKLINE>
               [[ 65.]],
        <BLANKLINE>
               [[ 85.]],
        <BLANKLINE>
               [[105.]],
        <BLANKLINE>
               [[125.]],
        <BLANKLINE>
               [[145.]],
        <BLANKLINE>
               [[165.]],
        <BLANKLINE>
               [[185.]]])

        ```
        """

        if not input_cols:

            input_cols = Utils.get_column_names(df)
            input_cols = input_cols[:-1]
        
        if not output_cols:

            output_cols = Utils.get_column_names(df)
            output_cols = [output_cols[-1]]

        if isinstance(timesteps, list):

            if not len(timesteps) == len(input_cols):

                raise ValueError('timesteps and input_cols arguments must be same length')
            
        else:

            timesteps = [timesteps] * len(input_cols)

        input_data = df.loc[:, input_cols].values
        output_data = df.loc[:, output_cols].values
        iteration = list(range(0, input_data.shape[0] - max(timesteps) + stepsize, stepsize))

        self.x_sequences = np.zeros((len(iteration), max(timesteps), len(input_cols)))
        self.y_sequences = np.zeros((len(iteration), 1, len(output_cols)))

        self.start = 0  

        options = {
            'output_data': output_data,
            'input_data': input_data,
            'timesteps': timesteps,
            'maxlen': maxlen,
            'dtype': dtype,
            'padding': padding,
            'truncating': truncating,
            'value': value
        }
        
        self.__split_sequences(iteration, **options)

        return self.x_sequences, self.y_sequences

    @ProgressBar(desc="Splitting sequences...", unit="windows")
    def __split_sequences(self, sequence, **kwargs):
        """
        Documentation here
        """
        to_pad = list()
        output_data = kwargs['output_data']
        input_data = kwargs['input_data']
        timesteps = kwargs['timesteps']
        maxlen = kwargs['maxlen']
        dtype = kwargs['dtype']
        padding = kwargs['padding']
        truncating = kwargs['truncating']
        value = kwargs['value']

        for count, timestep in enumerate(timesteps):

            to_pad.append(input_data[self.start + max(timesteps) - timestep: self.start + max(timesteps), count].tolist())
        
        self.x_sequences[self.start] = self.pad_sequences(
            to_pad, 
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
            value=value
            )
        
        self.y_sequences[self.start] = output_data[self.start + max(timesteps) - 1, :]

        self.start += 1

        return

    def pad_sequences(
        self,
        sequences,
        maxlen=None,
        dtype='int32',
        padding='pre',
        truncating='pre',
        value=0.
        ):
        """
        Pads sequences to the same length.

        This function transforms a list (of length `num_samples`) of sequences (lists of integers)
        into a 2D Numpy array of shape `(num_samples, num_timesteps)`.

        `num_timesteps` is either the `maxlen` argument if provided, or the length of the longest 
        sequence in the list.

        Sequences that are shorter than `num_timesteps` are padded with `value` until they are 
        `num_timesteps` long.

        Sequences longer than `num_timesteps` are truncated so that they fit the desired length.

        The position where padding or truncation happens is determined by the arguments `padding` 
        and `truncating`, respectively.
        Pre-padding or removing values from the beginning of the sequence is the
        default.

        **Parameters**

        * **:param sequences:** (list) List of sequences (each sequence is a list of integers).
        * **:param maxlen:** (Optional Int), maximum length of all sequences. If not provided,
        sequences will be padded to the length of the longest individual sequence.
        * **:param dtype:** (Optional, defaults to int32). Type of the output sequences.
        To pad sequences with variable length strings, you can use `object`.
        * **:param padding:** (String, 'pre' or 'post') (optional, defaults to 'pre'):
        pad either before or after each sequence.
        * **:param truncating:** (String, 'pre' or 'post') (optional, defaults to 'pre'):
        remove values from sequences larger than `maxlen`, either at the beginning or at the end of the sequences.
        * **:param value:** (Float or String), padding value. (Optional, defaults to 0.)
        
        **returns:**
            
        * **Numpy array** with shape `(len(sequences), maxlen)`
        
        **Raises:**
        
        * **ValueError:** In case of invalid values for `truncating` or `padding`, or in case of invalid
        shape for a `sequences` entry.

        ```python
        >>> from rackio_AI import RackioAI
        >>> sequence = [[1], [2, 3], [4, 5, 6]]
        >>> preprocessing = RackioAI.get("Preprocessing", _type="Preprocessing")
        >>> preprocessing.lstm_data_preparation.pad_sequences(sequence)
        array([[0, 0, 4],
               [0, 2, 5],
               [1, 3, 6]])
        >>> preprocessing.lstm_data_preparation.pad_sequences(sequence, value=-1)
        array([[-1, -1,  4],
               [-1,  2,  5],
               [ 1,  3,  6]])
        >>> preprocessing.lstm_data_preparation.pad_sequences(sequence, padding='post')
        array([[1, 2, 4],
               [0, 3, 5],
               [0, 0, 6]])
        >>> preprocessing.lstm_data_preparation.pad_sequences(sequence, maxlen=2)
        array([[0, 2, 5],
               [1, 3, 6]])
        
        ```
        """

        return np.transpose(pad_seq(sequences, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating, value=value))
