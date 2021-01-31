from sklearn.model_selection import train_test_split as TTS
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad_seq
from rackio_AI.utils.utils_core import Utils
from easy_deco.progress_bar import ProgressBar
import numpy as np
import pandas as pd
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


ONE_SPLIT = 1
TWO_SPLIT = 2

@set_to_methods(del_temp_attr)
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
        >>> preprocess = Preprocessing(name='Preprocess splitter init', description='preprocess for data', problem_type='regression')
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
        >>> preprocess = Preprocessing(name='Preprocess splitter split', description='preprocess for data', problem_type='regression')
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

        self._x_sequences_ = np.zeros((len(iteration), max(timesteps), len(input_cols)))
        self._y_sequences_ = np.zeros((len(iteration), 1, len(output_cols)))

        self._start_ = 0  

        self._output_data_ = output_data
        self._input_data_ = input_data
        self._timesteps_ = timesteps
        self._maxlen_ = maxlen
        self._dtype_ = dtype
        self._padding_ =  padding
        self._truncating_ = truncating
        self._value_ = value
        
        self.__split_sequences(iteration)

        return self._x_sequences_, self._y_sequences_

    @ProgressBar(desc="Splitting sequences...", unit="windows")
    def __split_sequences(self, sequence, **kwargs):
        """
        Documentation here
        """
        to_pad = list()
        output_data = self._output_data_
        input_data = self._input_data_
        timesteps = self._timesteps_
        maxlen = self._maxlen_
        dtype = self._dtype_
        padding = self._padding_
        truncating = self._truncating_
        value = self._value_

        for count, timestep in enumerate(timesteps):

            to_pad.append(input_data[self._start_ + max(timesteps) - timestep: self._start_ + max(timesteps), count].tolist())
        
        self._x_sequences_[self._start_] = self.pad_sequences(
            to_pad, 
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
            value=value
            )
        
        self._y_sequences_[self._start_] = output_data[self._start_ + max(timesteps) - 1, :]

        self._start_ += 1

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
        >>> from rackio_AI import Preprocessing
        >>> sequence = [[1], [2, 3], [4, 5, 6]]
        >>> preprocessing = Preprocessing(name='Pad sequence', description='preprocess for data', problem_type='regression')
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


if __name__=="__main__":
    import doctest
    doctest.testmod()
    # a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]).reshape(-1,1)
    # b = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95]).reshape(-1,1)
    # c = np.array([a[i]+b[i] for i in range(len(a))]).reshape(-1,1)
    # data = np.hstack((a,b,c))
    # print(data)
    # df = pd.DataFrame(data, columns=['a', 'b', 'c'])
    # lst = LSTMDataPreparation()
    # x, y = lst.split_sequences(df, [2,1])
    # print(x)
    # print(x.shape)
    # print(y.shape)
    