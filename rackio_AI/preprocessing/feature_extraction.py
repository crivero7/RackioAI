from scipy.stats import kurtosis, skew
from rackio_AI.utils.utils_core import Utils
import pywt
import numpy as np
import pandas as pd
from rackio_AI.decorators.wavelets import WaveletDeco
from easy_deco.progress_bar import ProgressBar
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr

# @set_to_methods(del_temp_attr)
class StatisticalsFeatures:
    """
    When we consider the original discretized time domain signal , some basic discriminative
    information can be extracted in form of statistical parameters from the $n$ samples
    $s_{1},\cdots s_{n}$
    """

    _instances = list()

    def mean(
        self, 
        s,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue
        ):
        r"""
        Compute the arithmetic mean along the specified axis.
        Returns the average of the array elements.  The average is taken over
        the flattened array by default, otherwise over the specified axis.
        `float64` intermediate and return values are used for integer inputs.
        
        **Parameters**
        
        * **s:** (2d array_like) Array containing numbers whose mean is desired. If `s` is not an
        array, a conversion is attempted.
        * **axis:** (None or int or tuple of ints, optional) Axis or axes along which the means are computed. 
        The default is to compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis or all the
        axes as before.
        * **dtype:** (data-type, optional) Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the input dtype.
        * **out:** (ndarray, optional) Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the expected output, but the type will be cast if
        necessary.
        * **keepdims:** (bool, optional) If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option, the result will broadcast correctly against the 
        input array. If the default value is passed, then `keepdims` will not be passed through to the `mean` method 
        of sub-classes of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement
        `keepdims` any exceptions will be raised.
        
        **Returns**
        
        * **m:** (ndarray, see dtype parameter above) If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = np.array([[1, 2], [3, 4]])
        >>> feature_extraction.stats.mean(s)
        2.5
        >>> feature_extraction.stats.mean(s, axis=0)
        array([2., 3.])
        >>> feature_extraction.stats.mean(s, axis=1)
        array([1.5, 3.5])

        ```
        """
        s = Utils.check_dataset_shape(s)
        return np.mean(s, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def median(
        self, 
        s, 
        axis=None, 
        out=None, 
        overwrite_input=False, 
        keepdims=False
        ):
        r"""
        Compute the median along the specified axis.
        Returns the median of the array elements.
        
        **Parameters**
        
        * **s:** (2d array_like) Input array or object that can be converted to an array.
        * **axis:** ({int, sequence of int, None}, optional) Axis or axes along which the medians \
        are computed. The default is to compute the median along a flattened version of the array.
        * **out:** (ndarray, optional) Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output, but the type (of the output) 
        will be cast if necessary.
        * **overwrite_input:** (bool, optional) If True, then allow use of memory of input array 
        `s` for calculations. The input array will be modified by the call to `median`. 
        This will save memory when you do not need to preserve the contents of the input array. 
        Treat the input as undefined, but it will probably be fully or partially sorted. Default is
        False. If `overwrite_input` is ``True`` and `s` is not already an `ndarray`, an error
        will be raised.
        * **keepdims:** (bool, optional) If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option, the result will broadcast 
        correctly against the original `array`.
        
        **Returns**
        
        * **median:** (ndarray) A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is ``np.float64``.  
        Otherwise, the data-type of the output is the same as that of the input. If `out` is 
        specified, that array is returned instead.

        ## Notes
        
        Given a vector $V$ of length $N$, the median of $V$ is the
        middle value of a sorted copy of $V$, $V_{sorted}$ - i
        e., $V_{sorted}\left[\frac{N-1}{2}\right]$, when $N$ is odd, and the average of the
        two middle values of $V_{sorted}$ when $N$ is even.
        
        ## Snippet code
        
        ```python
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = np.array([[10, 7, 4], [3, 2, 1]])
        >>> feature_extraction.stats.median(s)
        3.5
        >>> feature_extraction.stats.median(s, axis=0)
        array([6.5, 4.5, 2.5])
        >>> feature_extraction.stats.median(s, axis=1)
        array([7., 2.])
        >>> m = feature_extraction.stats.median(s, axis=0)
        >>> out = np.zeros_like(m)
        >>> feature_extraction.stats.median(s, axis=0, out=m)
        array([6.5, 4.5, 2.5])
        >>> m
        array([6.5, 4.5, 2.5])
        >>> b = s.copy()
        >>> feature_extraction.stats.median(b, axis=1, overwrite_input=True)
        array([7., 2.])
        >>> assert not np.all(s==b)
        >>> b = s.copy()
        >>> feature_extraction.stats.median(b, axis=None, overwrite_input=True)
        3.5
        >>> assert not np.all(s==b)

        ```
        """
        s = Utils.check_dataset_shape(s)
        return np.median(
            s, 
            axis=axis, 
            out=out, 
            overwrite_input=overwrite_input, 
            keepdims=keepdims
            )

    def kurt(
        self, 
        s, 
        axis: int=0, 
        fisher: bool=True, 
        bias: bool=True, 
        nan_policy: str='propagate'
        ):
        r"""
        Compute the kurtosis (Fisher or Pearson) of a dataset $s$

        Kurtosis is the fourth central moment divided by the square of the variance. If Fisher's definiton
        is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.

        If bias is False then the kurtosis is calculated using k statistics to eliminate bias coming from
        biased moment estimators

        **Parameters**

        * **s:** (2d array) Data for which the kurtosis is calculated
        * **axis:** (int or None) Axis along which the kurtosis is calculated. Default is 0. If None, compute
        over the whole array dataset.
        * **fisher:** (bool) If True, Fisher's definition is used (normal ==> 0.0). If False, Pearson's deifnition
        is used (normal ==> 3.0)
        * **bias:** (bool) If False, then the calculations are corrected for statistical bias.
        * **nan_policy:** ({'propagate', 'raise', 'omit'}) Defines how to handle when inputs contains nan. 'propagate' 
        returns nan, 'raise' throws an error, 'omit' performs the calculations ignoring nan values. Default is propagate.

        **returns**

        * **kurtosis** (array 1xcols_dataset) The kurtosis of values along an axis. If all values are equal, return -3 for Fisher's definition
        and 0 for Pearson's definition

        ## Snippet Code
        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = norm.rvs(size=1000, random_state=3)
        >>> feature_extraction.stats.kurt(s)
        array([-0.06928694])
        >>> s = norm.rvs(size=(1000,2), random_state=3)
        >>> feature_extraction.stats.kurt(s)
        array([-0.00560946, -0.1115389 ])

        ```
        """
        s = Utils.check_dataset_shape(s)
        return kurtosis(
            s, 
            axis=axis, 
            fisher=fisher, 
            bias=bias, 
            nan_policy=nan_policy
            )

    def std(
        self, 
        s, 
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=np._NoValue
        ):
        r"""
        Compute the standard deviation along the specified axis.

        Returns the standard deviation, a measure of the spread of a distribution,
        of the array elements. The standard deviation is computed for the
        flattened array by default, otherwise over the specified axis.
        
        **Parameters**
        
        * **s:** (2d array_like) Calculate the standard deviation of these values.
        * **axis:** (None or int or tuple of ints, optional) Axis or axes along which the standard deviation is computed.
        The default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single
        axis or all the axes as before.
        * **dtype:** (dtype, optional) Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is the same as the array type.
        * **out:** (ndarray, optional) Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated values) will be cast if necessary.
        * **ddof:** (int, optional) Means Delta Degrees of Freedom.  The divisor used in calculations
        is $N - ddof$, where $N$ represents the number of elements. By default `ddof` is zero.
        * **keepdims:** (bool, optional) If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option, the result will broadcast correctly 
        against the input array. If the default value is passed, then `keepdims` will not be passed through 
        to the `std` method of sub-classes of `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any exceptions will be raised.

        **Returns**
        
        * **standard_deviation:** (ndarray) If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

        ## Notes
        
        The standard deviation is the square root of the average of the squared
        deviations from the mean, i.e.

        $\mu = \frac{1}{N}\sum_{i=1}^{n}s_{i}$

        $std = \sqrt{\frac{1}{N}\sum_{i=1}^{n}|s_{i}-\mu|^2}$
        
        ## Snippet code
        ```python
        >>> import numpy as np
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = np.array([[1, 2], [3, 4]])
        >>> feature_extraction.stats.std(s, axis=0)
        array([1., 1.])
        >>> feature_extraction.stats.std(s, axis=1)
        array([0.5, 0.5])

        ```
        
        ### In single precision, std() can be inaccurate

        ```python
        >>> s = np.zeros((2, 512*512), dtype=np.float32)
        >>> s[0, :] = 1.0
        >>> s[1, :] = 0.1
        >>> feature_extraction.stats.std(s)
        0.45000005
        >>> s = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
        >>> feature_extraction.stats.std(s)
        2.614064523559687

        ```
        """
        s = Utils.check_dataset_shape(s)
        return np.std(
            s, 
            axis=axis, 
            dtype=dtype, 
            out=dtype, 
            ddof=ddof, 
            keepdims=keepdims
            )

    def skew(
        self, 
        s, 
        axis=0,
        bias=True,
        nan_policy='propagate'
        ):
        r"""
        Compute the sample skewness of a data set.
        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.
        
        **Parameters**

        * **s:** (ndarray) Input array.
        * **axis:** (int or None, optional) Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array `s`.
        * **bias:** (bool, optional) If False, then the calculations are corrected for statistical bias.
        * **nan_policy:** ({'propagate', 'raise', 'omit'}, optional) Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
            * 'propagate': returns nan
            * 'raise': throws an error
            * 'omit': performs the calculations ignoring nan values
        
        **Returns**
        
        * **skewness:** (ndarray) The skewness of values along an axis, returning 0 where all values are equal.
        
        ## Notes

        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.

        $g_1=\frac{m_3}{m_2^{3/2}}$

        where

        $m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i$
        
        is the biased sample $i\texttt{th}$ central moment, and $\bar{x}$ is
        the sample mean.  If $bias$ is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        $G_1=\frac{k_3}{k_2^{3/2}}=\frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.$
        
        ## References
        
        .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
        Probability and Statistics Tables and Formulae. Chapman & Hall: New
        York. 2000.
        Section 2.2.24.1
        
        ## Snippet code
        ```python
        >>> import numpy as np
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = np.array([1, 2, 3, 4, 5])
        >>> feature_extraction.stats.skew(s)
        array([0.])
        >>> s = np.array([2, 8, 0, 4, 1, 9, 9, 0])
        >>> feature_extraction.stats.skew(s)
        array([0.26505541])
        
        ```
        """
        s = Utils.check_dataset_shape(s)
        return skew(
            s, 
            axis=axis, 
            bias=bias, 
            nan_policy=nan_policy
            )

    def rms(
        self, 
        s, 
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        initial=np._NoValue
        ):
        r"""
        Root Mean Square One of the most important basic features that can be extracted directly from the time-domain
        signal is the RMS which describe the energy of the signal. It is defined as the square root
        of the average squared value of the signal and can also be called the normalized energy of the
        signal.

        $RMS = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1}s_{i}^{2}}$
        
        Especially in vibration analysis the RMS is used to perform fault detection, i.e. triggering an
        alarm, whenever the RMS surpasses a level that depends on the size of the machine, the nature
        of the signal (for instance velocity or acceleration), the position of the accelerometer, and so on.
        After the detection of the existence of a failure, fault diagnosis is performed relying on more
        sophisticated features. For instance the ISO 2372 (VDI 2056) norms define three different velocity
        RMS alarm levels for four different machine classes divided by power and foundations of the rotating
        machines.

        RMS of array elements over a given axis.

        **Parameters**
        
        * **s:** (2d array_like) Elements to get RMS.
        * **axis:** (None or int or tuple of ints, optional) Axis or axes along which a RMS is performed.  
        The default, axis=None, will get RMS of all the elements of the input array. If axis is negative
        it counts from the last to the first axis. If axis is a tuple of ints, a RMS is performed on all
        of the axes specified in the tuple instead of a single axis or all the axes as before.
        * **dtype:** (dtype, optional) The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `s` is used by default unless `s` has an integer 
        dtype of less precision than the default platform integer.  In that case, if `s` is signed 
        then the platform integer is used while if `s` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
        * **out:** (ndarray, optional) Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output values will be cast if necessary.
        * **keepdims:** (bool, optional) If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option, the result will broadcast correctly 
        against the input array. If the default value is passed, then `keepdims` will not be passed through
        to the `sum` method of sub-classes of `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any exceptions will be raised.
        * **initial:** (scalar, optional) Starting value for the sum.

        **Returns**
        
        * **RMS_along_axis:** (darray) An array with the same shape as `s`, with the specified
        axis removed.   If `s` is a 0-d array, or if `axis` is None, a scalar is returned. 
        If an output array is specified, a reference to `out` is returned.

        ## Snippet code
        
        ```python
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> feature_extraction.stats.rms(np.array([0.5, 1.5]))
        1.118033988749895
        >>> feature_extraction.stats.rms(np.array([0.5, 0.7, 0.2, 1.5]), dtype=np.int32)
        0.7071067811865476
        >>> feature_extraction.stats.rms(np.array([[0, 1], [0, 5]]))
        3.605551275463989
        >>> feature_extraction.stats.rms(np.array([[0, 1], [0, 5]]), axis=0)
        array([0.        , 3.60555128])
        >>> feature_extraction.stats.rms(np.array([[0, 1], [0, 5]]), axis=1)
        array([0.70710678, 3.53553391])

        ```
        You can also start the sum with a value other than zero:
        ```python
        >>> feature_extraction.stats.rms(np.array([2, 7, 10]), initial=5)
        7.2571803523590805

        ```
        """
        s = Utils.check_dataset_shape(s)
        return (np.sum(
            s ** 2, 
            axis=axis, 
            dtype=dtype, 
            out=out, 
            keepdims=keepdims, 
            initial=initial
            ) / s.shape[0]) ** 0.5

    def peak_2_valley(self, s, axis=0):
        r"""
        Another important measurement of a signal, considering a semantically coherent sampling
        interval, for instance a fixed-length interval or one period of a rotation, is the peak-to-valley
        (PV) value which reflects the amplitude spread of a signal:

        $PV=\frac{1}{2}\left(\max(s)\quad -\quad \min(s)\right)$

        **Parameters**

        * **s:**
        * **axis:**

        **Returns**

        * **peak_2_valley:**

        ## Snippet code

        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = norm.rvs(size=1000, random_state=3)
        >>> feature_extraction.stats.peak_2_valley(s)
        array([3.34321422])
        >>> s = norm.rvs(size=(1000,2), random_state=3)
        >>> feature_extraction.stats.peak_2_valley(s)
        array([2.99293034, 3.34321422])

        ```
        """
        s = Utils.check_dataset_shape(s)
        return (np.max(s, axis=axis)-np.min(s, axis=axis)) / 2

    def peak(self, s, ref=None, axis=0, rate=None, **kwargs):
        r"""
        I we consider only the maximum amplitude relative to zero $s_{ref}=0$ or a general reference
        level $s_{ref}$, we get the peak value

        $peak = \max\left(s_{i}-ref\right)$

        Often the peak is used in conjunction with other statistical parameters, for instance the 
        peak-to-average rate.

        $peak = \frac{\max\left(s_{i}-ref\right)}{\frac{1}{N}\sum_{i=0}^{N-1}s_{i}}$

        or peak-to-median rate

        **Parameters**

        * **s:**
        * **ref:**
        * **axis:**
        * **rate:**

        **Returns**

        * **peak:**

        ## Snippet code

        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = norm.rvs(size=1000, random_state=3)
        >>> feature_extraction.stats.peak(s)
        array([1.91382976])
        >>> s = norm.rvs(size=(1000,2), random_state=3)
        >>> feature_extraction.stats.peak(s)
        array([1.0232499 , 3.26594839])

        ```

        """
        s = Utils.check_dataset_shape(s)
        if not ref == None:

            _peak = np.max(s - ref, axis=axis)

        else:
            
            _peak = np.max(s - s[0,:], axis=axis)

        if not rate == None:

            if rate.lower() == 'average':
                
                return _peak / self.mean(s, **kwargs)

            elif rate.lower() == 'median':

                return _peak / self.median(s, **kwargs)

        else:
            
            return _peak
            
    def crest_factor(self, s, **kwargs):
        r"""
        When we relate the peak value to the RMS of the signal, we obtain the crest facto:

        $CF=\frac{peak}{RMS}$

        which expresses the spikiness of the signal. The crest factor is also known as peak-to-average
        ratio or peak-to-average power ratio and is used to characterize signals containing repetitive
        impulses in addition to a lower level continuous signal. The modulus of the signal should be
        used in the calculus.

        **Parameters**

        * **s:**

        **Returns**

        * **crest_factor:**

        ## Snippet code

        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> s = norm.rvs(size=1000, random_state=3)
        >>> feature_extraction.stats.crest_factor(s)
        array([1.89760521])
        >>> s = norm.rvs(size=(1000,2), random_state=3)
        >>> feature_extraction.stats.crest_factor(s)
        array([0.71532677, 2.28313758])

        ```
        """
        peak = self.peak(s, **kwargs)
        rms = self.rms(s, **kwargs)
        return peak / rms


# @set_to_methods(del_temp_attr)
class Wavelet:
    r"""
    A wavelet is a mathematical function used to divide a given function or continuous-time 
    signal into different scale components. Usually one can assign a frequency range to each 
    scale component. Each scale component can then be studied with a resolution that matches 
    its scale. A wavelet transform is the representation of a function by wavelets. 
    
    The wavelets are scaled and translated copies (known as "daughter wavelets") of a 
    finite-length or fast-decaying oscillating waveform (known as the "mother wavelet"). 
    
    Wavelet transforms have advantages over traditional Fourier transforms for representing 
    functions that have discontinuities and sharp peaks, and for accurately deconstructing 
    and reconstructing finite, non-periodic and/or non-stationary signals.

    """

    _instances = list()

    @WaveletDeco.is_valid
    @WaveletDeco.mode_is_valid
    def wavedec(self, s, wavelet, mode='symmetric', level=None, axis=-1):
        r"""
        Multilevel 1D Discrete Wavelet Transform of signal $s$
        
        **Parameters**

        * **s:** (array_like) Input data
        * **wavelet:** (Wavelet object or name string) Wavelet to use
        * **mode:** (str) Signal extension mode.
        * **level:** (int) Decomposition level (must be >= 0). If level is None (default)
        then it will be calculated using the `dwt_max_level` function.
        * **axis:** (int) Axis over which to compute the DWT. If not given, the last axis 
        is used.

        **Returns**

        * **[cA_n, cD_n, cD_n-1, ..., cD2, cD1]:** (list) Ordered list of coefficients arrays where
        $n$ denotes the level of decomposition. The first element `(cA_n)` of the result is approximation
        coefficients array and the following elements `[cD_n - cD1]` are details coefficients arrays.

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> coeffs = feature_extraction.freq.wavelet.wavedec([1,2,3,4,5,6,7,8], 'db1', level=2)
        >>> cA2, cD2, cD1 = coeffs
        >>> cD1
        array([-0.70710678, -0.70710678, -0.70710678, -0.70710678])
        >>> cD2
        array([-2., -2.])
        >>> cA2
        array([ 5., 13.])
        >>> s = np.array([[1,1], [2,2], [3,3], [4,4], [5, 5], [6, 6], [7, 7], [8, 8]])
        >>> coeffs = feature_extraction.freq.wavelet.wavedec(s, 'db1', level=2, axis=0)
        >>> cA2, cD2, cD1 = coeffs
        >>> cD1
        array([[-0.70710678, -0.70710678],
               [-0.70710678, -0.70710678],
               [-0.70710678, -0.70710678],
               [-0.70710678, -0.70710678]])
        >>> cD2
        array([[-2., -2.],
               [-2., -2.]])
        >>> cA2
        array([[ 5.,  5.],
               [13., 13.]])

        ```
        """
        coeffs = pywt.wavedec(s, wavelet, mode=mode, level=level, axis=axis)

        return coeffs

    def wave_energy(self, s, wavelet, mode='symmetric', level=None, axis=-1):
        r"""
        The energy of time-series data distributed in Approximate and Detailed coefficients
        are calculated as follows:

        $ED_{i}=\sum_{j=1}^{N}|D_{ij}|^2,\quad i={1,2,\cdots,level}$

        $EA_{level}=\sum_{j=1}^{N}|A_{lj}|^2$

        Where $ED_{i}$ represents energy in the $i^{th}$ detailed coefficient and $EA_{level}$
        is the energy in the $level^{th}$ approximate coefficient respectively. Further, the
        fraction of total signal energy present in the approximate and detailed components is
        calculated which serves as a feature vector for every sensor.

        **Parameters**

        * **s:** (array_like) Input data
        * **wavelet:** (Wavelet object or name string) Wavelet to use
        * **mode:** (str) Signal extension mode.
        * **level:** (int) Decomposition level (must be >= 0). If level is None (default)
        then it will be calculated using the `dwt_max_level` function.
        * **axis:** (int) Axis over which to compute the DWT. If not given, the last axis 
        is used.

        **Returns**

        * **[EA_n, ED_n, ED_n-1, ..., ED2, ED1]:** (list) Ordered list of energy arrays where
        $n$ denotes the level of decomposition.

        ## Snippet code

        ```python
        >>> from rackio_AI import RackioAIFE
        >>> feature_extraction = RackioAIFE()
        >>> energies = feature_extraction.freq.wavelet.wave_energy([1,2,3,4,5,6,7,8], 'db1', level=2)
        >>> eA2, eD2, eD1 = energies
        >>> eD1
        2.000000000000001
        >>> eD2
        8.000000000000004
        >>> eA2
        194.00000000000006
        >>> s = np.array([[1,1], [2,2], [3,3], [4,4], [5, 5], [6, 6], [7, 7], [8, 8]])
        >>> energies = feature_extraction.freq.wavelet.wave_energy(s, 'db1', level=2, axis=0)
        >>> eA2, eD2, eD1 = energies
        >>> eD1
        array([2., 2.])
        >>> eD2
        array([8., 8.])
        >>> eA2
        array([194., 194.])

        ```
        """
        energy = list()
        # Wavelet decomposition
        coeffs = self.wavedec(s, wavelet, mode=mode, level=level, axis=axis)
        # Get approximation coefficients
        approximation_coeff = coeffs.pop(0)
        # Energy approximation computation
        energy.append(np.sum(approximation_coeff ** 2, axis=axis))
        # Energy detailed computation
        for detailed_coeff in coeffs:
            energy.append(np.sum(detailed_coeff ** 2, axis=axis))
    
        return energy

    def get_energies(
        self, 
        s, 
        input_cols=None, 
        output_cols=None, 
        timesteps=10, 
        wavelet_type='db2',
        wavelet_lvl=2,
        axis=0,
        slide=False
        ):
        r"""
        Documentation here
        """
        self._s_ = s
        self.wavelet_type = wavelet_type
        self.wavelet_lvl = wavelet_lvl
        self.axis = axis
        self.result = list()
        rows = s.shape[0]
        if slide:
            rows = range(0 , rows - timesteps)
        else:
            rows = range(0 , rows - timesteps, timesteps)
        self.timesteps = timesteps
        self.input_cols = input_cols

        self.__get_energies(rows)
        
        result = np.array(self.result)

        return result

    @ProgressBar(desc="Getting wavelet energies...", unit=" Sliding windows")
    def __get_energies(self, row):
        r"""
        Documentation here
        
        """
        if isinstance(self._s_, pd.DataFrame):
            data = self._s_.loc[row: row + self.timesteps, self.input_cols].values
        else:
            data = self._s_[row,:,:]

        energies = self.wave_energy(
            data, 
            self.wavelet_type, 
            level=self.wavelet_lvl, 
            axis=self.axis
            )
        energies = np.concatenate(list(energies))
        self.result.append(list(energies))
        return

class FrequencyFeatures:
    r"""
    Documentation here
    """

    _instances = list()

    def __init__(self):
        r"""
        Documentation here
        """
        self.wavelet = Wavelet()

    def stft(self, s):
        r"""
        In construction...

        The Short Time Fourier Transform (SFTF for short) of a given frame $s\left(m,n\right)$
        is a Fourier transform performed in successive frames:

        $S\left(m,n\right)=\sum_{n}s\left(m,n\right)\cdot{e^{\frac{-j2\pi nk}{N}}}$

        where $s\left(m,n\right)=s\left(n\right)w\left(n-mL\right)$ and $w\left(n\right)$ is a windowing
        function of $N$ samples

        **Parameters**

        * **s:**

        **Returns**

        * **stft:**
        """
        pass


class RackioAIFE:
    r"""
    Rack Input/Output Artificial Intelligence Feature Extraction (RackioAIFE for short) is a class
    to allows to you make feature extraction for pattern recognition models.

    The feature extraction transforms originally high-dimensional patterns into lower dimensional vectors
    by capturing the essential of their characteristics. Various feature extraction techniques have been
    proposed in the literature for different signal applications. In speech and speaker recognition,
    fault diagnosis, they are essentially based on Fourier transform, cepstral analysis, autoregressive
    modeling, wavelet transform and statistical analysis.
    """
    stats = StatisticalsFeatures()
    freq = FrequencyFeatures()


if __name__=='__main__':

    import doctest

    doctest.testmod()