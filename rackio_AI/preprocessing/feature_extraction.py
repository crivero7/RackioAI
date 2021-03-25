from scipy.stats import kurtosis, skew
from rackio_AI.utils.utils_core import Utils
import pywt
import numpy as np
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


@set_to_methods(del_temp_attr)
class RackioAIFE:
    """
    Documentation here
    """
    _instances = list()

    def __init__(self):
       """Documentation here"""
       pass

    def kurt(self, dataset, axis=0, fisher=True, bias=True, nan_policy='propagate'):
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset

        Kurtosis is the fourth central moment divided by the square of the variance. If Fisher's definiton
        is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.

        If bias is False then the kurtosis is calculated using k statistics to eliminate bias coming from
        biased moment estimators

        **Parameters**

        * **dataset:** (array) Data for which the kurtosis is calculated
        * **axis:** (int or None) Axis along which the kurtosis is calculated. Default is 0. If None, compute
        over the whole array dataset.
        * **fisher:** (bool) If True, Fisher's definition is used (normal ==> 0.0). If False, Pearson's deifnition
        is used (normal ==> 3.0)
        * **bias:** (bool) If False, then the calculations are corrected for statistical bias.
        * **nan_policy:** ({'propagate', 'raise', 'omit'}) Defines how to handle when inputs contains nan. 'propagate' 
        returns nan, 'raise' throws an error, 'omit' performs the calculations ignoring nan values. Default is propagate.

        **returns**

        * **kurtosis** (array) The kurtosis of values along an axis. If all values are equal, return -3 for Fisher's definition
        and 0 for Pearson's definition

        ## Snippet Code
        ```python
        >>> from scipy.stats import norm
        >>> from rackio_AI import Preprocessing
        >>> dataset = norm.rvs(size=1000, random_state=3)
        >>> preprocessing = Preprocessing()
        >>> preprocessing.feature_extraction.kurt(dataset)
        -0.06928694200380558

        ```
        """
        _kurt = [kurtosis(col, axis=axis) for col in datasey]
        _kurt = np.concatenate(_kurt, axis=0)
        _kurt = _kurt.reshape((dataset.shape[0], dataset.shape[2]))
        return _kurt

    def mean(self, data):
        """Documentation here"""
        _mean = [np.mean(col, axis=0) for col in data]
        _mean = np.concatenate(_mean, axis=0)
        _mean = _mean.reshape((data.shape[0], data.shape[2]))
        return _mean

    def std(self, data):
        """Documentation here"""
        _std = [np.std(col, axis=0) for col in data]
        _std = np.concatenate(_std, axis=0)
        _std = _std.reshape((data.shape[0], data.shape[2]))
        return _std

    def skew(self, data):
        """Documentation here"""
        _skew = [skew(col, axis=0) for col in data]
        _skew = np.concatenate(_skew, axis=0)
        _skew = _skew.reshape((data.shape[0], data.shape[2]))
        return _skew

    def rms(self, data):
        """Documentation here"""
        _rms = [(np.sum(col ** 2, axis=0) / data.shape[0]) ** 0.5 for col in data]
        _rms = np.concatenate(_rms, axis=0)
        _rms = _rms.reshape((data.shape[0], data.shape[2]))
        return _rms

    def peak_2_valley(self, data):
        """Documentation here"""
        _peak_2_valley = [(np.max(col, axis=0)-np.min(col, axis=0)) / 2 for col in data]
        _peak_2_valley = np.concatenate(_peak_2_valley, axis=0)
        _peak_2_valley = _peak_2_valley.reshape((data.shape[0], data.shape[2]))
        return _peak_2_valley

    def peak(self, data):
        """Documentation here"""
        _peak = [np.max(col - col[0,:], axis=0) for col in data]
        _peak = np.concatenate(_peak, axis=0)
        _peak = _peak.reshape((data.shape[0], data.shape[2]))
        return _peak

    def crest_factor(self):
        """Documentation here"""
        peak = self.peak()
        rms = self.rms()
        return np.array([peak[i,:] / rms[i,:] for i in range(peak.shape[0])])

    def statistical(
        self, 
        data, 
        std=True, 
        mean=False,
        rms=False,
        kurt=False,
        skew=False,
        peak=False,
        peak_2_valley=False,
        crest_factor=False,
        concatenate=True
        ):
        """Documentation here"""
        result = list()
        features = {
            'std': std,
            'mean': mean,
            'rms': rms, 
            'kurt': kurt, 
            'skew': skew, 
            'peak': peak, 
            'peak_2_valley': peak_2_valley, 
            'crest_factor': crest_factor
        }

        for key, value in features.items():
            
            if value:
                feature = getattr(self, key)
                result.append(feature(data))

        if concatenate:

            result = np.concatenate(result, axis=1)

        return result

    def wavelets(self, waveletType='db2', lvl=3):
        """Documentation here"""
        waveletFeatures = list()
        lastCoeffs = 3
        for data in self.inputData:
            coeffs = pywt.wavedec(data, waveletType, level=lvl, axis=0)
            waveletFeatures.append(np.concatenate([sum(coeff ** 2, 0) for coeff in coeffs[0:lastCoeffs]],axis=0))
        return np.array(waveletFeatures)


if __name__=='__main__':

    import doctest

    doctest.testmod()