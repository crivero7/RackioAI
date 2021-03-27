from functools import wraps
import pywt

class WaveletDeco(object):

    def __init__(self, f):
        self.func = f

    def __get__(self, instance, cls):
        self.instance = instance
        return types.MethodType(self, instance)

    @staticmethod
    def is_valid(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            wavelet = args[2]
            families = pywt.families()
            wavelets = list()
            [wavelets.extend(pywt.wavelist(family)) for family in families]
            if not wavelet in wavelets:

                raise TypeError("{} is not a valid wavelet".format(wavelet))

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def mode_is_valid(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'mode' in kwargs:
                mode = kwargs['mode']
                modes = pywt.Modes.modes
                if not mode in modes:

                    raise TypeError("{} is not a valid mode, use: {}".format(mode, modes))

            return func(*args, **kwargs)

        return wrapper


def mode_is_valid(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        mode = kwargs['mode']
        modes = pywt.Modes.modes
        if not mode in modes:

            raise TypeError("{} is not a valid mode, use: {}".format(mode, modes))

        return func(*args, **kwargs)

    return wrapper

def is_valid(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        wavelet = args[2]
        families = pywt.families()
        wavelets = list()
        wavelets = [wavelets.extend(pywt.wavelist(family)) for family in families]
        if not wavelet in wavelets:

            raise TypeError("{} is not a valid wavelet".format(wavelet))

        return func(*args, **kwargs)

    return wrapper