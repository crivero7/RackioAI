from tqdm import tqdm
from .core import decorator

@decorator
def progress_bar(func, *args, **kwargs):
    """

    """
    defaultOptions = {'desc': 'Reading files',
                      'unit': 'Files'}

    Options = {key: kwargs[key] if key in list(kwargs.keys()) else defaultOptions[key] for key in
               list(defaultOptions.keys())}

    (self, iterableObject) = args[0]

    if not hasattr(iterableObject, '__iter__'):

        raise ValueError('You must provide an iterableObject in {}'.format(func.__name__))

    for i in tqdm(range(len(iterableObject)), desc=Options['desc'], unit=Options['unit']):

        result = func(self, iterableObject[i])

    return result