from functools import wraps
from tqdm import tqdm

def progressBar(**kwargs):
    """
    ...Description here...

    **Parameters**

    * **:param kwargs:**
        * **desc:** (str) description
        * **unit:** (str) measure unit

    **:return:**

    func
    """
    defaultOptions = {'desc': 'Reading files',
                      'unit': 'Files'}
    Options = {key:kwargs[key] if key in list(kwargs.keys()) else defaultOptions[key] for key in list(defaultOptions.keys())}
    def decorator(func):
        """

        :param func:
        :return:
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            iterableObject = args[1]
            if not hasattr(iterableObject,'__iter__'):
                raise ValueError('You must provide an iterableObject in {}'.format(func.__name__))
            for i in tqdm(range(len(iterableObject)), desc=Options['desc'], unit=Options['unit']):
                result = func(args[0], iterableObject[i])
            return result
        return wrapper
    return decorator