from .core import decorator

@decorator
def raise_error(func, args , kwargs):
    """

    """
    try:
        return func(*args, **kwargs)

    except Exception as e:

        raise eval(e.__class__.__name__)(e)