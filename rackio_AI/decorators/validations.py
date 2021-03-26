from functools import wraps


def check_instrument_attributes(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        inst = args[0]
        if not hasattr(inst, "accuracy"):

            raise NotImplementedError("Please add instrument attributes")

        return func(*args, **kwargs)

    return wrapper

