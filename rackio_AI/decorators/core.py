import functools

def decorator(declared_decorator):
    """
    Create a decorator out of a function, which will be used as a wrapper.
    """

    @functools.wraps(declared_decorator)
    def final_decorator(func=None, **kwargs):
        # This will be exposed to the rest  of your application as a decorator

        def decorated(func):
            # This will be exposed to the rest of your application as a decorated function, regardless how it was called
            @functools.wraps(func)
            def wrapper(*a, **kw):
                # This is used when actually executing the function that was decorated

                return declared_decorator(func, a, kw, **kwargs)

            return wrapper

        if func is None:
            # The decorator was called with arguments, rather than function to decorate

            return decorated

        else:
            # The decorator was called without arguments, so the function should be decorated inmediately
            return decorated(func)

    return final_decorator