import functools

def decorator(declared_decorator):
    """
    Create a decorator out of a function, which will be used as a wrapper
    """

    @functools.wraps(declared_decorator)
    def final_decorator(func=None, **kwargs):
        # This will be exposed to the rest of your application as a decorator
        def decorated(func):
            # This will be exposed to the rest of your application as a decorated
            # function, regardless how it was called
            @functools.wraps(func)
            def wrapper(*a, **kw):
                # This is used when actually executing the function that was decorated

                return declared_decorator(func, a, kw, **kwargs)
            
            return wrapper
        
        if func is None:
            
            return decorated
        
        else:
            # The decorator was called without arguments, so the function should be
            # decorated immediately
            return decorated(func)

    return final_decorator


@decorator
def check_if_is_list(func, args, kwargs):
    # print(f"func: {func} - args: {args} - kwargs: {kwargs}")
    elem_to_check = args[1]
    _self = args[0]
    new_result = list()

    if isinstance(elem_to_check, list):

        # print("Se hace la modificación y retornamos el mismo tipo de dato")
        for elem in elem_to_check:

            df = elem['tpl']
            _args = [arg for arg in args[2:] if arg]
            # print(args)
            _result = func(args[0], df, *_args, **kwargs)

            new_result.append(
                {
                    'tpl': _result,
                    'genkey': elem['genkey'],
                    'settings': elem['settings']
                }
            )

        result = new_result
        setattr(_self, 'etl_data', result)

    else:
        
        result = func(*args, **kwargs)

    # breakpoint()

    return result