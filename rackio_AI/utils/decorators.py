import functools
import logging


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


def check_if_is_list(one_key_dict: bool = False):
    def _check_if_is_list(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            try:
                elem_to_check = args[1]
                _self = args[0]
                new_result = list()

                if isinstance(elem_to_check, list):
                    for elem in elem_to_check:
                        # TODO Change df to read data from 'data' key also.

                        if 'data' in elem.keys():
                            df = elem['data']
                        else:
                            df = elem['tpl']

                        _args = [arg for arg in args[2:] if arg]
                        # print(args)
                        _result = func(args[0], df, *_args, **kwargs)

                        if not one_key_dict:
                            new_result.append(
                                {
                                    'tpl': _result,
                                    'genkey': elem['genkey'],
                                    'settings': elem['settings']
                                }
                            )
                        else:
                            new_result.append(
                                {
                                    'data': _result
                                }
                            )
                    result = new_result
                    setattr(_self, 'etl_data', result)

                else:

                    result = func(*args, **kwargs)

                # breakpoint()

                return result

            except Exception as err:

                logging.ERROR(str(err))

        return decorated

    return _check_if_is_list
