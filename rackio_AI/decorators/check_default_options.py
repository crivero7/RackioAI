from .core import decorator

@decorator
def check_default_options(func, args, options):
    """

    """
    default_options = func(*args, **options)

    options = {key: options[key] if key in options.keys() else default_options[key] for key in default_options.keys()}

    return options