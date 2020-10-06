import functools

class Typed:

    def __init__(self, name, expectedType):
        self.name = name
        self.expectedType = expectedType

    def __set__(self, instance, value):
        if not any([isinstance(value, expectedType) for expectedType in self.expectedType]):
            raise TypeError('Expected {}'.format(self.expectedType))
        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class MinMaxLengthString(object):
    def  __init__(self, min_default, max_default):
        self.max_default = max_default
        self.min_default = min_default
        self.data = {}

    def __get__(self, instance, owner):
        return self.data.get(instance)

    def __set__(self, instance, value):
        if len(value) > self.max_default or len(value) < self.min_default:
            raise ValueError('Invalid length')
        self.data[instance] = value


class MinIntegerValue(object):
    def __init__(self, min_value):
        self.min_value = min_value

        self.data = {}

    def __get__(self, instance, owner):
        return self.data.get(instance)

    def __set__(self, instance, value):
        if value < self.min_value:
            raise ValueError('Valor menor de lo permitido')

        self.data[instance] = value

# Class decorator that applies it to selected attributes
def typeassert(**kwargs):
    def decorate(cls):
        for name, expectedType in kwargs.items():
            setattr(cls, name, Typed(name, expectedType))
        return cls
    return decorate


def checkOptions(function=None, **kwargs):
    def decorator(function):
        @functools.wraps(function)
        def wrap(*args, **options):
            defaultOptions = function(*args, **options)
            options = {key: options[key] if key in options.keys() else defaultOptions[key] for key in defaultOptions.keys()}
            lengthOfEachOption = [len(options[option]) for option in options]
            # verificar cantidad de sensores
            lengthOfEachOption.append(args[-1].data.shape[-1])
            if not all([x==lengthOfEachOption[0] for x in lengthOfEachOption]):
                raise ValueError('{} must be the same length that data sensor'.format(options.keys()))
            return options
        return wrap
    if function is None:
        return decorator
    else:
        return decorator(function)

def typeMinMaxValue(**kwargs):
    def decorate(cls):
        for name, value in kwargs.items():
            if isinstance(cls[name],str):
                setattr(cls, name, MinMaxLengthString(value[0], value[1]))
            else:
                setattr(cls, name, MinIntegerValue(value))
        return cls
    return decorate