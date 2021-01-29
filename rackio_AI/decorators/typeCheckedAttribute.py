from easy_deco import decorator

class Typed:
    """
    ...Description here...

    """

    def __init__(self, name, expectedType):
        """
        ...Description here...

        **Parameters**

        * **:param name:**
        * **:param expectedType:**

        """
        self.name = name
        self.expectedType = expectedType

    def __set__(self, instance, value):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**
        * **:param value:**

        **:return:**

        """
        if not any([isinstance(value, expectedType) for expectedType in self.expectedType]):
            raise TypeError('Expected {}'.format(self.expectedType))
        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**
        * **:param owner:**

        **:return:**

        """
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __delete__(self, instance):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**

        **:return:**

        """
        del instance.__dict__[self.name]


class MinMaxLengthString(object):
    """
    ...Description here...

    """
    def  __init__(self, min_default, max_default):
        """
        ...Description here...

        **Parameters**

        * **:param min_default:**
        * **:param max_default:**

        """
        self.max_default = max_default
        self.min_default = min_default
        self.data = {}

    def __get__(self, instance, owner):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**
        * **:param owner:**

        **:return:**

        """
        return self.data.get(instance)

    def __set__(self, instance, value):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**
        * **:param value:**

        **:return:**

        """
        if len(value) > self.max_default or len(value) < self.min_default:
            raise ValueError('Invalid length')
        self.data[instance] = value


class MinIntegerValue(object):
    """
    ...Description here...

    """
    def __init__(self, min_value):
        """
        ...Description here...

        **Parameters**

        * **:param min_value:**

        """
        self.min_value = min_value

        self.data = {}

    def __get__(self, instance, owner):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**
        * **:param owner:**

        **:return:**

        """
        return self.data.get(instance)

    def __set__(self, instance, value):
        """
        ...Description here...

        **Parameters**

        * **:param instance:**
        * **:param value:**

        **:return:**

        """
        if value < self.min_value:
            raise ValueError('Valor menor de lo permitido')

        self.data[instance] = value

# Class decorator that applies it to selected attributes
def typeassert(**kwargs):
    """
    ...Description here...

    **Parameters**

    * **:param kwargs:**

    **:return:**

    """
    def decorate(cls):
        
        for name, expectedType in kwargs.items():
            
            setattr(cls, name, Typed(name, expectedType))
        
        return cls
    
    return decorate

@decorator
def check_instrument_options(func, args, options):
    """
    ...Descriptions here...
    """
    options = func(*args, **options)

    lengthOfEachOption = [len(options[option]) for option in options]

    # verificar cantidad de sensores
    lengthOfEachOption.append(args[-1].data.shape[-1])

    if not all([x == lengthOfEachOption[0] for x in lengthOfEachOption]):

        raise ValueError('{} must be the same length that data sensor'.format(options.keys()))

    return options


def typeMinMaxValue(**kwargs):
    """
    ...Description here...

    **Parameters**

    * **:param kwargs:**

    **:return:**

    """
    def decorate(cls):
        for name, value in kwargs.items():
            if isinstance(cls[name],str):
                setattr(cls, name, MinMaxLengthString(value[0], value[1]))
            else:
                setattr(cls, name, MinIntegerValue(value))
        return cls
    return decorate