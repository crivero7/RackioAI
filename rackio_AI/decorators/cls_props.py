import inspect
import types

class DecoMeta(type):
    def __new__(cls, name, bases, attrs):

        for attr_name, attr_value in attrs.iteritems():
            
            if isinstance(attr_value, types.FunctionType):
                
                attrs[attr_name] = cls.deco(attr_value)

        return super(DecoMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def deco(cls, func):

        def wrapper(*args, **kwargs):

            result = func(*args, **kwargs)
            attributes = inspect.getmembers(cls, lambda variable:not(inspect.isroutine(variable)))
                
            for variable in attributes:
                
                if variable[0].startswith('_') and variable[0].endswith('_'):

                    if not(variable[0].startswith('__') and variable[0].endswith('__')):

                        delattr(cls, variable[0])
            
            return result

        return wrapper