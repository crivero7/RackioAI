import inspect
import types


class DeleteTemporaryProperties(type):
    def __new__(cls, name, bases, attrs):

        for attr_name, attr_value in attrs.items():

            if attr_name.startswith('_') and attr_name.endswith('_'):

                if isinstance(attr_value, types.FunctionType):

                    attrs[attr_name] = cls.deco(attr_value)

        return super(DecoMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def deco(cls, func):

        def wrapper(*args, **kwargs):
            
            result = func(*args, **kwargs)
            
            attributes = inspect.getmembers(cls, lambda variable:not(inspect.isroutine(variable)))
                
            for prop, _ in attributes:
                
                if prop.startswith('_') and prop.endswith('_'):

                    if not(prop.startswith('__') and prop.endswith('__')):

                        delattr(cls, prop)

            return result

        return wrapper