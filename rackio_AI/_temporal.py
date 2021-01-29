from easy_deco.del_temp_attr import del_temp_attr, set_to_methods

# def del_temp_attr(fn):
#     """
#     Decorator to delete all temporary attributes generated in each pipeline component

#     **Parameters**

#     * **:param fn:** (Function) function to be decorated
#     """
#     def wrapper(*args, **kwargs):

#         cls = args[0]

#         for instance in getattr(cls, "_instances"):
            
#             for attr in list(instance.__dict__.keys()):
  
#                 if attr.startswith('_') and attr.endswith('_'):
    
#                     if not(attr.startswith('__') and attr.endswith('__')):
                        
#                         delattr(instance, attr)

#         return result

#     return wrapper

# def set_to_methods(decorator):

#     def decorate(cls):

#         attrs = inspect.getmembers(cls, predicate=lambda x: inspect.isroutine(x))

#         for attr, _ in attrs:
            
#             if not attr.startswith('_'):

#                 setattr(cls, attr, decorator(getattr(cls, attr)))
        
#         return cls

#     return decorate


@set_to_methods(del_temp_attr)
class TemporalMeta:
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """
    _instances = list()

    def __new__(cls):
    
        inst = super(TemporalMeta, cls).__new__(cls)
        cls._instances.append(inst)

        return inst
