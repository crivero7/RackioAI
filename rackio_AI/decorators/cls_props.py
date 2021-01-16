import inspect


def cls_temp_props(decorator):
    """
    Documentation here
    """

    def decorate(cls):

        for name, fn in inspect.getmembers(cls, inspect.ismethod):

            if not(name.startswith('_') and name.endswith('_')):
            
                setattr(cls, name, decorator(fn))

            # attributes = inspect.getmembers(cls, lambda prop:not(inspect.isroutine(prop)))
        
            # for prop, _ in attributes:
                
            #     if prop.startswith('_') and prop.endswith('_'):

            #         if not(prop.startswith('__') and prop.endswith('__')):

            #             delattr(cls, prop)

        return cls

    return decorate


@cls_temp_props
def del_temp_props(fn): 
    
    def gn(*args, **kwargs):
        
        fn(*args, **kwargs)
        print(args)

    gn.__name__ = fn.__name__ 
    return gn 