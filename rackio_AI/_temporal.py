from easy_deco.del_temp_attr import del_temp_attr, set_to_methods


# @set_to_methods(del_temp_attr)
class TemporalAttrs:
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """
    _instances = list()

    def __init__(self):
        
        TemporalAttrs._instances.append(self)
        

