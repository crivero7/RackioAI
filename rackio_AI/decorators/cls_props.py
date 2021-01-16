import inspect

def decorating_meta(decorator):
    class DecoratingMetaclass(type):
        def __new__(self, class_name, bases, namespace):
            for key, value in list(namespace.items()):
                if callable(value):
                    namespace[key] = decorator(value)

            return type.__new__(self, class_name, bases, namespace)

    return DecoratingMetaclass