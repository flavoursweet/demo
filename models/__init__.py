REGISTRY = {}

def register(name):
    def decorate(cls):
        REGISTRY[name] = cls
        return cls
    return decorate
