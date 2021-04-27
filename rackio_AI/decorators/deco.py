from functools import wraps

def scaler(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        self = args[0]
        x = args[1]
        
        if self.scaler:
            
            x = self.scaler.apply(x)

        y = f(*args, **kwargs)

        if self.scaler:

            y = self.scaler.inverse(y)[0]

        return y

    return decorated