import math


# Automatic differentiation: single variable function
def ad(f, x):
    return f(DualNumber(x, 1))


# Automatic differentiation: multi variable function, compute the partial derivatives with respect to x[i]
def ad_multi(f, x, i):
    n = len(x)
    x_dual = [DualNumber(0, 0) for _ in range(0, n)]
    for k in range(0, n):
        x_dual[k].real = x[k]
        if k == i:
            x_dual[k].dual = 1
    return f(x_dual)


# Automatic differentiation: multi variable function, compute the partial derivatives with respect to x[i]
# f has additional variable number arguments (args).
def ad_multi_params(f, t, x, i, args):
    n = len(x)
    x_dual = [DualNumber(0, 0) for _ in range(0, n)]
    for k in range(0, n):
        x_dual[k].real = x[k]
        if k == i:
            x_dual[k].dual = 1
    return f(t, x_dual, *args)


class DualNumber:
    real: float = None
    dual: float = None

    # Methods
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __repr__(self):
        # This method overrides the behaviour of print() built-in function
        return str(self.real) + "+" + str(self.dual) + "\u03B5"

    # Operator Overload
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, float) or isinstance(other, int):
            return DualNumber(self.real + other, self.dual)
        else:
            pass

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, float) or isinstance(other, int):
            return DualNumber(self.real - other, self.dual)
        else:
            pass

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real,
                              self.real * other.dual + self.dual * other.real)
        elif isinstance(other, float) or isinstance(other, int):
            return DualNumber(self.real * other, self.dual * other)
        else:
            pass

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real / other.real,
                              (self.dual * other.real - self.real * other.dual) / other.dual / other.dual)
        elif isinstance(other, float) or isinstance(other, int):
            return DualNumber(self.real / other, self.dual / other)
        else:
            pass

    def __rtruediv__(self, other):
        return DualNumber(other, 0).__truediv__(self)

    def __pow__(self, power, modulo=None):
        if isinstance(power, float) or isinstance(power, int):
            return DualNumber(self.real ** power,
                              self.dual * power * self.real ** (power - 1))
        else:
            pass

# TODO: OP. overload: exceptions might be handled vie function decorators
# TODO: OP. overload: else branch (pass) -> what to do if "other" is not a numeric variable

# functions

def sin(self):
    return DualNumber(math.sin(self.real), self.dual * math.cos(self.real))


def cos(self):
    return DualNumber(math.cos(self.real), -self.dual * math.sin(self.real))

# TODO: add pow(), sqrt(), tan(), ctan(), atan(), actan(), exp(), log(), log10()
