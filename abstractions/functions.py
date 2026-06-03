import numpy as np

from abstractions.base import Function, to_function


class Const(Function):

    def __init__(self, value):
        self.value = value

    def eval(self, x):
        return self.value

    def derivative(self):
        return Const(0)

    def __eq__(self, other):
        return isinstance(other, Const) and self.value == other.value


class Variable(Function):

    def eval(self, x):
        return x

    def derivative(self):
        return Const(1)

    def __eq__(self, other):
        return isinstance(other, Variable)


class Sin(Function):

    def __init__(self, f):
        self.f = to_function(f)

    def eval(self, x):
        return np.sin(self.f.eval(x))

    def derivative(self):
        return Cos(self.f) * self.f.derivative()

    def __eq__(self, other):
        return isinstance(other, Sin) and self.f == other.f


class Cos(Function):

    def __init__(self, f):
        self.f = to_function(f)

    def eval(self, x):
        return np.cos(self.f.eval(x))

    def derivative(self):
        return -Sin(self.f) * self.f.derivative()

    def __eq__(self, other):
        return isinstance(other, Cos) and self.f == other.f


class Exp(Function):

    def __init__(self, f):
        self.f = to_function(f)

    def eval(self, x):
        return np.exp(self.f.eval(x))

    def derivative(self):
        return Exp(self.f) * self.f.derivative()

    def __eq__(self, other):
        return isinstance(other, Exp) and self.f == other.f


class Ln(Function):

    def __init__(self, f):
        self.f = to_function(f)

    def eval(self, x):
        return np.log(self.f.eval(x))

    def derivative(self):
        return self.f.derivative() / self.f

    def __eq__(self, other):
        return isinstance(other, Ln) and self.f == other.f
    

class Neg(Function):

    def __init__(self, f):
        self.f = to_function(f)

    def eval(self, x):
        return -self.f.eval(x)

    def derivative(self):
        return -self.f.derivative()

    def __eq__(self, other):
        return isinstance(other, Neg) and self.f == other.f


def sin(f):
    return Sin(f)


def cos(f):
    return Cos(f)


def exp(f):
    return Exp(f)


def ln(f):
    return Ln(f)