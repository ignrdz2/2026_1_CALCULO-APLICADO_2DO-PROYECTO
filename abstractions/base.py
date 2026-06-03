from __future__ import annotations
from abc import ABC, abstractmethod


class Function(ABC):

    @abstractmethod
    def eval(self, x: float) -> float:
        pass

    @abstractmethod
    def derivative(self) -> "Function":
        pass

    # -------------------
    # operadores
    # -------------------

    def __add__(self, other):
        import abstractions.operations as ops
        return ops.Add(self, to_function(other))

    def __radd__(self, other):
        import abstractions.operations as ops
        return ops.Add(to_function(other), self)

    def __sub__(self, other):
        import abstractions.operations as ops
        return ops.Sub(self, to_function(other))

    def __rsub__(self, other):
        import abstractions.operations as ops
        return ops.Sub(to_function(other), self)

    def __mul__(self, other):
        import abstractions.operations as ops
        return ops.Mul(self, to_function(other))

    def __rmul__(self, other):
        import abstractions.operations as ops
        return ops.Mul(to_function(other), self)

    def __truediv__(self, other):
        import abstractions.operations as ops
        return ops.Div(self, to_function(other))

    def __rtruediv__(self, other):
        import abstractions.operations as ops
        return ops.Div(to_function(other), self)

    def __pow__(self, power):
        import abstractions.operations as ops
        return ops.Pow(self, to_function(power))
    
    def __neg__(self):
        from abstractions.functions import Neg
        return Neg(self)


class Operation(Function):

    def __init__(self, f, g):
        self.f = f
        self.g = g


def to_function(obj):
    from abstractions.functions import Const

    if isinstance(obj, Function):
        return obj

    return Const(obj)