from abstractions.base import Operation
from abstractions.functions import Const


def is_const(x, value=None):

    if not isinstance(x, Const):
        return False

    if value is None:
        return True

    return x.value == value


class BinaryOperation(Operation):

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.f == other.f
            and self.g == other.g
        )


class Add(BinaryOperation):

    def eval(self, x):
        return self.f.eval(x) + self.g.eval(x)

    def derivative(self):
        return self.f.derivative() + self.g.derivative()


class Sub(BinaryOperation):

    def eval(self, x):
        return self.f.eval(x) - self.g.eval(x)

    def derivative(self):
        return self.f.derivative() - self.g.derivative()


class Mul(BinaryOperation):

    def eval(self, x):
        return self.f.eval(x) * self.g.eval(x)

    def derivative(self):
        return self.f.derivative() * self.g + self.f * self.g.derivative()


class Div(BinaryOperation):

    def eval(self, x):
        return self.f.eval(x) / self.g.eval(x)

    def derivative(self):
        return (self.f.derivative() * self.g - self.f * self.g.derivative()) / (self.g ** 2)
    

class Pow(BinaryOperation):

    def eval(self, x):
        return self.f.eval(x) ** self.g.eval(x)

    def derivative(self):
        from abstractions.functions import Const, ln
        # caso especial: f(x)^c
        if isinstance(self.g, Const):
            c = self.g.value
            return (
                c
                * (self.f ** (c - 1))
                * self.f.derivative()
            )

        # regla general
        return (
            self
            * (
                self.g.derivative() * ln(self.f)
                + self.g * self.f.derivative() / self.f
            )
        )

