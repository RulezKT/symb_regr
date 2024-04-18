from math import sin, cos, log


def square(x):
    return x * x


def calc_grad(x0):

    return (
        282.64227 * (-0.03858877 + square(cos(3.4575312 + (x0 / -5010.1094))))
    ) - log(square(square(square(sin(x0 / -5010.1094)))))


x0 = -4726914296  # 0.0
print(calc_grad(x0))
x0 = -4718897264  # 90.0
print(calc_grad(x0))
x0 = -4710806338  # 180.0
print(calc_grad(x0))
x0 = -4703045938  # 270.0
print(calc_grad(x0))
