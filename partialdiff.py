# 計算f(x0,x1) = x0*x0 + x1*x1

import numpy as np
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
import matplotlib.pylab as plt


def function_2(x):
    return np.sum(x**2, axis=1)


def get_der_f():
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    y = x0**2+x1**2
    der_x0 = y.diff(x0)
    der_x1 = y.diff(x1)
    # print(f*{der_x0} and {der_x1}")
    f_der_x0 = lambdify(x0, der_x0)
    f_der_x1 = lambdify(x1, der_x1)
    return f_der_x0, f_der_x1


def cal_gradient(x):
    grad = np.zeros_like(x)
    der_f, _ = get_der_f()
    for idx in range(x.size):
        grad[idx] = der_f(x[idx])
    return grad


def get_gradient(x):
    grad = np.zeros_like(x)
    for idx, x in enumerate(x):
        grad[idx] = cal_gradient(x)
    return grad


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = get_gradient(np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="uv")

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid()
    plt.draw()
    plt.show()
