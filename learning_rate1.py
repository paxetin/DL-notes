import numpy as np
import matplotlib.pylab as plt
from sympy import Symbol
from sympy.utilities.lambdify import lambdify


def get_der_f():
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    y = x0**2 + (2*x1+1)
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
    return cal_gradient(x)


def gradient_descent(init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = get_gradient(x)
        x -= lr * grad
    return x, np.array(x_history)


# hyperparameter
init_x = np.array([-3.0, 4.0])

lr = 0.01
step_num = 200
x, x_history = gradient_descent(init_x, lr=lr, step_num=step_num)


plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim([-3.5, 3.5])
plt.ylim([-4.5, 4.5])
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()
