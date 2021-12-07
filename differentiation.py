# 計算y = 0.01x**2+0.1x 數值微分

import numpy as np
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
import matplotlib.pylab as plt


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


x = Symbol('x')
y = 0.01*x**2+0.1*x
der_y = y.diff(x)
print(f"{der_y} and {type(der_y)}")

x_val = np.arange(-10, 20.0, 0.1)
y_val = function_1(x_val)
f_der_y = lambdify(x, der_y)
y_val1 = f_der_y(x_val)

print(f"{f_der_y(-5)}")


def tangent_line(f, der_f, x):
    d = der_f(x)
    y = f(x) - d*x
    return lambda t: d*t + y


tf = tangent_line(function_1, f_der_y, -5)
y_val2 = tf(x_val)
tf1 = tangent_line(function_1, f_der_y, 10)
y_val3 = tf1(x_val)

plt.plot(x_val, y_val, label="y function")
plt.plot(x_val, y_val1, label="y deviation function",
         linestyle="--", color="C1")
plt.plot(x_val, y_val2, label="tangent line (-5)",
         linestyle="dotted", color="C2")
plt.plot(x_val, y_val3, label="tangent line (10)", linestyle="-.", color="C3")
plt.xlabel("x")
plt.ylabel("y")
plt.title("functions")
plt.legend()
plt.show()
