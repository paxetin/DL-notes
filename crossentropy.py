import numpy as np
import matplotlib.pylab as plt

x = np.arange(1e-10, 1.0, 0.01)
y = np.log(x)
plt.plot(x, y, label="logx")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-5.1, 0.1)
plt.title("log functions")
plt.legend()
plt.show()


def cross_entropy_error(y, t):
    delta = 1e-10
    return -np.sum(t*np.log(y+delta))
