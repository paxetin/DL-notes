import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)

    plt.plot(x, y, label="relu function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-0.1, 6)
    plt.title("activation functions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
