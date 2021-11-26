import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_function(x):
    return sigmoid(x) * (1 - sigmoid(x))


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    y1 = sigmoid(x)
    y2 = sigmoid_function(x)

    plt.plot(x, y, label="step function")
    plt.plot(x, y1, label="sigmoid function", linestyle="dashed", color="C1")
    plt.plot(x, y2, label="sigmoid function", linestyle="dotted", color="C2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-0.1, 1.1)
    plt.title("activation functions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
