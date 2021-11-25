import numpy as np


def AND(x):
    w = np.array([0.5, 0.5])
    theta = -0.7
    y = sum(w*x) + theta
    if y <= theta:
        return 0
    else:
        return 1


def OR(x):
    w = np.array([0.5, 0.5])
    theta = -0.2
    y = sum(w*x) + theta
    if y <= 0:
        return 0
    else:
        return 1


def NAND(x):
    w = np.array([-0.5, -0.5])
    theta = -0.7
    y = sum(w*x) + theta
    if y <= 0:
        return 0
    else:
        return 1


def XOR(x):
    s1 = NAND(x)
    s2 = OR(x)
    tmp = np.array([s1, s2])
    y = AND(tmp)
    return y


input1 = np.array([0, 0])
input2 = np.array([0, 1])
input3 = np.array([1, 0])
input4 = np.array([1, 1])

print(f"output1 is {XOR(input1)}")
print(f"output2 is {XOR(input2)}")
print(f"output3 is {XOR(input3)}")
print(f"output4 is {XOR(input4)}")
