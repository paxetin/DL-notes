import numpy as np


def AND(x):
    w = np.array([0.5, 0.5])
    theta = 0.7
    y = w.dot(x)
    if y <= theta:
        return 0
    else:
        return 1


input1 = np.array([0, 0])
input2 = np.array([0, 1])
input3 = np.array([1, 0])
input4 = np.array([1, 1])

print(f"output1 is {AND(input1)}")
print(f"output2 is {AND(input2)}")
print(f"output3 is {AND(input3)}")
print(f"output4 is {AND(input4)}")