import numpy as np
# A = 100
A = 2
x = 10
epsilon = 1e-6
def f(x):
    # return A-x**3
    return A-x**2
def df(x):
    # return - 3 * x ** 2
    return - 2 * x
while np.abs(f(x)) > epsilon:
    x = x - f(x)/df(x)
print(f"{A}的3次方根为{x}")
