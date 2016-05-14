from scipy.optimize.linesearch import line_search
import numpy as np

def f(x):
    return -x**2

def fprime(x):
    return -2*x

res = line_search(f, fprime, np.array([2.0, 2.0]), np.array([1.0, 1.0]))

print(res)