import scipy as sp
import scipy.optimize
import numpy as np

def test_func(x):
    return (x[0])**2+(x[1])**2

def test_grad(x):
    return [2*x[0],2*x[1]]

res = sp.optimize.line_search(test_func,test_grad,np.array([1.8,1.7]),np.array([-10.,-10.]))
print(res)