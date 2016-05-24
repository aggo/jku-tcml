from scipy import optimize


def f(x):   # The rosenbrock function
   return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def fprime(x):
    import numpy as np
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))

print(fprime([2,2]))
# optimize.fmin_cg(f, [2, 2], fprime=fprime)

w = optimize.minimize(f,
                      [2, 2],
                      method='bfgs',
                      jac=fprime,
                      options={'disp': 1})