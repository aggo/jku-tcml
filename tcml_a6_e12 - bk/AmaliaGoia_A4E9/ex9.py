#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 8


from pprint import pprint
import numpy as np

# Initialize constants
from sklearn import linear_model
from sklearn.linear_model.logistic import _logistic_grad_hess

NR_FEATURES = 6
NR_OBSERVATIONS = 4


# ---------------------Helper methods--------------------------------------------

def get_random_row_vector(how_many_columns, ip=False):
    rv = np.random.rand(1, how_many_columns)
    if (ip):
        print("rv is {0} and has shape {1}".format(rv, rv.shape))
    return rv


def get_random_column_vector(how_many_rows, ip=False):
    rv = get_random_row_vector(how_many_rows)
    rv = np.transpose(rv)
    if (ip):
        print("rv is {0} and has shape {1}".format(rv, rv.shape))
    return rv


def get_data_matrix_with_columns_as_observations(nr_features=NR_FEATURES, nr_observations=NR_OBSERVATIONS, ip=False):
    mat = np.random.rand(nr_features, nr_observations)
    if ip:
        print("mat is {0} and has shape {1}".format(mat, mat.shape))
    return mat


def generate_random_data(nr_features=NR_FEATURES, nr_observations=NR_OBSERVATIONS):
    x = get_data_matrix_with_columns_as_observations()
    y = np.random.randint(0, 2, size=(nr_observations, 1))
    for i in range(y.size):
        if y[i] == 0:
            y[i] = -1
    w = get_random_column_vector(nr_features)

    pprint("X is {0} and has shape {1}".format(x, x.shape))
    pprint("y is {0} and has shape {1}".format(y, y.shape))
    pprint("w is {0} and has shape {1}".format(w, w.shape))
    return x, y, w


def sigmoid(x):
    '''
    :return: the value of the sigmoid function for the given x (aka the prob. to be a positive example)
    '''
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------------------------------------------


def compute_log_likelihood(w, x, y):  # the cost function
    # compute the probability for class 1
    prob_1 = sigmoid(np.dot(w.T, x))
    # compute the value of the log likelihood vector (aka cross entropy error)
    log_likelihood = (y) * np.log(prob_1) + (1 - y) * np.log(1 - prob_1)
    # this shows how likely is each data observation to appear - the log likelihood of the whole ds is the mean
    return -log_likelihood.sum()


def logistic_gradient(w, x, y):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = np.dot(y - sigmoid(np.dot(w.T, x)), x.T)
    return -np.sum(gradi, axis=0).reshape(len(w), 1)


def numerical_gradient(w, x, y):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: the gradient dL/dw computed using the central difference quotient
    """
    eps = 1 / 1000

    grad_approx = np.zeros(w.shape)

    for i in range(len(w)):
        current_ei = np.zeros(w.size).reshape(w.size, 1)
        current_ei[i] = eps
        L_plus_eps = compute_log_likelihood(w + current_ei, x, y)
        L_minus_eps = compute_log_likelihood(w - current_ei, x, y)
        grad_approx[i] = (L_plus_eps - L_minus_eps) / (2 * eps)
    return grad_approx


if __name__ == "__main__":
    import numpy as np

    x, y, w = generate_random_data(NR_FEATURES, NR_OBSERVATIONS)

    print("Numerically computed gradient:")
    ng = numerical_gradient(w, x, y)
    pprint(ng)

    print("Gradient with formula:")
    lg = logistic_gradient(w, x, y)
    pprint(lg)

    print('Difference in computed gradients:')
    print(ng - lg)

    # y = y.reshape(1,y.size)[0]
    # w = w.reshape(1,w.size)[0]
    # print(y)
    # print(w)
    # print(_logistic_grad_hess(X = x, y = y.T, w = w, alpha = 0.05))
