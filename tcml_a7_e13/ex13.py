#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 12

import matplotlib.pyplot as plt
import time
from matplotlib import cm
from scipy.optimize import optimize

# Initialize constants
from sklearn import preprocessing

NR_EPOCHS = 100


# ---------------------Helper methods--------------------------------------------

def sigmoid(x):
    '''
    :return: the value of the sigmoid function for the given x (aka the prob. to be a positive example)
    '''
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------Data manip--------------------------------------------

def get_data_and_weights():
    data = np.genfromtxt("dataset-cf10-46.csv", delimiter=',')
    data = data[1:, :]  # remove the first line, the header
    labels = np.copy(data[:, 0])  # extract labels - convert them from 4 and 6 to 0/1
    for i in range(len(labels)):
        if labels[i] == 4:
            labels[i] = 0
        else:
            labels[i] = 1

    data[:, 0] = np.ones((data.shape[0]))  # replace the first column, the labels, with 1 for bias
    data = preprocessing.normalize(data)
    # plotImg(data, 0, title="Normal")

    # set observations on columns
    data = data.T

    nr_weights = data.shape[0]
    w = np.random.rand(nr_weights, 1)

    return data, labels, w


def split_data(inputs, true_labels):
    size = inputs.shape[1]

    train_inputs = inputs[:, :3 / 10 * size]
    val_inputs = inputs[:, 3 / 10 * size:5 / 10 * size]
    test_inputs = inputs[:, 5 / 10 * size:]

    train_labels = true_labels[:3 / 10 * size]
    val_labels = true_labels[3 / 10 * size:5 / 10 * size]
    test_labels = true_labels[5 / 10 * size:]

    xtrain_plus_val = np.hstack((train_inputs, val_inputs))
    ytrain_plus_val = np.hstack((train_labels, val_labels))

    return train_inputs, train_labels, test_inputs, test_labels, val_inputs, val_labels, xtrain_plus_val, ytrain_plus_val


# ---------------------Plotting--------------------------------------------

def plot_train_val_error_vs_epoch(errors_on_train, errors_on_test):
    import matplotlib.pyplot as plt

    epochs_train = len(errors_on_train)
    epochs_test = len(errors_on_test)

    epochs1 = np.linspace(0, epochs_train, epochs_train)
    epochs2 = np.linspace(0, epochs_test, epochs_test)

    fig = plt.figure()
    plt.interactive(False)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.plot(epochs1, errors_on_train, 'b', epochs2, errors_on_test, 'r')
    return plt

# ---------------------Logistic regression--------------------------------------------

def compute_log_likelihood(w, x, y):  # the cost function
    # compute the probability for class 1
    prob_1 = sigmoid(np.dot(w.T, x))
    # compute the value of the log likelihood vector (aka cross entropy error)
    log_likelihood = (y) * np.log(prob_1 + 0.00000001) + (1 - y) * np.log(1 - prob_1 + 0.00000001)
    # this shows how likely is each data observation to appear - the log likelihood of the whole ds is the mean
    return -log_likelihood.sum() / x.shape[0]


def compute_log_likelihood_one_var_w(w):  # the cost function
    # compute the probability for class 1
    prob_1 = sigmoid(np.dot(w.T, xtrain_plus_val))
    # compute the value of the log likelihood vector (aka cross entropy error)
    log_likelihood = (ytrain_plus_val) * np.log(prob_1 + 0.00000001) + (1 - ytrain_plus_val) * np.log(
        1 - prob_1 + 0.00000001)
    # this shows how likely is each data observation to appear - the log likelihood of the whole ds is the mean
    return -log_likelihood.sum() / xtrain_plus_val.shape[0]


def logistic_gradient(w, x, y):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = -np.dot(y - sigmoid(np.dot(w.T, x)), x.T)
    return np.array(gradi[0].reshape(len(w), 1) / x.shape[0])


def logistic_gradient_one_var_w(w):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = -np.dot(ytrain_plus_val - sigmoid(np.dot(w.T, xtrain_plus_val)), xtrain_plus_val.T)
    # if gradi[0] is float:
    #     return np.array(gradi[0] / xtrain_plus_val.shape[0])
    return np.array(gradi/xtrain_plus_val.shape[0])


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
1. Simple
"""


def gradient_descent(learning_rate, w, x, y, xval, yval):
    errors_on_train, errors_on_val = [], []

    for i in range(NR_EPOCHS):
        # compute the gradients
        grad = logistic_gradient(w, x, y)

        # update the weights
        w -= learning_rate * grad

        # compute the errors
        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)
        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)
        # print("Iteration {0}, error tr = {1}, error val = {2}".format(i, error_tr, error_val, w))

    return w, errors_on_train, errors_on_val

# --------------------------CALLBACK FUNCTIONS FOR OPTIMIZE----------------------------------------------
def save_errors(w):
    error_tr = compute_log_likelihood(w, x, y)
    error_val = compute_log_likelihood(w, xval, yval)
    errors_tr.append(error_tr)
    errors_val.append(error_val)

# -------------------------------------------------------------------------------------------------------
import numpy as np


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# get and split data----------------------------------------------------------------------------------------------------
x, y, w = get_data_and_weights()
xtrain, ytrain, xtest, ytest, xval, yval, xtrain_plus_val, ytrain_plus_val = split_data(x, y)

initw = w.copy()

for method in ["BFGS","CG"]:
    global errors_tr, errors_val
    errors_tr = []
    errors_val = []

    from scipy import optimize
    w = initw.copy()
    print(w)
    start = time.clock()
    optimize.minimize(fun=compute_log_likelihood_one_var_w,
                          x0=w,
                          method=method,
                          jac=logistic_gradient_one_var_w,
                          options={'maxiter':NR_EPOCHS},
                          callback=save_errors)
    end = time.clock()
    print(method+" took "+str(end-start)+" seconds.")
    # plt = plot_train_val_error_vs_epoch(errors_tr, errors_val)
    # plt.legend([method+" train error", method+" val error"])

    # fig = plt.figure()
    # fig.savefig(method+ ".png", bbox_inches='tight')

    # print(w)

plt.show()
start = time.clock()
w = initw.copy()
print(w)
w, errors_tr, errors_val = gradient_descent(0.05, w, x, y, xval, yval)
end = time.clock()
plot_train_val_error_vs_epoch(errors_tr, errors_val)
plt.legend(["Grad. desc. " + " train error", "Grad. desc. " + " val error"])
plt.show()

print("Gradient descent took "+str(end-start)+" seconds.")