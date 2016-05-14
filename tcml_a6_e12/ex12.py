#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 12

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import optimize

# Initialize constants
NR_EPOCHS = 100


# ---------------------Helper methods--------------------------------------------

def plotImg(x, index, title):
    x = x[index, 1:]  # needed 1: to remove the bias

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_ylim([32, 0])
    ax.pcolor(x.reshape(32, 32), cmap=cm.gray)

    ax.grid(False)
    ax.axis("off")

    fig.subplots_adjust(left=0, top=1, bottom=0, right=1)
    fig.savefig(title + str(index) + ".png", bbox_inches='tight')


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
    plotImg(data, 0, title="Normal")

    # set observations on columns
    data = data.T

    nr_weights = data.shape[0]
    w = np.ones((nr_weights, 1))

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

def plot_train_test_error_vs_epoch(errors_on_train, errors_on_test, title, legend1, legend2):
    import matplotlib.pyplot as plt

    epochs_train = len(errors_on_train)
    epochs_test = len(errors_on_test)

    epochs1 = np.linspace(0, epochs_train, epochs_train)
    epochs2 = np.linspace(0, epochs_test, epochs_test)

    fig = plt.figure()
    plt.interactive(False)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    print(epochs_train)
    print(errors_on_train)
    plt.plot(epochs1, errors_on_train, 'b', epochs2, errors_on_test, 'r')
    plt.legend([legend1, legend2])
    plt.plot()
    plt.title(title)

    fig.savefig(title + ".png", bbox_inches='tight')


def plot_train_test_error_and_lr_vs_epoch(errors_on_train, errors_on_test, learning_rates, title, legend1, legend2):
    import matplotlib.pyplot as plt

    epochs_train = len(errors_on_train)
    epochs_test = len(errors_on_test)
    epochs_lr = len(learning_rates)

    epochs1 = np.linspace(0, epochs_train, epochs_train)
    epochs2 = np.linspace(0, epochs_test, epochs_test)
    epochs3 = np.linspace(0, epochs_lr, epochs_lr)

    fig = plt.figure()
    plt.interactive(False)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    plt.plot(epochs1, errors_on_train, 'b', epochs2, errors_on_test, 'r', epochs3, learning_rates, 'g')
    plt.legend([legend1, legend2, "Lr*10"])
    plt.plot()
    plt.title(title)

    fig.savefig(title + ".png", bbox_inches='tight')


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
    return gradi[0].reshape(len(w), 1) / x.shape[0]


def logistic_gradient_one_var_w(w):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = -np.dot(ytrain_plus_val - sigmoid(np.dot(w.T, xtrain_plus_val)), xtrain_plus_val.T)
    return gradi[0] / xtrain_plus_val.shape[0]


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
        print("Iteration {0}, error tr = {1}, error val = {2}".format(i, error_tr, error_val, w))

    return w, errors_on_train, errors_on_val


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. With momentum
"""


def gradient_descent_with_momentum(learning_rate, w, x, y, xval, yval, u):
    errors_on_train, errors_on_val = [], []

    for epoch in range(NR_EPOCHS):
        # compute the gradients
        grad = logistic_gradient(w, x, y)

        # update the weights considering the momentum term
        if epoch == 0:
            w -= learning_rate * grad
        else:
            w -= learning_rate * (grad + u * oldgrad)

        # compute the errors
        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)
        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)
        print("Iteration {0}, error tr = {1}, error val = {2}".format(epoch, error_tr, error_val, w))

        oldgrad = grad

    return w, errors_on_train, errors_on_val


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. With momentum and adjusted learning rate
"""


def gradient_descent_with_momentum_and_adjusted_lr(learning_rate, w, x, y, xval, yval, u):
    errors_on_train, errors_on_val, learning_rates = [], [], []

    SIGMA = 0.5  # = learning rate adjustment factor when the risk increases - being much smaller than 1 manages to reduce
    # too large learning rates immediately
    RO = 1.1  # learning rate adjustment factor when the risk increases - leads to a sensible increase in the LR

    for epoch in range(NR_EPOCHS):
        # compute the gradients
        grad = logistic_gradient(w, x, y)

        # update the weights considering the momentum term
        if epoch == 0:
            w -= learning_rate * grad
        else:
            w -= learning_rate * (grad + u * oldgrad)

        """
            Perform adjustment of the learning rate. When the error increases, reduce it to half,
            when the error decreases, sensibly increase it. Only do this when the error drops under two.
            (from plots LR 0.05 and 1)
        """
        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)

        if epoch >= 2:
            # detect the change of the risk (aka error) (lecture notes, page 80)
            risk_change = error_val - errors_on_val[-1]
            # if positive, decrease learning rate
            if risk_change > 0:
                learning_rate *= SIGMA
            else:
                learning_rate *= RO

        learning_rates.append(learning_rate * 10)  # multiply with 10 for easier visualization
        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)
        print("Iteration {0}, error tr = {1}, error val = {2}, LR = {3}".format(epoch, error_tr, error_val,
                                                                                learning_rate))

        oldgrad = grad

    return w, errors_on_train, errors_on_val, learning_rates


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. With line search
"""


def gradient_descent_with_momentum_and_line_search(learning_rate, w, x, y, xval, yval, u):
    errors_on_train, errors_on_val, learning_rates = [], [], []

    for epoch in range(NR_EPOCHS):
        # compute the gradients
        grad = logistic_gradient(w, x, y)

        """
            line search to find a good learning rate
            # compute_log_likelihood = the function whose minimum needs to be found
            # logistic_gradient = its derivative wrt w
            # w = initial guess of the location of the minimum
            # -grad = direction of the minimum
        """
        res = optimize.line_search(compute_log_likelihood_one_var_w, logistic_gradient_one_var_w, w, -grad)
        if res[0] is not None:
            learning_rate = res[0]
        else:
            print("Did not converge")

        print("Found LR = {0}".format(learning_rate))

        # update the weights considering the momentum term
        if epoch == 0:
            w -= learning_rate * grad
        else:
            w -= learning_rate * (grad + u * oldgrad)

        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)

        learning_rates.append(learning_rate * 10)  # multiply with 10 for easier visualization
        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)
        print("Iteration {0}, error tr = {1}, error val = {2}, LR = {3}".format(epoch, error_tr, error_val,
                                                                                learning_rate))

        oldgrad = grad

    return w, errors_on_train, errors_on_val, learning_rates


# -------------------------------------------------------------------------------------------------------
import numpy as np


def find_good_learning_rate(xtrain, ytrain, xtest, ytest, w, legend1, legend2):
    learning_rates = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 1]
    min_error = 1000000000

    # part (1): finding a learning rate that gives good results
    for learning_rate in learning_rates:
        [w, errors_on_train, errors_on_test] = gradient_descent(learning_rate, w, xtrain, ytrain, xtest, ytest)
        plot_train_test_error_vs_epoch(errors_on_train, errors_on_test,
                                       "Learning rate = {0}".format(learning_rate), legend1, legend2)
        if errors_on_test[-1] < min_error:
            min_error = errors_on_test[-1]
            min_lr = learning_rate

    return min_lr


def find_momentum_term(best_lr, xtrain, ytrain, xtest, ytest, w, legend1, legend2):
    min_error = 10000000
    momentum_terms = [0.5, 0.6, 0.7, 0.8, 0.9]

    for momentum_term in momentum_terms:
        [w, errors_on_train, errors_on_test] = gradient_descent_with_momentum(best_lr, w, xtrain, ytrain, xval,
                                                                              yval, momentum_term)
        plot_train_test_error_vs_epoch(errors_on_train, errors_on_test,
                                       "LR = {0} Mom. = {1}".format(best_lr, momentum_term), legend1, legend2)
        if errors_on_test[-1] < min_error:
            min_error = errors_on_test[-1]
            min_u = momentum_term

    return min_u


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# get and split data----------------------------------------------------------------------------------------------------
x, y, w = get_data_and_weights()
xtrain, ytrain, xtest, ytest, xval, yval, xtrain_plus_val, ytrain_plus_val = split_data(x, y)

# part (1): find learning rate that gives good results on val-----------------------------------------------------------
best_lr = find_good_learning_rate(xtrain, ytrain, xval, yval, w, "Err. train", "Err. val.")
# find estimate of the generalization error by running on train+test now
[w, errors_on_train, errors_on_test] = gradient_descent(best_lr, w, xtrain_plus_val, ytrain_plus_val, xtest, ytest)
plot_train_test_error_vs_epoch(errors_on_train, errors_on_test,
                               "Best learning rate = {0}".format(best_lr), "Err. on train+val", "Err. on test")

# part (2): add momentum term ------------------------------------------------------------------------------------------
best_u = find_momentum_term(best_lr, xtrain, ytrain, xval, yval, w, "Err. train", "Err. val.")
# find estimate of the generalization error by running on train+test now
[w, errors_on_train, errors_on_test] = gradient_descent_with_momentum(best_lr, w, xtrain_plus_val, ytrain_plus_val,
                                                                      xtest, ytest, best_u)
plot_train_test_error_vs_epoch(errors_on_train, errors_on_test,
                               "Best LR = {0}, best u = {1}".format(best_lr, best_u), "Err. on train+val",
                               "Err. on test")

# part (3): learning rate schedule--------------------------------------------------------------------------------------
[w, errors_on_train, errors_on_val, lrs] = gradient_descent_with_momentum_and_adjusted_lr(best_lr, w, xtrain, ytrain,
                                                                                          xval, yval, best_u)
plot_train_test_error_and_lr_vs_epoch(errors_on_val, errors_on_train, lrs,
                                      "Mom. term = {0}. Scheduled LR adj. on train&val".format(best_u), "Err. train",
                                      "Err. val.")
# find estimate of the generalization error by running on train+test now
[w, errors_on_train, errors_on_val, lrs] = gradient_descent_with_momentum_and_adjusted_lr(best_lr, w, xtrain_plus_val,
                                                                                          ytrain_plus_val, xtest, ytest,
                                                                                          best_u)
plot_train_test_error_and_lr_vs_epoch(errors_on_val, errors_on_train, lrs,
                                      "Mom. term = {0}. Scheduled LR adj. on train+val&test".format(best_u),
                                      "Err. train+val",
                                      "Err. test.")

# part (4): line_search ------------------------------------------------------------------------------------------------
[w, errors_on_train, errors_on_val, lrs] = gradient_descent_with_momentum_and_line_search(best_lr, w, xtrain_plus_val,
                                                                                          ytrain_plus_val, xtest, ytest,
                                                                                          best_u)

plot_train_test_error_and_lr_vs_epoch(errors_on_val, errors_on_train, lrs,
                                      "Mom. term = {0}. Line search LR. Train+val&test".format(best_u),
                                      "Err. train+val",
                                      "Err. test.")
