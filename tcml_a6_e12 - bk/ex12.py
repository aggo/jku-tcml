#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 12


from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

# Initialize constants
from matplotlib import cm
from scipy.optimize import optimize

NR_FEATURES = 6
NR_OBSERVATIONS = 4


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

    train_labels = true_labels[ :3 / 10 * size]
    val_labels = true_labels[ 3 / 10 * size:5 / 10 * size]
    test_labels = true_labels[ 5 / 10 * size:]
    return train_inputs, train_labels, test_inputs, test_labels, val_inputs, val_labels


# ---------------------Plotting--------------------------------------------

def plot_train_val_error_vs_epoch(errors_on_val, errors_on_train, title):
    import matplotlib.pyplot as plt

    epochs_val = len(errors_on_val)
    epochs_train = len(errors_on_train)

    epochs = np.linspace(0, epochs_val, epochs_val)
    epochs2 = np.linspace(0, epochs_train, epochs_train)

    fig = plt.figure()
    plt.interactive(False)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    print(epochs_train)
    print(errors_on_train)
    plt.plot(epochs2, errors_on_train, 'r', epochs, errors_on_val, 'b')
    plt.legend(["Training error", "Validation error"])
    plt.plot()
    plt.title(title)

    fig.savefig(title + ".png", bbox_inches='tight')

def plot_train_val_error_vs_epoch_and_learning_rate(errors_on_val, errors_on_train, learning_rates, title):
    import matplotlib.pyplot as plt

    epochs_val = len(errors_on_val)
    epochs_train = len(errors_on_train)

    epochs = np.linspace(0, epochs_val, epochs_val)
    epochs2 = np.linspace(0, epochs_train, epochs_train)
    epochs3 = np.linspace(0, len(learning_rates), len(learning_rates))

    fig = plt.figure()
    plt.interactive(False)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    print(epochs_train)
    print(errors_on_train)
    plt.plot(epochs2, errors_on_train, 'r', epochs, errors_on_val, 'b', epochs3, learning_rates, 'g')
    # plt.legend(["Training error", "Validation error", "Learning rate * 100"])
    plt.plot()
    plt.title(title)

    fig.savefig(title + ".png", bbox_inches='tight')


# ---------------------Logistic regression--------------------------------------------

def compute_log_likelihood(w, x, y):  # the cost function
    # compute the probability for class 1
    prob_1 = sigmoid(np.dot(w.T, x))
    # compute the value of the log likelihood vector (aka cross entropy error)
    log_likelihood = (y) * np.log(prob_1 +0.00000001) + (1 - y) * np.log(1 - prob_1+0.00000001)
    # this shows how likely is each data observation to appear - the log likelihood of the whole ds is the mean
    return -log_likelihood.sum()/x.shape[0]

def compute_log_likelihood_one_var_w(w):  # the cost function
    # compute the probability for class 1
    prob_1 = sigmoid(np.dot(w.T, xtrain))
    # compute the value of the log likelihood vector (aka cross entropy error)
    log_likelihood = (ytrain) * np.log(prob_1 +0.00000001) + (1 - ytrain) * np.log(1 - prob_1+0.00000001)
    # this shows how likely is each data observation to appear - the log likelihood of the whole ds is the mean
    return -log_likelihood.sum()/xtrain.shape[0]

def logistic_gradient(w, x, y):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = -np.dot(y - sigmoid(np.dot(w.T, x)), x.T)
    return gradi[0].reshape(len(w), 1)/x.shape[0]

def logistic_gradient_one_var_w(w):
    """
    :param w: parameter vector
    :param x: data matrix
    :param y: label vector
    :return: a vector representing the gradient dL/dw
    """
    gradi = -np.dot(ytrain - sigmoid(np.dot(w.T, xtrain)), xtrain.T)
    return gradi[0]/xtrain.shape[0]


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


def gradient_descent(learning_rate, w, x, y, xval, yval):
    epoch = 1
    errors_on_train = []
    errors_on_val = []

    while True:
        # compute the gradients
        grad = logistic_gradient(w, x, y)

        w -= learning_rate * grad

        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)
        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)

        print("Iteration {0}, error tr = {1}, error val = {2}, w = {3}".format(epoch, error_tr, error_val, w))
        if epoch > 100:
            break
        epoch += 1

    return w, errors_on_train, errors_on_val

def gradient_descent_with_momentum(learning_rate, w, x, y, xval, yval, u):
    epoch = 1
    errors_on_train = []
    errors_on_val = []

    while True:
        # compute the gradients
        grad = logistic_gradient(w, x, y)
        if epoch == 1:
            w -= learning_rate * grad
        else:
            w -= learning_rate * (grad+u*oldgrad)
        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)
        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)

        print("Iteration {0}, error tr = {1}, error val = {2}, w = {3}".format(epoch, error_tr, error_val, w))
        if epoch > 100:
            break
        epoch += 1
        oldgrad = grad

    return w, errors_on_train, errors_on_val


def gradient_descent_with_momentum_and_adjusted_lr(learning_rate, w, xtrain, ytrain, xval, yval, u):
    epoch = 1
    errors_on_train = []
    errors_on_val = []
    learning_rates = []
    adjustment_index = 0

    learning_rate_adjustment_factor = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    how_many_times_subtract_curent_learning_rate = 3

    while True:
        # compute the gradients
        grad = logistic_gradient(w, x, y)
        if epoch == 1:
            w -= learning_rate * grad
        else:
            w -= learning_rate * (grad + u * oldgrad)

        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)

        if epoch >= 2:
            # detect when the error rate starts going up (after the 20th epoch)

            if error_val >= errors_on_val[-1]:
                # adjust slowly the learning rate
                learning_rate -= learning_rate_adjustment_factor[adjustment_index]
                if adjustment_index < len(learning_rate_adjustment_factor)-1 and how_many_times_subtract_curent_learning_rate==0:
                    adjustment_index += 1
                    how_many_times_subtract_curent_learning_rate = 3
                how_many_times_subtract_curent_learning_rate-=1

        learning_rates.append(learning_rate*10) # multiply with 10 for easier visualization

        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)

        print("Iteration {0}, error tr = {1}, error val = {2}, w = {3}".format(epoch, error_tr, error_val, w))
        if epoch > 100:
            break
        epoch += 1
        oldgrad = grad

    return w, errors_on_train, errors_on_val, learning_rates

def gradient_descent_with_momentum_and_line_search(learning_rate, w, xtrain, ytrain, xval, yval, u):
    epoch = 1
    errors_on_train = []
    errors_on_val = []
    learning_rates = []

    while True:
        # initial guess for the minimum position = w
        # direction of the minimum given by the negative gradient (lecture notes, page 83)

        grad = logistic_gradient(w, x, y)

        res = optimize.line_search(compute_log_likelihood_one_var_w, logistic_gradient_one_var_w, w, -grad)
        if res[0]is not None:
            learning_rate = res[0]
        else:
            print("Did not converge")

        print("Found LR = {0}".format(learning_rate))

        if epoch == 1:
            w -= learning_rate * grad
        else:
            w -= learning_rate * (grad + u * oldgrad)

        error_tr = compute_log_likelihood(w, x, y)
        error_val = compute_log_likelihood(w, xval, yval)

        learning_rates.append(learning_rate*10) # multiply with 10 for easier visualization

        errors_on_train.append(error_tr)
        errors_on_val.append(error_val)

        print("Iteration {0}, error tr = {1}, error val = {2}, w = {3}".format(epoch, error_tr, error_val, w))
        if epoch > 100:
            break
        epoch += 1
        oldgrad = grad



    return w, errors_on_train, errors_on_val, learning_rates

#-------------------------------------------------------------------------------------------------------
import numpy as np

x, y, w = get_data_and_weights()
xtrain, ytrain, xtest, ytest, xval, yval, = split_data(x, y)

learning_rates = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 1]

min_error = 1000000000
min_u = 1

# part (1): finding a learning rate that gives good results
for learning_rate in learning_rates:
    [w, errors_on_train, errors_on_val] = gradient_descent(learning_rate, w, xtrain, ytrain, xval, yval)
    plot_train_val_error_vs_epoch(errors_on_val, errors_on_train,
                                  "Learning rate = {0}".format(learning_rate))
    if errors_on_val[-1] < min_error:
        min_error = errors_on_val[-1]
        min_lr = learning_rate
        min_w = w


min_error = 10000000
# part (2): add a momentum term
momentum_terms = [0.5, 0.6, 0.7, 0.8, 0.9]
for momentum_term in momentum_terms:
    [w, errors_on_train, errors_on_val] = gradient_descent_with_momentum(learning_rate, w, xtrain, ytrain, xval, yval, momentum_term)
    plot_train_val_error_vs_epoch(errors_on_val, errors_on_train,
                                  "LR = {0} Momentum = {1}".format(min_lr, momentum_term))
    if errors_on_val[-1] < min_error:
        min_error = errors_on_val[-1]
        min_u = momentum_term

# part (3): learning rate schedule
[w, errors_on_train, errors_on_val, lrs] = gradient_descent_with_momentum_and_adjusted_lr(0.1, w, xtrain, ytrain, xval, yval, min_u)
print(lrs)
plot_train_val_error_vs_epoch_and_learning_rate(errors_on_val, errors_on_train, lrs,
                              "Lr = {1}, Mom. term = {0}. Scheduled learning rate adjusting".format(min_u, 0.1))

# part (3-2): learning rate schedule - now with train and test not train and val
xtrain = np.hstack((xtrain, xval))
ytrain = np.hstack((ytrain, yval))
[w, errors_on_train, errors_on_val, lrs] = gradient_descent_with_momentum_and_adjusted_lr(min_lr, w, xtrain, ytrain,
                                                                                          xtest, ytest, min_u)
print(lrs)
plot_train_val_error_vs_epoch_and_learning_rate(errors_on_val, errors_on_train, lrs,
                                                "Lr = {1}, Mom. term = {0}. LR adj. Train-Test".format(
                                                    min_u, min_lr))

# part (4): line_search
xtrain = np.hstack((xtrain, xval))
ytrain = np.hstack((ytrain, yval))
[w, errors_on_train, errors_on_val, lrs] = gradient_descent_with_momentum_and_line_search(min_lr, w, xtrain, ytrain,
                                                                                          xtest, ytest, min_u)
print(lrs)
plot_train_val_error_vs_epoch_and_learning_rate(errors_on_val, errors_on_train, lrs,
                                                "LS-Lr = {1}, Mom = {0}. Train-Test".format(
                                                    min_u, min_lr))



