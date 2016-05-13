#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 2

def read_data():
    import csv, numpy as np
    instances = []
    with open('DataSet3.txt', 'r') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        head = next(dataset)
        for row in dataset:
            instances.append(row)
    instances = np.array(instances).astype(np.float)
    return instances[:, 1], instances[:, 2]


if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt

    # read and split the data
    x, y = read_data()
    xtrain, xtest = np.split(x, 2)
    ytrain, ytest = np.split(y, 2)

    # plot the train and test points
    training_points, = plt.plot(xtrain, ytrain, 'b.', label='Training points')
    testing_points, = plt.plot(xtest, ytest, 'r.', label='Testing points')
    plt.legend(handles = [training_points, testing_points])

    # generate some points to plot the shape of the polynomial
    x_points = np.linspace(min(xtrain), max(xtrain))

    # initialize the vectors that will hold the obtained errors for various
    # degrees of the polynomials
    error_degree_train = []
    error_degree_test = []

    # define some polynomial degrees
    degrees_for_polynomial = [0, 1, 2, 5, 10, 15, 20, 25]

    for deg in degrees_for_polynomial:
        # fit on training data
        fitted_polynomial, error_on_train, _, _, _ = \
            np.polyfit(xtrain, ytrain, deg, full=True)

        # obtain the ys of the polynomial's points
        y_points = np.polyval(fitted_polynomial, x_points)
        # plot the polynomial
        plt.plot(x_points, y_points, label=str(deg) + "th polynomial")

        # evaluate on test data
        predicted_ytest = np.polyval(fitted_polynomial, xtest)
        # compute the quadratic loss
        error_on_test = sum((predicted_ytest[i] - ytest[i]) *
                            (predicted_ytest[i] - ytest[i])
                            for i in range(len(ytest)))

        # save results for further plotting
        error_degree_train.append(error_on_train)
        error_degree_test.append(error_on_test)

    plt.show()

    training_loss, = plt.plot(np.array(degrees_for_polynomial, float),
             np.array(error_degree_train, float), 'b-', label='Training loss')
    testing_loss, = plt.plot(np.array(degrees_for_polynomial, float),
             np.array(error_degree_test, float), 'r-', label="Testing loss")
    plt.legend(handles = [training_loss, testing_loss])
    plt.show()
