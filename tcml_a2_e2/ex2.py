#!/usr/bin/python
# Author: Amalia Ioana Goia
# Matr. Nr.: k1557854
# Exercise 2
from pprint import pprint

import math

def read_data(filename):
    import csv, numpy as np
    instances = []
    with open(filename, 'r') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        for row in dataset:
            instances.append(row)
    instances = np.array(instances).astype(np.float)
    size = len(row)
    return instances[:, :size-1], instances[:, size-1], size

if __name__ == "__main__":
    import numpy as np

    # read and split the data
    x, y, size = read_data('DataSet1.txt')
    x_plus = np.array([x[i] for i in range(len(x)) if y[i] == +1])
    x_minus = np.array([x[i] for i in range(len(x)) if y[i] == -1])

    # x needs to be reshaped: first array containing the first feature for all points
    # and second array the second feature - for being able to compute the mean/covariance.
    # (i.e. each row represents a variable (x1, x2 here) and each column an observation
    # Do this by simply transposing.
    x_plus = np.transpose(x_plus)
    x_minus = np.transpose(x_minus)

    # compute the mean vectors - use axis = 1 to compute the mean of each feature
    # Ex:
    # a = np.array([[1, 2], [3, 4]])
    # >>> np.mean(a, axis=1)
    # array([ 1.5,  3.5])

    mean_x_plus = np.mean(x_plus, axis=1) # [meanOfFirstFeature, meanOfSecondFeature]
    mean_x_minus= np.mean(x_minus, axis=1)
    print("Means:")
    print(mean_x_plus)
    print(mean_x_minus)

    # compute the covariance matrices
    #  - cov[i,j] = correlation between the
    #    ith entry of first feature and the
    #    jth entry of second feature

    cov_matrix_xplus = np.cov(x_plus)
    cov_matrix_xminus = np.cov(x_minus)
    print("Covariance matrix xplus:")
    pprint(cov_matrix_xplus)
    print("Covariance matrix xminus:")
    pprint(cov_matrix_xminus)

    # compute the determinants of the covariance matrices
    det_cov_matrix_xplus = np.linalg.det(cov_matrix_xplus)
    det_cov_matrix_xminus = np.linalg.det(cov_matrix_xminus)

    print("Determinants:")
    print(det_cov_matrix_xminus)
    print(det_cov_matrix_xplus)

    # compute the inverses of the covariance matrices
    #  - assume they are non-singular for inverse to exist
    if det_cov_matrix_xplus!=0:
        inv_cov_matrix_xplus = np.linalg.inv(cov_matrix_xplus)
    if det_cov_matrix_xminus!=0:
        inv_cov_matrix_xminus = np.linalg.inv(cov_matrix_xminus)

    # Compute p(y=+1) and p(y=-1)
    p_plus = sum(1 for label in y if label == +1)
    p_minus = len(y)-p_plus
    p_plus /= len(y)
    p_minus /= len(y)

    print("Probabilities (plus and minus):")
    print(p_plus)
    print(p_minus)


    # optimal classification function value in each point computation
    # initialize array holding the function's value in each point
    g = [0 for i in range(len(x))]
    # for each point, compute
    for i in range(len(x)):
        xi = x[i]
        g[i] = -1/2*np.mat(xi)* np.mat(inv_cov_matrix_xplus-inv_cov_matrix_xminus)*np.transpose(np.mat(xi)) + \
                np.mat(xi) * (np.mat(inv_cov_matrix_xplus)*np.transpose(np.mat(mean_x_plus)) - \
                              np.mat(inv_cov_matrix_xminus)* np.transpose(np.mat(mean_x_minus))) - \
                1/2 * np.mat(mean_x_plus)*np.mat(inv_cov_matrix_xplus)* np.transpose(np.mat(mean_x_plus)) + \
                1/2 * np.mat(mean_x_minus)*np.mat(inv_cov_matrix_xminus)*np.transpose(np.mat(mean_x_minus)) - \
                1/2 * math.log(det_cov_matrix_xplus) + 1/2 * math.log(det_cov_matrix_xminus) + \
                math.log(p_plus) - math.log(p_minus)

    # prepare arrays for visualization
    g=[i.tolist()[0][0] for i in g]
    x_plus_coordx = [x[i,0] for i in range(len(x)) if y[i]==1]
    x_plus_coordy = [x[i,1] for i in range(len(x)) if y[i]==1]
    xx_plus_coordx, xx_plus_coordy = np.meshgrid(x_plus_coordx, x_plus_coordy)

    x_minus_coordx = [x[i,0] for i in range(len(x)) if y[i]==-1]
    x_minus_coordy = [x[i,0] for i in range(len(x)) if y[i]==-1]
    xx_minus_coordx, xx_minus_coordy = np.meshgrid(x_minus_coordx, x_minus_coordy)

    z_plus = [g[i] for i in range(len(y)) if y[i]==1]
    zz_plus = np.meshgrid(z_plus)
    z_minus = [g[i] for i in range(len(y)) if y[i]==-1]
    zz_minus = np.meshgrid(z_minus)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x_plus_coordx,x_plus_coordy, z_plus,color='k')
    # ax.scatter(x_minus_coordx,x_minus_coordy, z_minus, color='r')

    # visualize the two classes
    ax.plot_surface(xx_plus_coordx, xx_plus_coordy, zz_plus, rstride=4, cstride=4, color='b')
    ax.plot_surface(xx_minus_coordx, xx_minus_coordy, zz_minus, rstride=4, cstride=4, color='r')\

    plt.show()

    # generate 10000 sample points from the first distribution
    plus_points = np.random.multivariate_normal(mean_x_plus, cov_matrix_xplus, 10000)
    # generate 10000 sample points from the second distribution
    minus_points = np.random.multivariate_normal(mean_x_minus, cov_matrix_xminus, 10000)

    # compute risk by evaluating g in all the points: if result has the same sign as the true sign,
    # (that of the distribution that it was generated from), then no loss. Otherwise, loss.
    loss = 0
    for xi in plus_points:
        computed_sign = -1/2*np.mat(xi)* np.mat(inv_cov_matrix_xplus-inv_cov_matrix_xminus)*np.transpose(np.mat(xi)) + \
                np.mat(xi) * (np.mat(inv_cov_matrix_xplus)*np.transpose(np.mat(mean_x_plus)) - \
                              np.mat(inv_cov_matrix_xminus)* np.transpose(np.mat(mean_x_minus))) - \
                1/2 * np.mat(mean_x_plus)*np.mat(inv_cov_matrix_xplus)* np.transpose(np.mat(mean_x_plus)) + \
                1/2 * np.mat(mean_x_minus)*np.mat(inv_cov_matrix_xminus)*np.transpose(np.mat(mean_x_minus)) - \
                1/2 * math.log(det_cov_matrix_xplus) + 1/2 * math.log(det_cov_matrix_xminus) + \
                math.log(p_plus) - math.log(p_minus)
        if computed_sign<0:
            loss+=1

    for xi in minus_points:
        computed_sign = -1/2*np.mat(xi)* np.mat(inv_cov_matrix_xplus-inv_cov_matrix_xminus)*np.transpose(np.mat(xi)) + \
                np.mat(xi) * (np.mat(inv_cov_matrix_xplus)*np.transpose(np.mat(mean_x_plus)) - \
                              np.mat(inv_cov_matrix_xminus)* np.transpose(np.mat(mean_x_minus))) - \
                1/2 * np.mat(mean_x_plus)*np.mat(inv_cov_matrix_xplus)* np.transpose(np.mat(mean_x_plus)) + \
                1/2 * np.mat(mean_x_minus)*np.mat(inv_cov_matrix_xminus)*np.transpose(np.mat(mean_x_minus)) - \
                1/2 * math.log(det_cov_matrix_xplus) + 1/2 * math.log(det_cov_matrix_xminus) + \
                math.log(p_plus) - math.log(p_minus)
        if computed_sign>0:
            loss+=1

    print("Generalization error is "+str(loss))