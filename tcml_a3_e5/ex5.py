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
    return instances[:, :size - 1], instances[:, size - 1], size


def calculate_variance_of_sample(sample, estimator_a_denominator):
    mean_of_sample = np.mean(sample)
    return (1 / estimator_a_denominator) * sum((mean_of_sample - sample[i]) ** 2 for i in range(len(sample)))


if __name__ == "__main__":
    import numpy as np

    # Initialize constants
    NR_EXPERIMENTS = 10000
    N_VALUES = [10, 1000]
    MEAN = 5
    VARIANCE = 4

    estimator_a_values = []
    estimator_b_values = []

    for n in N_VALUES:
        print("When N is " + str(n) + ":")
        for experiment in range(NR_EXPERIMENTS):
            #  take a sample of size n from a gaussian distribution with MEAN and STDEV
            #  loc = the mean and scale = the standard deviation (sqrt of variance)
            sample = np.random.normal(loc=MEAN, scale=math.sqrt(VARIANCE), size=n)
            # print("Mean is "+str(np.mean(sample))+" and variance is "+str(np.var(sample)))

            # define the denominator for each variance estimator formula
            estimator_a_denominator = n - 1
            estimator_b_denominator = n

            # compute the values of the estimators (which estimate the variance of the sample)
            estimator_a = calculate_variance_of_sample(sample, estimator_a_denominator)
            estimator_b = calculate_variance_of_sample(sample, estimator_b_denominator)
            # print(variance_using_estimator_a)
            # print(variance_using_estimator_b)

            estimator_a_values.append(estimator_a)
            estimator_b_values.append(estimator_b)

        print(estimator_a_values)
        print(estimator_b_values)

        average_of_values_of_estimator_a = np.mean(estimator_a_values)
        average_of_values_of_estimator_b = np.mean(estimator_b_values)

        variance_of_values_of_e_a = np.var(estimator_a_values)
        variance_of_values_of_e_b = np.var(estimator_b_values)

        closer = "estimator B = "
        if abs(average_of_values_of_estimator_a - VARIANCE) < abs(average_of_values_of_estimator_b - VARIANCE):
            closer = "estimator A = "

        less_variance = "estimator B"
        if variance_of_values_of_e_a < variance_of_values_of_e_b:
            less_variance = "estimator A"

        print("Average value of estimator A is " + str(average_of_values_of_estimator_a))
        print("Average value of estimator B is " + str(average_of_values_of_estimator_b))
        print("Variance of estimator A is " + str(variance_of_values_of_e_a))
        print("Variance of estimator B is " + str(variance_of_values_of_e_b))

        print("[1] On average, " + closer + " is closer to the real value of the variance (" + str(VARIANCE) + ")")
        print("[2] The " + less_variance + " has less variance.")

        estimator_a_values = []
        estimator_b_values = []
