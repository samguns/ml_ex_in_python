import numpy as np
from plotdata import *


def computeCost(X, y, theta):
    m = y.shape[0]
    hypothesis = np.dot(X, theta)
    error = hypothesis - y
    J = np.sum(np.power(error, 2)) / 2 / m
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))

    for it in range(num_iters):
        hypothesis = np.dot(X, theta)
        error = hypothesis - y
        p_derivative = np.sum((error * X), axis=0) / m * alpha
        p_derivative = np.reshape(p_derivative, (theta.shape[0], theta.shape[1]))
        theta = theta - p_derivative
        J_history[it] = computeCost(X, y, theta)

    return theta


def main():
    data = np.loadtxt("ex1data1.txt", dtype='f', delimiter=',')

    X = np.reshape(data[:, 0], (data.shape[0], 1))
    X = np.insert(X, 0, np.ones(data.shape[0]), axis=1)
    y = np.reshape(data[:, 1], (data.shape[0], 1))
    theta = np.zeros((2, 1))

    plotdata(data[:, 0], data[:, 1])

    print('\nTesting the cost function ...\n')
    J = computeCost(X, y, theta)
    print('Cost computed = ', J)
    print('Expected cost value (approx) 32.07')

    J = computeCost(X, y, [[-1], [2]])
    print('With theta = [-1 ; 2] \nCost computed = ', J)
    print('Expected cost value (approx) 54.24\n')

    alpha = 0.01
    iterations = 1500
    theta = gradientDescent(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent:\n')
    print(theta[0], '\n', theta[1], '\n')
    print('Expected theta values (approx)\n')
    print('[-3.6303]\n [1.1664]\n')


if __name__ == '__main__':
    main()
