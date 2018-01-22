import numpy as np
from sigmoid import sigmoid


def cost_function(theta, X, y):
    m = len(y)
    grad = np.zeros(theta.shape)

    g = sigmoid(np.dot(X, theta))
    j0 = np.dot(np.transpose(y), np.log(g))
    j1 = np.dot(np.transpose(1 - y), np.log(1 - g))
    j = (-j0 - j1) / m

    grad = np.dot(np.transpose(X), (g - y)) / m

    return j, grad


# def cost(theta, X, y):
#     m = len(y)
#     g = sigmoid(np.dot(X, theta))
#     j0 = np.dot(np.transpose(y), np.log(g))
#     j1 = np.dot(np.transpose(1 - y), np.log(1 - g))
#     j = (-j0 - j1) / m
#
#     return j


def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# def gradient(theta, X, y):
#     m = len(y)
#     g = sigmoid(np.dot(X, theta))
#     grad = np.dot(np.transpose(X), (g - y)) / m
#
#     return grad.flatten()


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad
