import numpy as np
import scipy.optimize as opt
from cost_function import *


def main():
    data = np.loadtxt("ex2data1.txt", dtype='f', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2:3]
    m, n = X.shape

    X = np.insert(X, 0, np.ones(m), axis=1)
    initial_theta = np.zeros((n + 1, 1))

    cost, grad = cost_function(initial_theta, X, y)
    print('Cost at initial theta (zeros): ', cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros):\n', grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    test_theta = np.array(([-24], [0.2], [0.2]))
    cost, grad = cost_function(test_theta, X, y)
    print('Cost at test theta : ', cost)
    print('Expected cost (approx): 0.218')
    print('Gradient at test theta :\n', grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

    opt_theta = opt.fmin_bfgs(f=cost, x0=initial_theta.flatten(), fprime=gradient, args=(X, y))
    opt.fmin_bfgs(cost_function, initial_theta.flatten(), fprime=gradient, args=(X, y))

    cost, grad = cost_function(opt_theta[0], X, y)
    print('Cost at optimal theta : ', cost)
    print('Gradient at optimal theta :\n', grad)



if __name__ == '__main__':
    main()
