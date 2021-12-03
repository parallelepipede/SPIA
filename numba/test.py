import numpy as np
from numba import jit, prange, njit

import time

# Descent regression for equation 3 * X  + 20 + Gaussian noise
#@jit(nopython=True, parallel=True)
@njit(parallel=True)
def gradient_regression():
    """
    Optimisation of gradient descent with Numba library
    Return Residual Sum of squares and coefficients for gradient_regression
    """
    # Gradient descent parameters
    alpha = 0.01
    iterations = 1000

    # Create data
    n_obs = 2_000_000
    np.random.seed(0)

    X  = np.full((n_obs,2), 1.0)
    X[:,0] = np.random.randint(n_obs)

    y = 2 * X[:,0] + 5 + np.random.normal(loc=0,scale = .5, size = n_obs)
    coef = np.array([.5, .5])

    #Residual Sum of Squares (the amount of variance in the dataset not explained by the regression)
    rss = np.zeros(iterations)

    for i in prange(iterations):

        y_pred = np.dot(X,coef)
        err = y - y_pred
        rss[i]  = np.sum(np.square(err))
        gradient = 2.0/n_obs * np.dot(np.transpose(X),err)
        coef = coef + alpha * np.transpose(gradient)
    #print(rss, coef)
    return rss, coef

tim = time.perf_counter()
gradient_regression()
print(time.perf_counter()  - tim)