from prediction import predict_

import numpy as np
# from prediction import predict_
def simple_gradient(x, y, theta):
    """
    Computes the gradient of the mean squared error loss function with respect to the parameters theta.

    Args:
    - x: Feature matrix (numpy array), shape (m, n+1) where m is the number of examples and n is the number of features.
    - y: Target values (numpy array), shape (m, 1).
    - theta: Model parameters (numpy array), shape (n+1, 1).

    Returns:
    - gradient: Vector of partial derivatives (numpy array), shape (n+1, 1).
    """
    m = len(y)
    X_prime = np.c_[np.ones(m), x]
    return (X_prime.T @ (X_prime @ theta - y)) / m