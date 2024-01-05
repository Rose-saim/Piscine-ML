import numpy as np
from prediction import predict_
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
    ele = predict_(x, theta) - y
    g0 = np.sum(ele) / m
    g1 = ((np.sum(ele * x)) / m)
    print(np.array([g0, g1]))     

# Example usage:
# x and y are your feature matrix and target values
# theta is your parameter vector
# gradient = compute_gradient(x, y, theta)


# Example usage:
# x and y are your feature matrix and target values
# theta is your parameter vector
# gradient = compute_gradient(x, y, theta)
import numpy as np
x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
# Example 0:
theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(theta1)
print(np.shape(theta1.reshape(-1, 1)))
simple_gradient(x, y, theta1)
# # Output:
# array([[-19.0342574], [-586.66875564]])
# Example 1:
theta2 = np.array([1, -0.4]).reshape((-1, 1))
simple_gradient(x, y, theta2)
# # Output:
# array([[-57.86823748], [-2230.12297889]])