import numpy as np

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """

    m = (x.shape[0])
    n = (x.shape[1])
    # if isinstance(x,np.ndarray) and len(x) > 0 and isinstance(theta,np.ndarray) and len(theta) > 1:
    #     print(x.shape[1])
    y_hat = np.zeros((m, 1))
    for j in range(m):
        y_hat[j] = theta[0]
        for i in range(n):
            y_hat[j] += theta[i+1] * x[j][i]
    return y_hat

x = np.arange(1,13).reshape((4,-1))
# Example 1:
theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print(predict_(x, theta1))
# # Ouput:
# array([[5.], [5.], [5.], [5.]])
# Do you understand why y_hat contains only 5’s here?
# Example 2:
theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(predict_(x, theta2))
# # Output:
# array([[ 1.], [ 4.], [ 7.], [10.]])
# Do you understand why y_hat == x[:,0] here?
# Example 3:
theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(predict_(x, theta3))
# # Output:
# array([[ 9.64], [24.28], [38.92], [53.56]])
# Example 4:
theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(predict_(x, theta4))
# # Output:
# array([[12.5], [32. ], [51.5], [71. ]])