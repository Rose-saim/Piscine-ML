import numpy as np
from prediction import predict_

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        return vect
    return None

def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    # y = check_dim_vector(y)
    # y_hat= check_dim_vector(y_hat)
    loss_e = (y_hat - y)**2
    return loss_e

def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    # y = check_dim_vector(y)
    # y_hat= check_dim_vector(y_hat)
    m = y.shape[0]
    v_loss = loss_elem_(y, y_hat) / (2 * m)
    return np.sum(v_loss)

import numpy as np
x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
print(loss_elem_(y1, y_hat1))
# # Output:
# array([[0.], [1], [4], [9], [16]])

# # Example 2:
print(loss_(y1, y_hat1))
# # # Output:
# # 3.0

x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
y_hat2 = predict_(x2, theta2)
y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

# Example 3:
print(loss_(y2, y_hat2))
# # Output:
# 2.142857142857143

# Example 4:
print(loss_(y2, y2))
# Outp