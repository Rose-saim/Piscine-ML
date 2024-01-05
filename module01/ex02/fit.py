from vec_gradiant import gradient
import numpy as np
from prediction import predict_
import numpy as np

def check_dim_vector(vect):
    if isinstance(vect, np.ndarray):
        if np.ndim(vect) == 1:
            vect = np.reshape(vect, (vect.shape[0], 1))       
        return vect
    return None

def check_theta(theta):
    if isinstance(theta,np.ndarray) and theta.shape[0] > 0:
        if np.ndim(theta) == 1:
            theta = np.reshape(theta, (theta.shape[0], 1))  
        if theta.shape[0] == 2 and theta.shape[1] == 1:
            return theta 
    return None

def check_alpha(alpha):
    if isinstance(alpha, float) and alpha >= 0 and alpha <= 1:
        return alpha
    return None  


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    for _ in range(max_iter):
        theta = theta - alpha * gradient(x, y, theta)
    return theta

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = (fit_(x, y, theta, alpha=5e-8, max_iter=1500000))
print(theta1)
# Output:
# array([[1.40709365],
# [1.1150909 ]])
# Example 1:
print(predict_(x, theta1))
# Output:
# array([[15.3408728 ],
# [25.38243697],
# [36.59126492],
# [55.95130097],
# [65.53471499]])