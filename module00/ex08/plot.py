import matplotlib.pyplot as plt
from prediction import predict_
import numpy as np
from loss import loss_
def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    if isinstance(x,np.ndarray) and len(x) > 0 and \
        isinstance(y,np.ndarray) and len(x) > 0 and \
        isinstance(theta,np.ndarray) and len(theta) == 2:
        plt.plot(x, y, 'o')
        X = predict_(x, theta)
        plt.plot(x, X)
        res = loss_(y, X)
        for i in range(len(x)):
            plt.plot([x[i], x[i]], [y[i], X[i]], linestyle='dashed', color='red')
    plt.title(f'Prediction and Loss (J = {res*2:.6f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


import numpy as np
x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1= np.array([18,-1])
plot_with_loss(x, y, theta1)
# Output:
# Example 2:
theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)
# Output:
# Example 3:
theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)
# Output: