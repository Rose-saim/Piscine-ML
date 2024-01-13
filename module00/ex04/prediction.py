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
    if isinstance(x,np.ndarray) and len(x) > 0 and \
        isinstance(theta,np.ndarray) and len(theta) == 2:
        ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
        X = np.reshape(x, (x.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        return np.matmul(X , theta)
    return None


    x = np.arange(1, 6)
    # x = [1, 2, 3, 4, 5]

    theta1 = np.array([5, 0])
    y_hat = predict_(x, theta1)
    print(y_hat)
    # array([5., 5., 5., 5., 5.])

    theta2 = np.array([0, 1])
    y_hat = predict_(x, theta2)
    print(y_hat)
    # array([1., 2., 3., 4., 5.])

    theta3 = np.array([5, 3])
    y_hat = predict_(x, theta3)
    print(y_hat)
    # array([ 8., 11., 14., 17., 20.])

    theta4 = np.array([-3, 1])
    y_hat = predict_(x, theta4)
    print(y_hat)
    # array([-2., -1.,  0.,  1.,  2.])