import numpy as np

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    m = y.shape[0]
    try:
        # Change the values of y_hat to avoid math error
        y_hat = np.clip(y_hat, eps, 1 - eps)
        dot1 = y.T.dot(np.log(y_hat))
        dot2 = (1 - y).T.dot(np.log(1 - y_hat))
        return ((dot1 + dot2) / - m)[0][0]

    except Exception:
        return None