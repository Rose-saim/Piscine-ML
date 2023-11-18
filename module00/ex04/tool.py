import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) and len(x):
        tmp = [[1] for _ in range(len(x))]
        return np.append(tmp, x.reshape((len(x), 1)) if isinstance(x[0], np.integer) else x, axis=1)
    return None