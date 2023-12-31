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

import numpy as np
# Example 1:
x = np.arange(1,6)
print(add_intercept(x))
# Output:
# array([[1., 1.],
# [1., 2.],
# [1., 3.],
# [1., 4.],
# [1., 5.]])
# Example 2:
y = np.arange(1,10).reshape((3,3))
print(add_intercept(y))
# Output:
# array([[1., 1., 2., 3.],
# [1., 4., 5., 6.],
# [1., 7., 8., 9.]])