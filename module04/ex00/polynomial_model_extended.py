import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    polynomial_matrix = x
    for i in range(2, power + 1):
        new_column = x ** i
        polynomial_matrix = np.c_[polynomial_matrix, new_column]
    return polynomial_matrix


if __name__ == "__main__":

    x = np.arange(1, 11).reshape(5, 2)
    print("x :")
    print(x)

    polynomial_matrix = add_polynomial_features(x, 3)
    print("\npolynomial matrix of x by degree 3 :")
    print(polynomial_matrix)
    # Output:
    # array([[   1,    2,    1,    4,    1,    8],
    #        [   3,    4,    9,   16,   27,   64],
    #        [   5,    6,   25,   36,  125,  216],
    #        [   7,    8,   49,   64,  343,  512],
    #        [   9,   10,   81,  100,  729, 1000]])

    polynomial_matrix = add_polynomial_features(x, 4)
    print("\npolynomial matrix of x by degree 4 :")
    print(polynomial_matrix)
    # Output:
    # array([[    1,     2,     1,     4,     1,     8,     1,    16],
    #        [    3,     4,     9,    16,    27,    64,    81,   256],
    #        [    5,     6,    25,    36,   125,   216,   625,  1296],
    #        [    7,     8,    49,    64,   343,   512,  2401,  4096],
    #        [    9,    10,    81,   100,   729,  1000,  6561, 10000]])