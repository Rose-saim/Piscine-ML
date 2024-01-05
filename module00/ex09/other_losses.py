import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mse_(y, y_hat):
    """
    Computes the Mean Squared Error (MSE) between the true values (y) and predicted values (y_hat).

    Args:
    - y: Actual output values (numpy array).
    - y_hat: Predicted output values (numpy array).

    Returns:
    - mse: Mean Squared Error.
    """
    m = len(y)
    mse = np.sum((y_hat - y) ** 2) / m
    return mse

def rmse_(y, y_hat):
    """
    Computes the Root Mean Squared Error (RMSE) between the true values (y) and predicted values (y_hat).

    Args:
    - y: Actual output values (numpy array).
    - y_hat: Predicted output values (numpy array).

    Returns:
    - rmse: Root Mean Squared Error.
    """
    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    return rmse

def mae_(y, y_hat):
    """
    Computes the Mean Absolute Error (MAE) between the true values (y) and predicted values (y_hat).

    Args:
    - y: Actual output values (numpy array).
    - y_hat: Predicted output values (numpy array).

    Returns:
    - mae: Mean Absolute Error.
    """
    m = len(y)
    mae = np.sum(np.abs(y_hat - y)) / m
    return mae

def r2score_(y, y_hat):
    """
    Computes the R-squared (coefficient of determination) score between the true values (y) and predicted values (y_hat).

    Args:
    - y: Actual output values (numpy array).
    - y_hat: Predicted output values (numpy array).

    Returns:
    - r2: R-squared score.
    """
    y_mean = np.mean(y)
    ss_res = np.sum((y_hat - y) ** 2)
    ss_total = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2


if __name__ == "__main__":

    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    # your implementation
    print(mse_(x, y))
    # Output: 4.285714285714286
    # sklearn implementation
    print(mean_squared_error(x, y))
    # Output: 4.285714285714286

    # Root mean squared error
    # your implementation
    print(rmse_(x, y))
    # Output: 2.0701966780270626
    # sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    # Output:
    # 2.0701966780270626

    # Mean absolute error
    # your implementation
    print(mae_(x, y))
    # Output: 1.7142857142857142
    # sklearn implementation
    print(mean_absolute_error(x, y))
    # Output:
    # 1.7142857142857142

    # R2-score
    # your implementation
    print(r2score_(x, y))
    # Output: 0.9681721733858745
    # sklearn implementation
    print(r2_score(x, y))
    # Output: 0.9681721733858745