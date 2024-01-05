import numpy as np

class MyLinearRegression():
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if theta is not None and alpha is not None and isinstance(max_iter, int) and max_iter > 0:
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
        else:
            raise ValueError("Invalid inputs")

    def predict_(self, x):
        if isinstance(x, np.ndarray) and len(x) > 0 and isinstance(self.theta, np.ndarray) and len(self.theta) == 2:
            ones = np.ones((x.shape[0], 1))
            X = np.concatenate((ones, x), axis=1)
            return np.dot(X, self.theta)
        else:
            return np.zeros(x.shape[0])  # ou toute autre valeur par défaut appropriée


    def gradient(self, x, y):
        m = len(y)
        ones_column = np.ones((m, 1))
        x = x.reshape(-1, 1)
        X = np.hstack((ones_column, x))
        X_th_y = self.predict_(x) - y
        gradient = (1 / m) * np.dot(X.T, X_th_y)
        return gradient

    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient(x, y)
        return self.theta

    def loss_elem_(self, y, y_hat):
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            return np.power(y_hat - y, 2)
        return None

    def loss_(self, y, y_hat):
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            tmp = self.loss_elem_(y, y_hat) * (1 / (2 * y.shape[0]))
            return np.sum(tmp)
        return None

    def gradient_descent(self, X, y, theta,):
        m = len(y)
        cost = np.zeros(self.max_iter)
        theta_h = np.zeros((self.max_iter, 2))
        for it in range(self.max_iter):
            pred = np.dot(X, theta)
            theta = theta - 1/m * self.alpha * (X.T.dot((pred - y)))
            theta_h[it, :] = theta.T
            cost[it] = self.loss_(theta, X, y)
        return theta, cost, theta_h