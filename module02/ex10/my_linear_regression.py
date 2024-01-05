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
        X_prime = np.c_[np.ones(x.shape[0]), x]
        return np.dot(X_prime, self.theta)


    def gradient(self, x, y):
        m = len(x)
        X_prime = np.c_[np.ones(m), x]
        return (X_prime.T @ (X_prime @ self.theta - y)) / m
    
    def fit_(self, x, y):
        self.theta = np.zeros((x.shape[1]+1, 1))
        print(self.theta.shape, x.shape)
        for _ in range(self.max_iter):
            gradient = self.gradient(x, y)
            self.theta = self.theta - self.alpha * gradient
        return self.theta

    def loss_elem_(self, y, y_hat):
        return np.power(y_hat - y, 2)

    def loss_(self, y, y_hat):
        tmp = self.loss_elem_(y, y_hat) * (1 / (2 * y.shape[0]))
        return np.sum(tmp)
    
    def mse_(self, y, y_hat):
        mse = np.sum((y_hat - y) ** 2)
        return mse.mean()
    
    def gradient_descent(self, X, y):
        m = len(y)
        cost = np.zeros(self.max_iter)
        theta_h = np.zeros((self.max_iter, 2))
        for it in range(self.max_iter):
            pred = np.dot(X, self.theta)
            self.theta = self.theta - 1/m * self.alpha * (X.T.dot((pred - y)))
            theta_h[it, :] = self.theta.T
            cost[it] = self.loss_(self.theta, X, y)
        return self.theta, cost, theta_h

    def add_polynomial_features(self, x, power):
        m = x.shape[0]
        arr = np.zeros((m, power))
        # Remplissez la matrice avec les caractéristiques polynomiales
        for i in range(power):
            arr[:, i] = x.ravel() ** (i + 1)
        return arr
