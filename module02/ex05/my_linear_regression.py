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
        m = len(y)
        X_prime = np.c_[np.ones(m), x]
        return (X_prime.T @ (X_prime @ self.theta - y)) / m
    
    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient(x, y)
        return self.theta

    def loss_elem_(self, y, y_hat):
        return np.power(y_hat - y, 2)

    def loss_(self, y, y_hat):
        tmp = self.loss_elem_(y, y_hat) * (1 / (2 * y.shape[0]))
        return np.sum(tmp)

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
    
import numpy as np
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
# Example 0:
y_hat = mylr.predict_(X)
print(mylr.predict_(X))
# Output:
# array([[8.], [48.], [323.]])
# Example 1:
print(mylr.loss_elem_(Y, y_hat))
# Output:)
# array([[225.], [0.], [11025.]])
# Example 2:
print(mylr.loss_(Y, y_hat))
# Output:
# 1875.0
# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.theta)
# Output:
# array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
# Example 4:
y_hat = mylr.predict_(X)
print(mylr.predict_(X))
# Output:
# array([[23.417..], [47.489..], [218.065...]])
# Example 5:
print(mylr.loss_elem_(Y, y_hat))
# Output:
# array([[0.174..], [0.260..], [0.004..]])
# Example 6:
print(mylr.loss_(Y, y_hat))
# Output:
# 0.0732..