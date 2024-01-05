import shutil
import time
import numpy as np
import sklearn.metrics as skm

class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, max_iter=1000, eps=1e-15):
        if theta is not None and alpha is not None and isinstance(max_iter, int) and max_iter > 0:
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
            self.eps = eps
        else:
            raise ValueError("Invalid inputs")

    @staticmethod
    def sigmoid_(x):
        return 1.0 / (1.0 + np.exp(-x))
    

    def predict_(self, x):
        m = x.shape[0]
        X_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = self.sigmoid_(X_prime.dot(self.theta))
        return y_hat

    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient_(x, y)
        return self.theta

    def loss_elem_(self, y, y_hat):
        m = y.shape[0]
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        loss_elem = []
        for y_value, y_hat_value in zip(y, y_hat):
            dot1 = y_value * np.log(y_hat_value)
            dot2 = (1 - y_value) * (np.log(1 - y_hat_value))
            loss = -(dot1 + dot2)
            loss_elem.append(loss)
        return loss_elem

    def loss_(self, y, y_hat):
        tmp = self.loss_elem_(y, y_hat)
        return np.mean(tmp)

    def gradient_(self, x, y):
        m, n = x.shape

        if m == 0 or n == 0:
            return None
        elif y.shape != (m, 1) or self.theta.shape != ((n + 1), 1):
            return None
        X_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = self.predict_(x)
        if y_hat is None:
            return None
        return (X_prime.T.dot(y_hat - y)) / m

    
if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.],
                  [5., 8., 13., 21.],
                  [3., 5., 9., 14.]])
    Y = np.array([[1.], [0.], [1.]])
    theta = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])

    mylr = MyLogisticRegression(theta)

    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[0.99930437],
    #        [1.        ],
    #        [1. ]])

    # Example 1:
    loss = mylr.loss_(Y, mylr.predict_(X))

    sklearn_loss = skm.log_loss(Y, mylr.predict_(X))
    print(loss, "vs", sklearn_loss)
    # Output:
    # 11.513157421577004

    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output:
    # array([[ 2.11826435]
    #        [ 0.10154334]
    #        [ 6.43942899]
    #        [-5.10817488]
    #        [ 0.6212541 ]])

    # Example 3:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[0.57606717]
    #       [0.68599807]
    #       [0.06562156]])

    # Example 4:
    loss = mylr.loss_(Y, y_hat)
    print(loss, "vs", skm.log_loss(Y, y_hat))
    # Output:
    # 1.4779126923052268