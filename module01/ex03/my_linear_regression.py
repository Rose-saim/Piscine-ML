import numpy as np

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if theta is not None and alpha is not None\
            and isinstance(max_iter, int) and max_iter > 0:
            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
        else:
            print("Error inputs")
            quit()
    
    def predict_(self, x):
        if isinstance(x,np.ndarray) and len(x) > 0 and \
            isinstance(self.theta,np.ndarray) and len(self.theta) == 2:
            ones = np.ones((x.shape[0], 1)) #matrice remplie de 1, 1 seule colonne
            X = np.reshape(x, (x.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)
            return np.matmul(X , self.theta)
        return None
    
    def gradient(self, x, y):
        m = len(y)
        ones_column = np.ones((m, 1))
        X = np.hstack((ones_column, x))
        X_th_y = self.predict_(x, self.theta) - y
        gradient = (1 / m) * np.matmul(X.T, X_th_y)
        return gradient

    def fit_(self, x, y):
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient(x, y, self.theta)
        return self.theta

    
    def loss_elem_(self, y, y_hat):
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            return (y_hat - y) **2
        return None

    def loss_(self, y, y_hat):
        if y is not None and y_hat is not None and y.shape == y_hat.shape:
            tmp = self.loss_elem_(y, y_hat) * (1 / (2 * y.shape[0]))
            return np.sum(tmp)
        return None





    