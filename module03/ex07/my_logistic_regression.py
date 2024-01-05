import shutil
import time
import numpy as np
import sklearn.metrics as skm
import pandas as pd

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

    def split_data_frames(self, features, target, ratio, zipcode):
        complete_df = pd.concat([features, target],
                                            axis=1, sort=False)
        # We shuffle the dataframe
        complete_df = complete_df.sample(frac=1).reset_index(drop=True)

        # Update the Origin column to be 1 if the planet is from the
        # specified zipcode, 0 otherwise
        complete_df["Origin"] = \
            complete_df["Origin"].apply(lambda x: 1 if x == zipcode else 0)

        # We split the dataframe into two dataframes
        # according to the ratio
        split_index = int(complete_df.shape[0] * ratio)
        features_train = complete_df.iloc[:split_index, :-1].to_numpy()
        features_test = complete_df.iloc[split_index:, :-1].to_numpy()
        target_train = complete_df.iloc[:split_index, -1:].to_numpy()
        target_test = complete_df.iloc[split_index:, -1:].to_numpy()

        return features_train, features_test, target_train, target_test

    def normalize_train(self, train: np) \
            -> np.ndarray:
            min_values = []
            max_values = []
            normalized = np.empty(train.shape)
            for i in range(train.shape[1]):
                min_values.append(np.min(train[:, i]))
                max_values.append(np.max(train[:, i]))
                normalized[:, i] = (train[:, i] - min_values[i]) / (max_values[i] - min_values[i])
            return normalized, min_values, max_values

    def normalize_test(self, test: np.ndarray, min: list, max: list) \
            -> np.ndarray:
            normalized = np.empty(test.shape)
            for i in range(test.shape[1]):
                normalized[:, i] = (test[:, i] - min[i]) / (max[i] - min[i])
            return normalized
    
    def accuracy_score_(self, y, y_hat):
        if not isinstance(y, np.ndarray) \
                    or not isinstance(y_hat, np.ndarray):
                return None

        if y.shape != y_hat.shape:
            return None

        if y.size == 0:
            return None
        true = np.where(y == y_hat)[0].shape[0]
        return true / y.size