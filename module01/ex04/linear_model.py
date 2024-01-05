import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error


def main():
    # Dataset
    data = pd.read_csv('../data.csv')
    X = np.array(data['Micrograms']).reshape(-1, 1)
    Y = np.array(data['Score']).reshape(-1, 1)
    # Modele
    # 1
    lr=MyLR(np.array([[0], [0]]),max_iter=50000)
    lr.fit_(X, Y)
    y_hat = lr.predict_(X)
    # Fonction loss
    cost = lr.loss_(Y, y_hat)
    # Minimisation algo
    plt.scatter(X, Y)
    plt.plot(X, y_hat)
    plt.show()



if __name__ == '__main__':
    main()