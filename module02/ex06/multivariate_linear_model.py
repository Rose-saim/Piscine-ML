import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error


def main():
    # Dataset
    lr=MyLR([[[0], [0]]])
    data = pd.read_csv('../spacecraft.csv')
    features = ["Age", "Thrust_power", "Terameters"]
    # for feature, c in zip(features, colors):
    #     lr.theta = np.array([[500], [1]])
    #     X = np.array(data[feature]).reshape(-1, 1)
    #     Y = np.array(data["Sell_price"]).reshape(-1, 1)
    #     # 1
    #     lr.fit_(X, Y)
    #     y_hat = lr.predict_(X)
    #     # Fonction mse
    #     print("\nMSE for feature {}: {}\n".format(
    #         feature, lr.mse_(Y, y_hat)))
    #     # Minimisation algo
    #     plt.scatter(X, Y, colors=c)
    #     plt.scatter(X, y_hat, color=c, marker=".")
    #     plt.title(
    #         "Sell price of a spacecraft depending on its {}".format(feature))
    #     plt.xlabel(feature)
    #     plt.ylabel("y: sell price (in keuros)")
    #     plt.grid()
    #     plt.legend(["Data", "Prediction"])
    #     plt.show()
    X = np.array(data[features])
    Y = np.array(data["Sell_price"]).reshape(-1, 1)
    print(X.shape, Y.shape)

    lr.theta = np.array([[1.0], [1.0], [1.0], [1.0]])
    # y_hat = lr.predict_(X)
    # print("initial MSE : {}".format(lr.mse_(Y, y_hat)))

    # Update thetas with appropriate values
    # lr.theta = np.array([[385.21139513],
    #                         [-24.33149116],
    #                         [5.67045772],
    #                         [-2.66684314]])
    lr.alpha = 1e-5
    lr.max_iter = 400000
    lr.fit_(X, Y)
    y_hat = lr.predict_(X)
    print("thetas : {}".format(lr.theta))

    for feature in features:
        plt.scatter(X[:, features.index(feature)], Y)
        plt.scatter(X[:, features.index(feature)], y_hat,
                    color="orange",
                    marker=".")
        plt.title(
            "Sell price of a spacecraft depending on its {}".format(feature))
        plt.xlabel(feature)
        plt.ylabel("Sell price")
        plt.grid()
        plt.legend(["Data", "Prediction"])
        plt.show()

if __name__ == "__main__":
    main()