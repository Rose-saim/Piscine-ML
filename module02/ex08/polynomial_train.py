from polynomial_model import add_polynomial_features
from my_linear_regression import MyLinearRegression as MyLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    linear_regression = MyLR([80, -10], 10e-10, 100_000)

    # Get the dataset in the file are_blue_pills_magics.csv
    try:
        dataset = pd.read_csv("../are_blue_pills_magics.csv")

    except Exception:
        print("Couldn't find the dataset file")
        exit(1)

    print(dataset, "\n")

    x = dataset['Micrograms'].values
    y = dataset['Score'].values.reshape(-1, 1)

    polynomial_x = linear_regression.add_polynomial_features(x, 6)

    # Loss = 18.13
    hypothesis_theta1 = np.array([[89.04720427],
                                  [-8.99425854]]
                                 ).reshape(-1, 1)

    # Loss = 26.988042374726316
    hypothesis_theta2 = np.array([[69.77316037],
                                  [1.49660362],
                                  [-1.21861482]]).reshape(-1, 1)

    # Loss = 27.87
    hypothesis_theta3 = np.array([[89.0],
                                  [-8.4],
                                  [0.8],
                                  [-0.1]]).reshape(-1, 1)

    # Loss = 45
    hypothesis_theta4 = np.array([[-19.9],
                                  [160.4],
                                  [-78.6],
                                  [13.6],
                                  [-0.8]]
                                 ).reshape(-1, 1)

    # Loss = 12.83
    hypothesis_theta5 = np.array([[1140],
                                  [-1850],
                                  [1110],
                                  [-305.2],
                                  [39.3],
                                  [-1.9]]
                                 ).reshape(-1, 1)

    hypothesis_theta6 = np.array([[9110],
                                  [-18015],
                                  [13400],
                                  [-4935],
                                  [966],
                                  [-96.4],
                                  [3.86]]
                                 ).reshape(-1, 1)

    hypothesis_thetas = [hypothesis_theta1, hypothesis_theta2,
                         hypothesis_theta3, hypothesis_theta4,
                         hypothesis_theta5, hypothesis_theta6]

    thetas = []
    mse_scores = []

    # Trains six separate Linear Regression models with polynomial
    # hypothesis with degrees ranging from 1 to 6
    # Plots the 6 models and the data points on the same figure.
    # Use lineplot style for the models and scaterplot for the data points.
    # Add more prediction points to have smooth curves for the models.
    fig, ax = plt.subplots(2, 3)

    for i in range(1, 7):
        print("Training model {} / 6\n".format(i))

        linear_regression.theta = hypothesis_thetas[i - 1]
        current_x = polynomial_x[:, :i]
        linear_regression.fit_(current_x, y)
        y_hat = linear_regression.predict_(current_x)

        thetas.append(linear_regression.theta)
        mse_scores.append(linear_regression.mse_(y, y_hat))

        # Plots the data points
        ax[(i - 1) // 3][(i - 1) % 3].scatter(x, y, color='blue')

        # Plots the model curve
        min_x = np.min(x)
        max_x = np.max(x)
        continuous_x = np.linspace(min_x, max_x, 100)
        predicted_x = linear_regression.add_polynomial_features(continuous_x,
                                                                i)
        predicted_y = linear_regression.predict_(predicted_x)
        ax[(i - 1) // 3][(i - 1) % 3].plot(continuous_x, predicted_y,
                                           color='orange')
        # Add title and axis names
        ax[(i - 1) // 3][(i - 1) % 3].set_title(
            "Degree {}, score : {}".format(i, mse_scores[i - 1]))
        ax[(i - 1) // 3][(i - 1) % 3].set_xlabel("Micrograms")
        ax[(i - 1) // 3][(i - 1) % 3].set_ylabel("Score")

        # Compute Loss
        loss = linear_regression.loss_(y, y_hat)
        print()
        print("Loss {} : {}".format(i, loss))
        print("Thetas : {}".format(linear_regression.theta))
        print()

    plt.show()

    for i in range(6):
        print("Model {} :".format(i + 1))
        print("Thetas : {}".format(thetas[i]))
        print("MSE : {}\n".format(mse_scores[i]))

    # Plots a bar plot showing the MSE score of the models in function of
    # the polynomial degree of the hypothesis,
    plt.bar([1, 2, 3, 4, 5, 6], mse_scores)
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.show()