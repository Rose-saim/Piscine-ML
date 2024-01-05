import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as ML
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as skl
from sklearn.model_selection import train_test_split
import yaml 

def get_model_x(x_train_poly, degree):
                    weight = x_train_poly[:, 0:degree[0]]
                    distance = x_train_poly[:, 4:4 + degree[1]]
                    time = x_train_poly[:, 8:8 + degree[2]]
                    model_x = \
                        np.concatenate((weight, distance, time), axis=1)
                    return model_x

def denormalize_prediction(y_hat, y_min, y_max):
    return y_hat * (y_max - y_min) + y_min
    
def normalize_train(training: np.ndarray) -> tuple:
        try:
            if not isinstance(training, np.ndarray):
                print("Error: training must be a np.ndarray")
                exit(1)
            elif training.shape[0] == 0:
                print("Error: training must not be empty")
                exit(1)
            min = []
            max = []
            normalized = np.empty(training.shape)
            for i in range(training.shape[1]):
                min.append(np.min(training[:, i]))
                max.append(np.max(training[:, i]))
                normalized[:, i] = \
                    (training[:, i] - min[i]) / (max[i] - min[i])
            return (normalized, min, max)
        except Exception:
            print("Error: Can't normalize the training dataset")
            exit(1)

def normalize_test(test: np.ndarray, min: list, max: list) \
        -> np.ndarray:
    try:
        if not isinstance(test, np.ndarray):
            print("Error: test must be a np.ndarray")
            exit(1)
        elif test.shape[0] == 0:
            print("Error: test must not be empty")
            exit(1)
        elif not isinstance(min, list) or not isinstance(max, list):
            print("Error: min and max must be lists")
            exit(1)
        elif len(min) != test.shape[1] or len(max) != test.shape[1]:
            print("Error: min and max must have the same size as test")
            exit(1)
        normalized = np.empty(test.shape)
        for i in range(test.shape[1]):
            normalized[:, i] = (test[:, i] - min[i]) / (max[i] - min[i])
        return normalized
    except Exception:
        print("Error: Can't normalize the test dataset")
        exit(1)


def main():
    data = pd.read_csv("../space_avocado.csv")
    features = ["weight", "prod_distance", "time_delivery"]

    X = np.array(data[features])
    Y = np.array(data["target"]).reshape(-1, 1)

    with open("models.yml", "r") as f:
        models = yaml.load(f, Loader=yaml.loader.UnsafeLoader)

    best_model = min(models, key=lambda x: x["cost"])

    scaler = skl()
    lr = ML([[0], [0]])

    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.2)
    lr.alpha = 1e-2
    lr.max_iter = 10000

    # Normalize the training features
    X_train_norm, x_min, x_max = normalize_train(X_train)
    # Normalize the training target
    Y_train_norm, y_min, y_max = normalize_train(Y_train)
    X_test_norm = normalize_test(X_test, x_min, x_max)
    Y_test_norm = normalize_test(Y_test, y_min, y_max)
    
    X_train_poly = add_polynomial_features(X_train_norm, 4)
    X_test_poly = add_polynomial_features(X_test_norm, 4)

    models = []
    # ##################################### #
    # Train the model with the training set #
    # ##################################### #
    mdl_x = get_model_x(X_train_poly, best_model['degree'])

    lr.learning_rate = 10e-2
    lr.n_cycle = 10_000

    lr.theta = lr.fit_(mdl_x, Y_train_norm)

    # ############################## #
    # The model with the testing set #
    # ############################## #
    t_mdlx = get_model_x(X_test_poly, best_model['degree'])
    y_hat = lr.predict_(t_mdlx)

    cost = lr.loss_(Y_test_norm, y_hat)
    best_model["cost"] = cost
    print(f"Done !\nCost = {cost}")
    print(f"-> cost: {best_model['cost']}\n")

    # ########################### #
    # Plot the train and test set #
    # ########################### #
    y_hat_denorm = denormalize_prediction(y_hat, y_min[0], y_max[0])

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):

        for j in range(3):

            if j == 0:
                ax[i, j].scatter(X_test[:, i],
                                 Y_test,
                                 color="blue", alpha=0.5, s=10)
            elif j == 1:
                ax[i, j].scatter(X_test[:, i],
                                 y_hat_denorm,
                                 color="red", alpha=0.5, s=5)
            elif j == 2:
                ax[i, j].scatter(X_test[:, i],
                                 Y_test,
                                 color="blue", alpha=0.5, s=10)
                ax[i, j].scatter(X_test[:, i],
                                 y_hat_denorm,
                                 color="red", alpha=0.5, s=5)
            ax[i, j].set_xlabel(features[i])
            ax[i, j].set_ylabel("Price")

    cols = ["Real values", "Predicted values", "Both"]

    for ax, col in zip(ax[0], cols):
        ax.set_title(col)

    plt.tight_layout()
    plt.show()

    # Two 4D plots to visualize the model prediction and the real values
    fig, ax = plt.subplots(1, 2,
                           figsize=(15, 5),
                           subplot_kw={"projection": "3d"})

    fig.colorbar(ax[0].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                               c=Y_test), ax=ax[0], label="Price")
    fig.colorbar(ax[1].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                               c=y_hat_denorm), ax=ax[1], label="Price")

    ax[0].set_title("Real")
    ax[1].set_title("Predicted")

    for i in range(2):
        ax[i].set_xlabel(features[0])
        ax[i].set_ylabel(features[1])
        ax[i].set_zlabel(features[2])

    plt.show()

    def r2score_elem(y, y_hat):
        m = y.shape[0]
        mean = y.mean()
        numerator = 0.
        denominator = 0.
        for i in range(m):
            numerator += (y_hat[i] - y[i]) ** 2
            denominator += (y[i] - mean) ** 2
        return numerator / denominator

    def r2score_(y, y_hat):
        """
            Description:
                Calculate the R2score between the predicted output and the output.
            Args:
                y: has to be a numpy.array, a vector of dimension m * 1.
                y_hat: has to be a numpy.array, a vector of dimension m * 1.
            Returns:
                r2score: has to be a float.
                None if there is a matching dimension problem.
            Raises:
                This function should not raise any Exceptions.
        """
        print(f"y: {y.shape}")
        print(f"y_hat: {y_hat.shape}")
        return 1 - r2score_elem(y, y_hat)

    print(f"R2 score: {r2score_(Y_test, y_hat_denorm)}")



if __name__ == "__main__":
    main()