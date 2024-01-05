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
    for degree in range(1, 5):
            for degree2 in range(1, 5):
                for degree3 in range(1, 5):
                    model = {}
                    model["name"] = f"W{degree}D{degree2}T{degree3}"
                    model["degree"] = (degree, degree2, degree3)
                    # ##################################### #
                    # Train the model with the training set #
                    # ##################################### #
                    mdl_x = get_model_x(X_train_poly, model['degree'])

                    tet = []
                    lr.learning_rate = 10e-2
                    lr.n_cycle = 10_000
                    theta = np.zeros((mdl_x.shape[1], 1))
                    lr.theta = lr.fit_(mdl_x, Y_train_norm)
                    for theta_i in theta:
                        tet.append(float(theta_i[0]))
                    model["theta"] = tet

                    t_mdlx = get_model_x(X_test_poly, model['degree'])
                    y_hat = lr.predict_(t_mdlx)

                    cost = lr.loss_(Y_test_norm, y_hat)
                    model["cost"] = cost
                    print(f"Done !\nCost = {cost}")
                    print(f"-> cost: {model['cost']}\n")

                    models.append(model)

    costs = [model["cost"] for model in models]
    names = [model["name"] for model in models]
    plt.bar(names, costs)
    plt.xticks(rotation=90)
    plt.ylabel("Cost")
    plt.xlabel("Model name")
    plt.title("Comparaison of the models based on"
              + " their cost (lower is better)")
    plt.show()

     # Sort the models by cost
    models = sorted(models, key=lambda k: k['cost'])

    # Print the best models
    print("The 5 best models:")
    for model in models[:5]:
        print(f"- {model['name']} : {model['cost']}")

    # Save the models in the file "models.yml"
    with open("models.yml", "w") as file:
        yaml.dump(models, file)

    print("Models saved in the file \"models.yml\"")

    print("Done")



if __name__ == "__main__":
    main()