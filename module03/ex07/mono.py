import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as ML
from matplotlib import patches


def zip_code():
    parser = argparse.ArgumentParser(description="Script for logisctic regression with zipcode.")
    parser.add_argument("-zipcode", type=int, help="zipcode", choices=(0, 1, 2, 3), required=True)
    args = parser.parse_args()
    return args.zipcode

def read_csv(path: str):
    try:
        if path is not None:
            data = pd.read_csv(path)
            if data.empty:
                raise ValueError("Error: Empty file.")
            return data
        else:
            raise ValueError("Error: 'path' cannot be None.")
    except pd.errors.EmptyDataError:
        raise ValueError("Error: Empty file.")
    except Exception as e:
        raise ValueError(f"Error: An unexpected error occured - {str(e)}")
    
def main():
    feature = ["weight", "height", "bone_density"]
    features = read_csv("../solar_system_census.csv")[feature]
    target = ["Origin"]
    targets = read_csv("../solar_system_census_planets.csv")[target]

    mylr = ML([0], max_iter=250000, alpha=0.01)
    code = zip_code()


    features_train, features_test, target_train, target_test = mylr.split_data_frames(features, targets, ratio=0.2, zipcode=code)    
    norm_train, min_v, max_v = mylr.normalize_train(features_train)
    norm_test = mylr.normalize_test(features_test, min_v, max_v)

    mylr.theta = np.zeros((norm_train.shape[1] + 1, 1))
    mylr.fit_(norm_train, target_train)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    feat = np.vstack((norm_train, norm_test))
    feat_denorm = np.vstack((features_train, features_test))
    target = np.vstack((target_train, target_test))

    y_hat = mylr.predict_(feat)
    y_hat = np.where(y_hat >= 0.5, 1, 0)

    accuracy_score = mylr.accuracy_score_(target, y_hat)
    print("Accuracy score: {} %".format(accuracy_score * 100))

    colors = np.select(
            [
                (y_hat == 1) & (target == 1),  # True positive
                (y_hat == 0) & (target == 0),  # True negative
                (y_hat == 1) & (target == 0),  # False positive
                (y_hat == 0) & (target == 1)   # False negative
            ],
            [
                'green',
                'blue',
                'orange',
                'red'
            ],
            default='black'  # Choose a default color if none of the conditions are satisfied
        )

    for i in range(3):
        index = i if i < 2 else -1

        ax[i].scatter(
            feat_denorm[:, index],
            feat_denorm[:, index+1],
            c=colors.flatten(),
            marker='o',
            alpha=0.5,
            edgecolors='none'
        )

    fig.legend(
        handles=[
            patches.Patch(color='green',
                          label='True positive'),
            patches.Patch(color='blue',
                          label='True negative'),
            patches.Patch(color='red',
                          label='False positive'),
            patches.Patch(color='yellow',
                          label='False negative'),
        ],
        loc='lower center', ncol=4, fontsize='small',
    )
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        feat_denorm[:, 0],
        feat_denorm[:, 1],
        feat_denorm[:, 2],
        c=colors.flatten(),
        marker='o',
        alpha=0.5,
        edgecolors='none'
    )

    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title('bone_density vs height vs weight')

    fig.legend(
        handles=[
            patches.Patch(color='green',
                          label='True positive'),
            patches.Patch(color='blue',
                          label='True negative'),
            patches.Patch(color='orange',
                          label='False positive'),
            patches.Patch(color='red',
                          label='False negative'),
        ],
        loc='lower center', ncol=4, fontsize='small',
    )

    plt.show()

if __name__ == "__main__":
    main()
    