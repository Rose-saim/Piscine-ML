import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as ML
from matplotlib import patches

def parse_zipcode():
    parser = argparse.ArgumentParser()
    parser.add_argument("-zipcode", type=int, help="zipcode",
                        choices=(0, 1, 2, 3), required=True)
    args = parser.parse_args()
    return args.zipcode

def normalize_train(train: np) \
        -> np.ndarray:
        min_values = []
        max_values = []
        normalized = np.empty(train.shape)
        for i in range(train.shape[1]):
            min_values.append(np.min(train[:, i]))
            max_values.append(np.max(train[:, i]))
            normalized[:, i] = (train[:, i] - min_values[i]) / (max_values[i] - min_values[i])
        return normalized, min_values, max_values

def normalize_test(test: np.ndarray, min: list, max: list) \
        -> np.ndarray:
        normalized = np.empty(test.shape)
        for i in range(test.shape[1]):
            normalized[:, i] = (test[:, i] - min[i]) / (max[i] - min[i])
        return normalized

def split_data_frames(features, target, ratio, zipcode):
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

def accuracy_score_(y, y_hat):
    true = np.where(y == y_hat)[0].shape[0]
    return true / y.size

if __name__ == "__main__":
    feature = ["weight", "height", "bone_density"]
    features = pd.read_csv("../solar_system_census.csv")[feature]
    target = ["Origin"]
    targets = pd.read_csv("../solar_system_census_planets.csv")[target]
  
    mylr = ML([0], max_iter=150000, alpha=1)
    zipcode = parse_zipcode()
   
    feat_train, feat_test, target_train, target_test = split_data_frames(features, targets, 0.8, zipcode)
    x_train, x_min, x_max = normalize_train(feat_train)
    x_test = normalize_test(feat_test, x_min, x_max)

    mylr.theta = np.zeros((x_train.shape[1] + 1, 1))
    mylr.fit_(x_train, target_train)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    features_pairs = [
        ('weight', 'height'),
        ('weight', 'bone_density'),
        ('height', 'bone_density')
    ]

    features = np.vstack((x_train, x_test))
    feature_denorm = np.vstack((feat_train, feat_test))
    target = np.vstack((target_train, target_test))

    print(features, feature_denorm, target)
    y_hat = mylr.predict_(features)
    y_hat = np.where(y_hat >= 0.5, 1, 0)

    accuracy_score = accuracy_score_(target, y_hat)
    print("Accuracy score: {} %".format(accuracy_score * 100))

    colors = np.where(
        y_hat == target,
        np.where(
            y_hat == 1,
            'green',    # True positive
            'blue'      # True negative
        ),
        np.where(
            y_hat == 1,
            'orange',   # False positive
            'red'       # False negative
        )
    )

    fig.suptitle('Logistic regression')

    for i in range(3):
        index = i if i != 2 else -1

        ax[i].scatter(
            feature_denorm[:, index],
            feature_denorm[:, index + 1],
            c=colors.flatten(),
            marker='o',
            alpha=0.5,
            edgecolors='none'
        )

        ax[i].set_xlabel(features_pairs[i][0])
        ax[i].set_ylabel(features_pairs[i][1])
        ax[i].set_title(f'{features_pairs[i][1]} vs {features_pairs[i][0]}')

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

    # Plot a 3D scatter plot with the dataset and the final prediction
    # of the model. The points must be colored following the real class
    # of the citizen.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        feature_denorm[:, 0],
        feature_denorm[:, 1],
        feature_denorm[:, 2],
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