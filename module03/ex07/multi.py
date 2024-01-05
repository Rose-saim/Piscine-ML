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

def split_dataframes(features_dataframe: pd.DataFrame,
                         target_dataframe: pd.DataFrame,
                         ratio: float):
        try:
            if ratio < 0 or ratio > 1:
                print("Error: ratio must be between 0 and 1")
                exit(1)
            elif features_dataframe.shape[0] != target_dataframe.shape[0]:
                print("Error: The dataset and the target don't" +
                      " have the same number of elements")
                exit(1)

            complete_df = pd.concat([features_dataframe, target_dataframe],
                                        axis=1, sort=False)

            # We shuffle the dataframe
            complete_df = complete_df.sample(frac=1).reset_index(drop=True)

            # We split the dataframe into two dataframes
            # according to the ratio
            split_index = int(complete_df.shape[0] * ratio)
            features_train = complete_df.iloc[:split_index, :-1].to_numpy()
            features_test = complete_df.iloc[split_index:, :-1].to_numpy()
            target_train = complete_df.iloc[:split_index, -1:].to_numpy()
            target_test = complete_df.iloc[split_index:, -1:].to_numpy()

            return features_train, features_test, target_train, target_test

        except Exception:
            print("Error: Can't split the dataset")
            exit(1)

def main():
    feature = ["weight", "height", "bone_density"]
    features = read_csv("../solar_system_census.csv")[feature]
    target = ["Origin"]
    targets = read_csv("../solar_system_census_planets.csv")[target]

    mylr = ML([0], max_iter=150000, alpha=0.1)
    code = zip_code()


    features_train, features_test, target_train, target_test = split_dataframes(features, targets, ratio=0.2)    
    norm_train, min_v, max_v = mylr.normalize_train(features_train)
    norm_test = mylr.normalize_test(features_test, min_v, max_v)

    trained_thetas = []

    for curr_train in range(4):
        print("Training model {} / 4 ...".format(curr_train + 1))

        theta = np.zeros((norm_train.shape[1] + 1, 1))
        logistic_reg = ML(theta, 1, 50_000)

        y: np.ndarray = target_train.copy()

        y = np.where(y == curr_train, 1, 0)

        theta = logistic_reg.fit_(norm_train, y)

        trained_thetas.append(theta)

    # Predict for each example the class according to each classifiers
    # and select the one with the highest output probability.

     # Predict for each example the class according to each classifiers
    # and select the one with the highest output probability.

    normalized_features = np.concatenate(
        (norm_test, norm_train),
        axis=0)
    denormalized_features = np.concatenate(
        (features_test, features_train),
        axis=0)
    total_target = np.concatenate((target_test, target_train), axis=0)
    y_predictions = np.empty((normalized_features.shape[0], 1))

    for i in range(normalized_features.shape[0]):
        y_proba = np.empty((4, 1))
        for curr_test in range(4):
            logistic_reg.theta = trained_thetas[curr_test]
            y_proba[curr_test] = logistic_reg.predict_(normalized_features[i].reshape(1, normalized_features.shape[1]))
        y_predictions[i] = y_proba.argmax()
    
    accuracy_score = logistic_reg.accuracy_score_(total_target, y_predictions)
    print("Accuracy score: {} %".format(accuracy_score * 100))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    colors = np.where(
        y_predictions < 2,
        np.where(
            y_predictions == 0,
            'green',    # 0
            'blue'      # 1
        ),
        np.where(
            y_predictions == 2,
            'orange',   # 2
            'red'       # 3
        )
    )

    real_colors = np.where(
        total_target < 2,
        np.where(
            total_target == 0,
            'green',    # 0
            'blue'      # 1
        ),
        np.where(
            total_target == 2,
            'orange',   # 2
            'red'       # 3
        )
    )

    features_pairs = [
        ('weight', 'height'),
        ('weight', 'bone_density'),
        ('height', 'bone_density')
    ]


    for i in range(3):
        index = i if i != 2 else -1

        ax[i].scatter(
            denormalized_features[:, index],
            denormalized_features[:, index+1],
            c=colors.flatten(),
            marker='o',
            alpha=0.5,
            edgecolors=real_colors.flatten()
        )
        ax[i].set_xlabel(features_pairs[i][0])
        ax[i].set_ylabel(features_pairs[i][1])
        ax[i].set_title(f'{features_pairs[i][1]} vs {features_pairs[i][0]}')


    fig.legend(
        handles=[
            patches.Patch(
                          color='green',
                          label='The flying cities of Venus'),
            patches.Patch(
                          color='blue',
                          label='United Nations of Earth'),
            patches.Patch(
                          color='orange',
                          label='Mars Republic'),
            patches.Patch(
                          color='red',
                          label="The Asteroid's Belt colonies"),
        ],
        loc='lower center', ncol=4, fontsize='small',
    )

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        denormalized_features[:, 0],
        denormalized_features[:, 1],
        denormalized_features[:, 2],
        c=colors.flatten(),
        marker='o',
        alpha=0.5,
        edgecolors=real_colors.flatten()

    )

    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title('bone_density vs height vs weight')

    fig.legend(
        handles=[
            patches.Patch(
                          color='green',
                          label='The flying cities of Venus'),
            patches.Patch(
                          color='blue',
                          label='United Nations of Earth'),
            patches.Patch(
                          color='orange',
                          label='Mars Republic'),
            patches.Patch(
                          color='red',
                          label="The Asteroid's Belt colonies"),
        ],
        loc='lower center', ncol=4, fontsize='small',
    )


    plt.show()

if __name__ == "__main__":
    main()
    


