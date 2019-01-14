import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def estimate_coefficients(X, Y):
    n = X.shape[0]
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

# Deviation and cross deviation across X
    CD_XY = np.sum(Y * X - n * mean_Y * mean_X)
    D_XX = np.sum(X * X - n * mean_X * mean_X)

# Regression coefficients
    b1 = CD_XY / D_XX
    b0 = mean_Y - b1 * mean_X
    return(b0, b1)


def plot(X, Y, b):
    # Data plot
    print(X, Y)
    plt.scatter(X, Y)

    y = b[0] + b[1] * X
    # Regression line plot
    plt.plot(X, y, color="r")

    plt.xlabel("Size")
    plt.ylabel("Cost")
    plt.show()


def main():
    # Reading the data
    my_data = pd.read_csv("home.csv", names=["size", "bedroom", "price"])
    # Normalising the my_data
    #my_data = (my_data - my_data.mean()) / my_data.std()

    # Creating matrices now
    X = my_data.iloc[:, 0:1]

    X = preprocessing.scale(X)
    #ones = np.ones((X.shape[0], 1))

    # = np.concatenate((ones, X), axis=1)
    Y = my_data.iloc[:, 2:3].values  # convert pandas to numpy
    Y = preprocessing.scale(Y)
    # Estimating coefficients
    b = estimate_coefficients(X, Y)
    print("Estimated values are: \nb0 = {} \nb1 = {}".format(b[0], b[1]))
    plot(X, Y, b)


if __name__ == "__main__":
    main()
