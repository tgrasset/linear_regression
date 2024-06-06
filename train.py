import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

def calculateCost(estimate, prices):
    """
    Mean Squared Error:
    To get a good idea of how close we are to the dataset, we take the square of
    the difference between the real price and the estimate price with the current
    parameters and get the average value which will be the "cost" for the current
    parameters
    """
    m = len(prices)
    cost = (1 / (2*m)) * np.sum((estimate - prices)**2)
    return cost

def tryParams(km, tmp0, tmp1):
    """
    This function applies the current parameters to all the mileage data items and
    returns the resulting array
    """
    return tmp0 + tmp1 * km

def updateParams(estimate, km, prices, tmp0, tmp1, learningRate):
    """
    theta0 and theta1 are updated according to the formulas given in the subject
    according to the gradient descent rule. The new values are returned.
    """
    m = len(prices)
    tmp0 = tmp0 - (learningRate * (1 / m) * np.sum(estimate - prices))
    tmp1 = tmp1 - (learningRate * (1 / m) * np.sum((estimate - prices) * km))
    return tmp0, tmp1


def linearRegression(normKm, prices, kmMean, kmStd):
    """
    This function implements the gradient descent algorithm to find the best
    parameters for our model, updating the two arguments given to it 10000 times
    """
    tmp0 = 0
    tmp1 = 0
    epochs = 10000
    learningRate = 0.001
    costs = []
    x = np.linspace(min(normKm), max(normKm), 100)

    for i in range(epochs):
        estimate = tryParams(normKm, tmp0, tmp1)
        costs.append(calculateCost(estimate, prices))
        tmp0, tmp1 = updateParams(estimate, normKm, prices, tmp0, tmp1, learningRate)
        if i % 500 == 0:
            deNormX = x * kmStd + kmMean
            y = tmp0 + tmp1 * x
            plt.plot(deNormX, y, label=f'Epoch {i}')
    deNormX = x * kmStd + kmMean
    y = tmp0 + tmp1 * x
    plt.plot(deNormX, y, 'r', label='Final Regression Line')
    plt.legend()
    plt.title('Cars selling price vs Mileage')
    plt.xlabel('km')
    plt.ylabel('price')
    return tmp0, tmp1

def readData(filePath):
    """
    Gets data from the file and returns two arrays corresponding to mileage and price
    """
    data = pd.read_csv(filePath)
    assert data is not None, "The data set is not a proper csv file"
    assert data.shape[1] == 2, "The data set must contain exactly two columns"
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    return x, y

def normalizeData(x):
    """
    Since kilometers have such a big scale, we normalize the data, reducing values to
    small values around 0 to minimize the error during gradient descent math
    """
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, mean, std


def main():
    try:
        assert len(argv) == 2, "The script needs a data file as argument"
        km, prices = readData(argv[1])
        normKm, kmMean, kmStd = normalizeData(km)
        normTheta0, normTheta1= linearRegression(normKm, prices, kmMean, kmStd)
        # following two lines convert theta1 and theta0 to their un-normalized values
        theta1 = normTheta1 / kmStd
        theta0 = normTheta0 - (theta1 * kmMean)
        print("Training complete !")
        print(f"Final values ---> theta 0 : {theta0}, theta1 : {theta1}")
        df = pd.DataFrame({
            "theta0": [theta0],
            "theta1": [theta1]
        })
        df.to_csv("thetas.csv", index=False)
        print("Final theta values saved to thetas.csv")
        plt.scatter(km, prices)
        plt.show()
        return 0
    
    except AssertionError as err:
        print("Error: ", err)
        return 1

    except Exception as err:
        print("Error: ", err)
        return 1

if __name__ == "__main__":
    main()