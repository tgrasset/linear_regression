import numpy as np
import pandas as pd
from sys import argv
import os

def calculateCost(km, prices, theta0, theta1):
    """
    Mean Squared Error:
    To get a good idea of how close we are to the dataset, we take the square of
    the difference between the real price and the estimate price with the chosen
    parameters and get the average value which will be the "cost" 
    """
    m = len(prices)
    cost = (1 / (m)) * np.sum((prices - (theta0 + theta1 * km))**2)
    return cost

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

def calculatePrecision(km, prices, theta0, theta1):
    """
    Calculates MSE, RMSE and Coefficient of determination. These values give an idea
    of the performance of our model. The coefficient of determination is between 0 and 1,
    a higher value indicating a better fit of our regression line to the original data.
    """
    mse = calculateCost(km, prices, theta0, theta1)
    sqrtMse = np.sqrt(mse)
    mean = sum(prices) / len(prices)
    r2 = 1 - mse/(sum((prices - mean)**2) / len(prices))
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {sqrtMse}")
    print(f"Coefficient of determination: {r2}")


def main():
    try:
        km, prices = readData('data.csv')
        if os.path.isfile('thetas.csv') and os.access('thetas.csv', os.R_OK):
            data = pd.read_csv('thetas.csv')
            assert data is not None, "Something is wrong with the thetas.csv file."
            if not data.empty and 'theta0' in data.columns and 'theta1' in data.columns:     
                theta0 = float(data.loc[0, 'theta0'])
                theta1 = float(data.loc[0, 'theta1'])
            else:
                print("Something is wrong with the thetas.csv file.")
                return 1
        else:
            theta0 = 0
            theta1 = 0
        print("My linear regression:")
        calculatePrecision(km, prices, theta0, theta1)
        # pf = np.polyfit(km, prices, 1)
        # print("\nnumpy.polyfit's linear regression:")
        # calculatePrecision(km, prices, pf[1], pf[0])
        
    
    except (Exception, AssertionError) as err:
        print("Error: ", err)
        return 1

if __name__ == "__main__":
    main()