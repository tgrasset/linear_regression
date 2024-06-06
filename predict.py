import os
import pandas as pd

def main():
    try:
        if os.path.isfile('thetas.csv') and os.access('thetas.csv', os.R_OK):
            data = pd.read_csv('thetas.csv')
            assert data is not None, "Something is wrong with the thetas.csv file."
            if not data.empty and 'theta0' in data.columns and 'theta1' in data.columns:
                try:       
                    theta0 = float(data.loc[0, 'theta0'])
                    theta1 = float(data.loc[0, 'theta1'])
                except ValueError:
                    print("Wrong values for theta0 and/or theta1.")
                    return 1
            else:
                print("Something is wrong with the thetas.csv file.")
                return 1
        else:
            theta0 = 0
            theta1 = 0
        while True:
            user_input = input("Please enter a number of kilometers: ")
            try:
                value = float(user_input)
                if value < 0:
                    print("Enter a positive number.")
                else:
                    break
            except ValueError:
                print("Invalid input. Please enter a valid float.")
        print("Estimated selling price for a car with a mileage of ", value, " km : ", theta0 + value * theta1)
        return 0
    
    except (AssertionError, Exception) as err:
        print("Error: ", err)
        return 1
        
if __name__ == "__main__":
    main()