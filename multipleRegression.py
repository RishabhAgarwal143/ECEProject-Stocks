from re import I
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime as dt

#Takes a list of columns to extract from the data and a number of values to use for the EMA smoothing.
#Returns the data with EMA columns appended in order of apple then google, and the EMA name
def dataCreation(columns_extract, ema_n):
    df_apple = pd.read_csv("AAPL.csv")
    df_google = pd.read_csv("GOOG.csv")

    apple = df_apple[["Date"]]
    google = df_google[["Date"]]

    apple_d, timeNull = deltaDays(apple)
    google_d, timeNull = deltaDays(google)

    apple_d = pd.DataFrame(apple_d).rename(columns={0:"Days"})
    google_d = pd.DataFrame(google_d).rename(columns={0:"Days"})

    apple = apple.join(apple_d)
    google = google.join(google_d)


    for column in columns_extract:
        apple = apple.join(df_apple[[column]])
        google = google.join(df_google[[column]])

    for column in columns_extract:
        ema = apple[[column]].ewm(span = ema_n, adjust = False).mean()
        ema.rename(columns = {column:f"{column}_EMA_{ema_n}"}, inplace = True)
        apple = pd.concat([apple, ema], axis = 1)
        
        ema = google[[column]].ewm(span = ema_n, adjust = False).mean()
        ema.rename(columns = {column:f"{column}_EMA_{ema_n}"}).mean()
        google = pd.concat([google, ema], axis = 1)

    apple.set_index(pd.DatetimeIndex(df_apple['Date']), inplace = True)
    google.set_index(pd.DatetimeIndex(df_google['Date']), inplace = True)

    apple = apple.iloc[ema_n:]
    google = google.iloc[ema_n:]

    return apple, google, f'EMA_{ema_n}'


#Creates a linear regression model, takes the dates and 
def linearReg(X, Y):
    regr = LinearRegression(fit_intercept = False)
    regr.fit(X, Y)
    return regr


def deltaDays(dates):
    dates = np.array(dates["Date"])
    datesNum = np.size(dates)
    timeNull = dt.strptime(dates[0], "%Y-%m-%d")

    #Fills empty list with converted datetime objects to find delta with timeNull.
    i = 0
    timeObsArr = np.zeros(datesNum).reshape([datesNum, 1])
    for date in dates:
        date1 = dt.strptime(date, "%Y-%m-%d")
        delta = date1 - timeNull

        diffDays = delta.days
        timeObsArr[i][0] = diffDays
        i += 1
    
    return(timeObsArr, timeNull) 

def main():
    columns = ["Volume", "Open", "Adj Close"] #type what columns of data should be returned
    apple, google, ema_title = dataCreation(columns, 500) # return data and ema_XXX title (XXX = 2nd var of dataCreation)

    output = apple[["Adj Close"]]
    input = apple[[f"Days", f"Volume_{ema_title}", f"Open_{ema_title}"]]

    deprivation = 30 #days of data deprived from model
    dep_test_output = output.iloc[-deprivation:]
    dep_test_input = input.iloc[-deprivation:]
    output = output.iloc[:-deprivation]
    input = input.iloc[:-deprivation]

    input_train, input_test, output_train, output_test = train_test_split(input, output)
    regr = linearReg(input_train, output_train)
    print(f"Coefficients: {regr.coef_}")

    output_predictions = regr.predict(input_test)
    mse_test = mean_squared_error(output_test, output_predictions)
    print("Mean Squared Error:", mse_test)

    # Uses the last available volume and open measurements while days continues
    volume_perm = input[f"Volume_{ema_title}"][-1]
    open_perm = input[f"Open_{ema_title}"][-1]
    to_perm = {"Volume":volume_perm, "Open":open_perm}

    columns.append("Days")
    arr = np.zeros(deprivation)
    j = 0
    for day in dep_test_input["Days"]:

        product1 = regr.coef_[0][0] * day
        prod_sum = product1
        i = 0
        for coefficient in regr.coef_[0][1:]:
            product_n = to_perm[columns[i]] * coefficient
            prod_sum = prod_sum + product_n
            i+=1
        arr[j] = prod_sum
        j+= 1

    print(arr)
    mse_test = mean_squared_error(dep_test_output, arr)
    print("Mean Squared Error:", mse_test)

if __name__ == "__main__":
    main()