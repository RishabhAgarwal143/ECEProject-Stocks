import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score 

class Data:
    def __init__(file,data):
        file.date = data['Date']
        file.open = data['Open']
        file.high = data['High']
        file.low = data['Low']
        file.close = data['Close']
        file.adj_close = data['Adj Close']
        file.volume = data['Volume']
        

def google():
    df_google = pd.read_csv("GOOG.csv")
    googledata = Data(df_google)
    df_google = df_google[['Close']]

    predictime = 30
    df_google['Prediction'] = df_google[['Close']].shift(-predictime)
    X = np.array(df_google.drop(['Prediction'],1))
    X = X[:-predictime]

    y = np.array(df_google['Prediction'])
    y = y[:-predictime]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    Gmodel = LinearRegression().fit(x_train, y_train)
    Gpred = Gmodel.predict(x_test)
    Gtrain = Gmodel.predict(x_train)
    print("Google Model Coefficients:", Gmodel.coef_)
    print("Mean Absolute Error for Testing data:", mean_absolute_error(y_test, Gpred))
    print("Mean Absolute Error for Training data:", mean_absolute_error(y_train, Gtrain))
    print("Coefficient of Determination:", r2_score(y_test, Gpred))

    model_test = Gmodel.predict(x_test)
    xtm, ytm = zip(*sorted(zip(x_test, model_test)))
    x_test, y_test = zip(*sorted(zip(x_test, y_test)))

    plt.scatter(x_test,y_test, 5,label = 'Actual', color ='red')
    plt.plot(xtm, ytm, '-', label = 'Model')
    plt.legend()
    plt.title('Google')
    plt.show()

    #plotting model for data that was NOT given 
    x_predict = np.array(df_google.drop(['Prediction'],1))[-predictime:]
    model_predict = Gmodel.predict(x_predict)
    xm, ym = zip(*sorted(zip(x_predict, model_predict)))
    plt.plot(xm, ym, '-', label = 'Model')
    plt.legend()
    plt.title('Model Prediction for 30 days into the future')
    plt.show()

def apple():
    df_apple = pd.read_csv("AAPL.csv")
    appledata = Data(df_apple)
    df_apple = df_apple[['Close']]

    predictime = 30
    df_apple['Prediction'] = df_apple[['Close']].shift(-predictime)
    X = np.array(df_apple.drop(['Prediction'],1))
    X = X[:-predictime]

    y = np.array(df_apple['Prediction'])
    y = y[:-predictime]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    Amodel = LinearRegression().fit(x_train, y_train)
    Apred = Amodel.predict(x_test)
    Atrain = Amodel.predict(x_train)
    print("Google Model Coefficients:", Amodel.coef_)
    print("Mean Absolute Error for Testing data:", mean_absolute_error(y_test, Apred))
    print("Mean Absolute Error for Training data:", mean_absolute_error(y_train, Atrain))
    print("Coefficient of Determination:", r2_score(y_test, Apred))

    model_test = Amodel.predict(x_test)
    xtm, ytm = zip(*sorted(zip(x_test, model_test)))
    x_test, y_test = zip(*sorted(zip(x_test, y_test)))

    plt.scatter(x_test,y_test, 5,label = 'Actual', color ='red')
    plt.plot(xtm, ytm, '-', label = 'Model')
    plt.title('Apple')
    plt.legend()
    plt.show()

    #plotting model for data that was NOT given 
    x_predict = np.array(df_apple.drop(['Prediction'],1))[-predictime:]
    model_predict = Amodel.predict(x_predict)
    xm, ym = zip(*sorted(zip(x_predict, model_predict)))
    plt.plot(xm, ym, '-', label = 'Model')
    plt.legend()
    plt.title('Model Prediction for 30 days into the future')
    plt.show()
google()
apple()
