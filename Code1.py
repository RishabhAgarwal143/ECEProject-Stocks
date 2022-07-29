import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score 
import datetime
import warnings
warnings.filterwarnings("ignore")

class Data:
    def __init__(file,data):
        file.date = data['Date']
        file.open = data['Open']
        file.high = data['High']
        file.low = data['Low']
        file.close = data['Close']
        file.adj_close = data['Adj Close']
        file.volume = data['Volume']
        

df_apple = pd.read_csv("AAPL.csv")
df_google = pd.read_csv("GOOG.csv")

appledata = Data(df_apple)
googledata = Data(df_google)

df_close_apple = df_apple[['Close']]
df_close_google = df_google[['Close']]

df_close_apple.set_index(pd.DatetimeIndex(df_apple['Date']),inplace = True)
df_close_google.set_index(pd.DatetimeIndex(df_google['Date']),inplace=True)


df_google.plot('Date','Close',color = 'red')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Google')

df_apple.plot('Date','Close',color = 'red')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Apple')

no_of_days_considered = 10

df_close_apple.ta.ema(close ='Close',length =no_of_days_considered,append =True)

df_close_apple = df_close_apple.iloc[no_of_days_considered:]
df_close_apple.tail(1000).plot(kind='line')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Apple')

df_close_google.ta.ema(close ='Close',length =no_of_days_considered,append =True)

df_close_google = df_close_google.iloc[10:]
df_close_google.tail(1000).plot(kind='line')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Google')

plt.show()

#Google linear reg model
def google_model():

    no_of_days_considered = 10
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_close_google[['EMA_%d'%no_of_days_considered]], df_close_google[['Close']], test_size = 0.1,random_state = 101)

    Gmodel = LinearRegression(fit_intercept=False).fit(Xtrain,ytrain)
    Gpred = Gmodel.predict(Xtest)
    Gtrain = Gmodel.predict(Xtrain)
    print("Google Model Coefficients:", Gmodel.coef_)
    print("Mean Absolute Error for Testing data:", mean_absolute_error(ytest, Gpred))
    print("Mean Absolute Error for Training data:", mean_absolute_error(ytrain, Gtrain))
    print("Coefficient of Determination:", r2_score(ytest, Gpred))

    yG = [(Gmodel.coef_[0][0])*i for i in Xtest['EMA_%d'%no_of_days_considered]]

    xm,Ym = zip(*sorted(zip(Xtest['EMA_%d'%no_of_days_considered],yG)))
    xt, yt = zip(*sorted(zip(Xtest['EMA_%d'%no_of_days_considered],ytest['Close'])))

    plt.scatter(xt,yt, 5,label = 'Actual', color ='red')
    plt.plot(xm, Ym, '-', label = 'Model')
    plt.xlabel('EMA')
    plt.ylabel('Close')
    plt.title('Google Model')
    plt.legend()
    plt.show()
    return Gmodel

#Apple Model
def apple_model():
    no_of_days_considered = 10
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_close_apple[['EMA_%d'%no_of_days_considered]], df_close_apple[['Close']], test_size = 0.1, random_state = 101)

    Amodel = LinearRegression(fit_intercept=False).fit(Xtrain,ytrain)
    Atrain = Amodel.predict(Xtrain)
    Apred = Amodel.predict(Xtest)
    print("Apple Model Coefficients:", Amodel.coef_)
    print("Mean Absolute Error for Testing Data:", mean_absolute_error(ytest, Apred))
    print("Mean Absolute Error for Training data:", mean_absolute_error(ytrain, Atrain))
    print("Coefficient of Determination:", r2_score(ytest, Apred))

    yG = [(Amodel.coef_[0][0])*i for i in Xtest['EMA_%d'%no_of_days_considered]]

    xm,Ym = zip(*sorted(zip(Xtest['EMA_%d'%no_of_days_considered],yG)))
    xt, yt = zip(*sorted(zip(Xtest['EMA_%d'%no_of_days_considered],ytest['Close'])))

    plt.scatter(xt,yt, 5,label = 'Actual', color ='red')
    plt.plot(xm, Ym, '-', label = 'Model')
    plt.xlabel('EMA')
    plt.ylabel('Close')
    plt.title('Apple Model')
    plt.legend()
    plt.show()

    return Amodel


model_g = google_model()
model_a = apple_model()


day = int(input("Enter a day: "))
month = int(input("Enter a month: "))
year = int(input("Enter a year: "))
date_1 = datetime.datetime(year,month,day,hour=0,minute=0,second=0)


try:
    print("The Predicted Stock price for apple is :",model_a.predict(df_close_apple.loc[date_1]['Close'].reshape(-1,1)))
    print("The Predicted Stock price for google is :",model_g.predict(df_close_google.loc[date_1]['Close'].reshape(-1,1)))
except:
    if (date_1 - (datetime.datetime(2022,6,6,0,0,0))) < (datetime.datetime(2022,6,6,0,0,0))-date_1:
        State = True
        while State:
            n_d =date_1 +datetime.timedelta(days=1)
            if n_d in df_close_apple.index:
                print("The Predicted Stock price for apple is :",model_a.predict(df_close_apple.loc[n_d]['Close'].reshape(-1,1)))
                print("The Predicted Stock price for google is :",model_g.predict(df_close_google.loc[n_d]['Close'].reshape(-1,1)))
                State=False


    else:
        new_pred_value_a = model_a.predict(df_close_apple.loc[(datetime.datetime(2022,6,6,0,0,0))]['Close'].reshape(-1,1))
        new_pred_value_g = model_g.predict(df_close_google.loc[(datetime.datetime(2022,6,6,0,0,0))]['Close'].reshape(-1,1))
        
        for i in range((date_1 - (datetime.datetime(2022,6,6,0,0,0))).days):
            new_pred_value_a = model_a.predict(new_pred_value_a.reshape(-1,1))
            new_pred_value_g = model_g.predict(new_pred_value_g.reshape(-1,1))

        print("The Predicted Stock price for apple is :",new_pred_value_a)
        print("The Predicted Stock price for google is :",new_pred_value_g)






    

