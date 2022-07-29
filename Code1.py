import pandas as pd
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



df_close_apple.ta.ema(close ='Close',length =10,append =True)

df_close_apple = df_close_apple.iloc[10:]
df_close_apple.tail(1000).plot(kind='line')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Apple')

df_close_google.ta.ema(close ='Close',length =10,append =True)

df_close_google = df_close_google.iloc[10:]
df_close_google.tail(1000).plot(kind='line')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Google')

plt.show()

#Google linear reg model
def google_model():
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_close_google[['EMA_10']], df_close_google[['Close']], test_size = 0.1,random_state = 101)

    Gmodel = LinearRegression(fit_intercept=False).fit(Xtrain,ytrain)
    Gpred = Gmodel.predict(Xtest)
    Gtrain = Gmodel.predict(Xtrain)
    print("Google Model Coefficients:", Gmodel.coef_)
    print("Mean Absolute Error for Testing data:", mean_absolute_error(ytest, Gpred))
    print("Mean Absolute Error for Training data:", mean_absolute_error(ytrain, Gtrain))
    print("Coefficient of Determination:", r2_score(ytest, Gpred))

    yG = [(Gmodel.coef_[0][0])*i for i in Xtest['EMA_10']]

    xm,Ym = zip(*sorted(zip(Xtest['EMA_10'],yG)))
    xt, yt = zip(*sorted(zip(Xtest['EMA_10'],ytest['Close'])))

    plt.scatter(xt,yt, 5,label = 'Actual', color ='red')
    plt.plot(xm, Ym, '-', label = 'Model')
    plt.xlabel('EMA')
    plt.ylabel('Close')
    plt.title('Google Model')
    plt.legend()
    plt.show()

#Apple Model
def apple_model():
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_close_apple[['EMA_10']], df_close_apple[['Close']], test_size = 0.1, random_state = 101)

    Amodel = LinearRegression(fit_intercept=False).fit(Xtrain,ytrain)
    Atrain = Amodel.predict(Xtrain)
    Apred = Amodel.predict(Xtest)
    print("Apple Model Coefficients:", Amodel.coef_)
    print("Mean Absolute Error for Testing Data:", mean_absolute_error(ytest, Apred))
    print("Mean Absolute Error for Training data:", mean_absolute_error(ytrain, Atrain))
    print("Coefficient of Determination:", r2_score(ytest, Apred))

    yG = [(Amodel.coef_[0][0])*i for i in Xtest['EMA_10']]

    xm,Ym = zip(*sorted(zip(Xtest['EMA_10'],yG)))
    xt, yt = zip(*sorted(zip(Xtest['EMA_10'],ytest['Close'])))

    plt.scatter(xt,yt, 5,label = 'Actual', color ='red')
    plt.plot(xm, Ym, '-', label = 'Model')
    plt.xlabel('EMA')
    plt.ylabel('Close')
    plt.title('Apple Model')
    plt.legend()
    plt.show()
google_model()
apple_model()