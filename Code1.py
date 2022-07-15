import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

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
