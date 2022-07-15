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


df_close = df_apple[['Close']]
df_close.set_index(pd.DatetimeIndex(df_apple['Date']),inplace = True)
appledata = Data(df_apple)
googledata = Data(df_google)

df_apple.plot('Date','Close',color = 'red')


df_close.ta.ema(close ='Close',length =10,append =True)

df_close = df_close.iloc[10:]
df_close.tail(1000).plot(kind='line')

plt.show()



