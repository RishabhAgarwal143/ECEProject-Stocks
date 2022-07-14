import pandas as pd
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

plt.plot(appledata.date,appledata.close)
plt.show()