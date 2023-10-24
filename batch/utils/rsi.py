# https://codingandfun.com/rsi-momentum-strategies-using-python/
import requests
import pandas as pd
import numpy as np

apiKey = 'f57bdcaa7d140c9de35806d47fbd2f91'
stock = 'TSM'
stockPrices = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?apikey={apiKey}').json()
stockPrices = stockPrices['historical'][0:1200]
stockPrices = pd.DataFrame(stockPrices)
stockPrices = stockPrices.set_index('date')
stockPrices = stockPrices.iloc[::-1]

stockPrices['return'] = np.log(stockPrices['close'] / stockPrices['close'].shift(1))
stockPrices['movement'] = stockPrices['close'] - stockPrices['close'].shift(1)

stockPrices['up'] = np.where((stockPrices['movement'] > 0), stockPrices['movement'], 0)
stockPrices['down'] = np.where((stockPrices['movement'] < 0), stockPrices['movement'], 0)

window_length = 14
up = stockPrices['up'].rolling(window_length).mean()
down = stockPrices['down'].abs().rolling(window_length).mean()

RS = up / down

RSI = 100.0 - (100.0 / (1.0 + RS))
RSI = RSI.rename('RSI')

# print(RSI)

new = pd.merge(stockPrices, RSI, left_index=True, right_index=True)
new['long'] = np.where((new['RSI'] < 30), 1, np.nan)
new['long'] = np.where((new['RSI'] > 70), 0, new['long'])
new['long'].ffill(inplace=True)
new['gain_loss'] = new['long'].shift(1) * new['return']
new['total'] = new['gain_loss'].cumsum()
print(new.head())
print(new.tail())