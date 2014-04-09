import requests
import pandas as pd


if __name__ == '__main__':
    url = 'http://ideaquant.cs.byu.edu/allhistories?n=20'
    r = requests.get(url)

    symbols = ['DIS', 'MSFT', 'YHOO', 'MU',
               'XOM', 'GE', 'CVX',
               'C', 'BAC', 'JPM']

    data = pd.DataFrame(r.json()).transpose().loc[:, symbols]
    asks = data.apply(lambda x: x.apply(lambda y: y['ask']))
    bids = data.apply(lambda x: x.apply(lambda y: y['bid']))

    asks.to_csv('askdata.csv', parse_dates=True)
    bids.to_csv('biddata.csv', parse_dates=True)
