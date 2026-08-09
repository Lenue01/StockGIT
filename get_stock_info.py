import pandas as pd
import yfinance as yf


class StockInfo:
    """Thin wrapper around yfinance for one ticker's intraday candles."""

    def __init__(self, ticker):
        self.ticker = ticker

    def download_intraday_candles(self, period='7d', interval='1m'):
        # yfinance only serves recent history at 1-minute resolution --
        # '7d' is close to the max lookback it allows for interval='1m'.
        data = yf.download(self.ticker, period=period, interval=interval)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if data.empty:
            print('No intraday data was downloaded. Try a more recent period or check your network/proxy settings.')
            return None
        elif missing_cols:
            print(f'Missing columns in intraday data: {missing_cols}')
            print('Available columns:', list(data.columns))
            return None

        data.index = pd.to_datetime(data.index)
        data = data[required_cols]
        data['date'] = data.index.date

        candles_by_day = {
            date: group[required_cols].copy()
            for date, group in data.groupby('date')
        }

        return candles_by_day
