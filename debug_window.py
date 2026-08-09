import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf

import config
from get_stock_info import StockInfo
from windows import normalize_window

TICKER = config.TICKERS[0]
CHART_PATH = 'debug_candles.png'

UP_COLOR = '#0ca30c'
DOWN_COLOR = '#d03b3b'


def plot_window(window_df, path=CHART_PATH):
    market_colors = mpf.make_marketcolors(
        up=UP_COLOR, down=DOWN_COLOR, edge='inherit', wick='inherit', volume='inherit',
    )
    style = mpf.make_mpf_style(
        marketcolors=market_colors,
        gridcolor='#e1e0d9',
        gridstyle='-',
        facecolor='#fcfcfb',
        figcolor='#f9f9f7',
        edgecolor='#c3c2b7',
        rc={
            'axes.labelcolor': '#0b0b0b',
            'text.color': '#0b0b0b',
            'xtick.color': '#52514e',
            'ytick.color': '#52514e',
        },
    )
    mpf.plot(
        window_df,
        type='candle',
        volume=True,
        style=style,
        title=f'{TICKER} — {config.WINDOW_SIZE}-minute window',
        ylabel='Price ($)',
        ylabel_lower='Volume',
        savefig=dict(fname=path, dpi=150, bbox_inches='tight'),
    )
    print(f'Saved chart to {path}')


def main():
    stock_info = StockInfo(TICKER)
    candles_by_day = stock_info.download_intraday_candles(period='7d', interval='1m')

    date, day_df = next(iter(candles_by_day.items()))
    window_df = day_df[0:config.WINDOW_SIZE]
    raw = window_df.to_numpy()
    normalized = normalize_window(raw)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    np.set_printoptions(suppress=True, precision=4)

    print(f'Day: {date}')
    print(f'Window shape: {window_df.shape}')
    print()
    print('Raw window (Open, High, Low, Close, Volume):')
    print(window_df)
    print()
    print('Normalized window (feeds the model):')
    print(pd.DataFrame(normalized, index=window_df.index,
                        columns=['Open', 'High', 'Low', 'Close', 'Volume']))

    plot_window(window_df)


if __name__ == '__main__':
    main()
