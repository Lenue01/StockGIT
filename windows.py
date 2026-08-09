"""Turns raw per-day candles into normalized sliding windows, across
however many tickers are listed in config.TICKERS."""

import hashlib
from pathlib import Path

import numpy as np

import config
from get_stock_info import StockInfo

CACHE_DIR = Path('.cache')
CACHE_SCHEMA_VERSION = 2  # bump whenever the channel layout changes, so stale caches can't silently load

# Channel layout: [Open, High, Low, Close, Volume, sin(session_pos), cos(session_pos)].
# The first NUM_PRICE_VOLUME_FEATURES are what the model is asked to reconstruct;
# the trailing NUM_TIME_FEATURES are always known (even for "masked" future
# timesteps -- you know what time it'll be) so they're never masked and never
# part of the reconstruction target.
NUM_PRICE_VOLUME_FEATURES = 5
NUM_TIME_FEATURES = 2
NUM_FEATURES = NUM_PRICE_VOLUME_FEATURES + NUM_TIME_FEATURES

SESSION_MINUTES = 390  # regular NYSE session length (9:30-16:00 ET)


def normalize_window(window):
    # window: (WINDOW_SIZE, 5) array of [Open, High, Low, Close, Volume]
    # Price columns are converted to percent change from the window's first
    # Open so patterns look the same regardless of the stock's price level.
    open_, high, low, close, volume = window.T
    ref = open_[0]

    norm_open = open_ / ref - 1
    norm_high = high / ref - 1
    norm_low = low / ref - 1
    norm_close = close / ref - 1

    # Volume is log-scaled first (it's heavily right-skewed), then
    # z-scored against this window's own mean/std so the scale is
    # comparable across windows with very different trading activity.
    log_volume = np.log1p(volume)
    norm_volume = (log_volume - log_volume.mean()) / (log_volume.std() + 1e-8)

    return np.stack([norm_open, norm_high, norm_low, norm_close, norm_volume], axis=1)


def session_time_features(day_df):
    # Position within the trading session, as (sin, cos) over a HALF cycle
    # (angle in [0, pi], not [0, 2*pi]) so market open and market close map
    # to different points -- a full cycle would alias them together, which
    # is wrong since they're very different volatility regimes, not adjacent
    # ones. sin peaks at midday (symmetric, echoes the open/close vol
    # U-shape); cos decreases monotonically across the session.
    minutes_elapsed = (day_df.index - day_df.index[0]).total_seconds() / 60.0
    frac = np.clip(minutes_elapsed / SESSION_MINUTES, 0.0, 1.0)
    angle = frac * np.pi
    return np.stack([np.sin(angle), np.cos(angle)], axis=1)


def windows_for_ticker(ticker):
    """Downloads one ticker's candles and slides a window within each
    trading day, so a window never spans the overnight gap between one
    day's close and the next day's open."""
    candles_by_day = StockInfo(ticker).download_intraday_candles(period=config.PERIOD, interval=config.INTERVAL)

    windows, dates = [], []
    for date, day_df in candles_by_day.items():
        time_features = session_time_features(day_df)
        for i in range(len(day_df) - config.WINDOW_SIZE + 1):
            window_data = day_df[i:i + config.WINDOW_SIZE].to_numpy()
            price_volume = normalize_window(window_data)
            time_slice = time_features[i:i + config.WINDOW_SIZE]
            windows.append(np.concatenate([price_volume, time_slice], axis=1))
            dates.append(str(date))

    return windows, dates


def chronological_split(windows, dates, tickers, val_days=None):
    """Holds out each ticker's most recent `val_days` trading day(s), so
    validation is always future-relative-to-train and every ticker shows
    up on both sides of the split (not just whichever has the latest data)."""
    val_days = val_days if val_days is not None else config.VAL_DAYS

    val_mask = np.zeros(len(windows), dtype=bool)
    for ticker in np.unique(tickers):
        idx = np.where(tickers == ticker)[0]
        ticker_days = np.unique(dates[idx])
        val_day_set = ticker_days[-val_days:]
        val_mask[idx] = np.isin(dates[idx], val_day_set)

    return windows[~val_mask], windows[val_mask], tickers[~val_mask], tickers[val_mask]


def _cache_path(tickers):
    key = f'{tickers}|{config.PERIOD}|{config.INTERVAL}|{config.WINDOW_SIZE}|v{CACHE_SCHEMA_VERSION}'
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f'windows_{digest}.npz'


def build_dataset(tickers=None, use_cache=True):
    """Downloads + windows every ticker in the list (config.TICKERS by
    default) and concatenates them into single arrays. To grow the
    dataset, just add tickers to config.TICKERS -- no other code changes.

    Caches the result to .cache/ keyed on (tickers, period, interval,
    window_size) -- downloading dozens of tickers from yfinance on every
    single run gets slow fast. Delete .cache/ if you ever suspect it's stale."""
    tickers = tickers if tickers is not None else config.TICKERS
    cache_path = _cache_path(tickers)

    if use_cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return data['windows'], data['dates'], data['tickers']

    all_windows, all_dates, all_tickers = [], [], []
    for ticker in tickers:
        windows, dates = windows_for_ticker(ticker)
        all_windows.extend(windows)
        all_dates.extend(dates)
        all_tickers.extend([ticker] * len(windows))

    windows_arr = np.array(all_windows)
    dates_arr = np.array(all_dates)
    tickers_arr = np.array(all_tickers)

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(cache_path, windows=windows_arr, dates=dates_arr, tickers=tickers_arr)

    return windows_arr, dates_arr, tickers_arr


if __name__ == '__main__':
    windows, dates, tickers = build_dataset()
    print(f'tickers: {list(np.unique(tickers))}')
    print(f'windows shape: {windows.shape}')
