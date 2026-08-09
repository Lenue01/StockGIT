"""Central knobs for the whole pipeline. Edit this file instead of hunting
through data/model/train/sample code for hardcoded values."""

# --- Data ---
TICKERS = [
    "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "AMD", "META", "GOOGL", "GOOG", "NFLX",
    "PLTR", "INTC", "BAC", "F", "PLUG", "SOFI", "NIO", "AAL", "MARA", "RIOT",
    "JPM", "BABA", "VALE", "RIG", "C", "PFE", "WFC", "XOM", "CVX", "LCID",
    "PYPL", "HOOD", "RIVN", "SMR", "DKNG", "COIN", "CLSK", "AUPH", "NOK", "SNAP",
    "UBER", "LYFT", "KVUE", "CSCO", "T", "VZ", "CMCSA", "DIS", "KO", "PEP",
    "WMT", "COST", "HD", "PG", "JNJ", "UNH", "LLY", "ABBV", "MRK", "BMY",
    "GILD", "AMGN", "BA", "GE", "CAT", "DE", "HON", "MMM", "LMT", "RTX",
    "DAL", "UAL", "LUV", "CCL", "RCL", "NCLH", "MGM", "WYNN", "PENN", "SCHW",
    "GS", "MS", "AXP", "V", "MA", "AFRM", "UPST", "SOUN", "BBAI",
    "AI", "RKLB", "ASTS", "HUT", "CIFR", "IREN", "BTBT", "MSTR", "MU"
]
 # add more tickers here to grow the dataset, e.g. ['OKLO', 'NVDA', 'SOFI']

PERIOD = '7d'        # yfinance's max lookback for 1-minute candles
INTERVAL = '1m'
WINDOW_SIZE = 60      # minutes per training window
VAL_DAYS = 1          # most recent trading day(s) per ticker held out for validation

# --- Model (Transformer denoiser) ---
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# --- Training ---
SEED = 0                # fixes weight init + masking/shuffling randomness so runs are comparable
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 20        # linear LR warmup before cosine decay kicks in
CHECKPOINT_PATH = 'denoiser.pt'

# --- Sampling / evaluation ---
FORECAST_HORIZON = 10  # trailing minutes to forecast at inference
SAMPLING_STEPS = 50     # progressive-unmasking steps used to fill in the horizon

# --- Training visualization ---
SNAPSHOT_DIR = 'snapshots'  # train.py drops one .npz here per epoch for visualize_training.py to render
SNAPSHOT_EVERY = 1           # epochs between snapshots
